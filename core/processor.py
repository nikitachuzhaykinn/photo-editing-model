"""ImageProcessor — полное ядро (максимальное качество для фото)."""
import time
import cv2
import numpy as np
import logging

from core.analyzer import analyze_frame
from core.enhancer import (
    denoise, bilateral_denoise, adaptive_gamma, correct_white_balance,
    suppress_color_noise, prevent_overexposure, enhance_saturation,
    add_grain, micro_contrast, enhance_brightness_global, warm_tone,
    final_sharpen, highlight_rolloff,
)
from core.segmenter import FaceSegmenter
from core.retoucher import retouch_skin, process_background
from core.grader import cinematic_grade
from ml.ai_enhance import ZeroDCE
from ml.scunet_denoise import SCUNetDenoise

logger = logging.getLogger(__name__)


def pick_ai_size(w: int, h: int) -> int:
    """Размер длинной стороны для Zero-DCE."""
    longest = max(w, h)
    if longest <= 640:
        return longest
    if longest <= 1280:
        return 640
    if longest <= 2560:
        return 1024
    return 1280


class ImageProcessor:
    """Полное ядро. Один пайплайн — максимальное качество."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.segmenter = FaceSegmenter()
        self.zero_dce = ZeroDCE()
        self.scunet = SCUNetDenoise()
        logger.info(
            f"mode={mode}, Zero-DCE ready={self.zero_dce.is_ready}, "
            f"SCUNet ready={self.scunet.is_ready}"
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        image = self._validate(image)
        h_orig, w_orig = image.shape[:2]

        # 1. АНАЛИЗ
        report = analyze_frame(image)
        mean_bright = report["mean_brightness"]
        diagnosis = report["diagnosis"]
        color_cast = report["color"]["color_cast"]

        print(f"Анализ: яркость={mean_bright:.1f}, диагноз={diagnosis}, "
              f"оттенок={color_cast}, размер={w_orig}×{h_orig}")

        # 2. ОБРАБОТКА
        use_ai = (
            self.mode == "ai"
            or (self.mode == "auto" and mean_bright < 50 and self.zero_dce.is_ready)
        )

        if use_ai:
            # 2.1. PRE-DENOISE
            if mean_bright < 15:
                work = bilateral_denoise(image, d=5, sigma_color=25, sigma_space=25)
                print("  → Pre-denoise bilateral (d=5, σ=25)")
            elif mean_bright < 30:
                work = bilateral_denoise(image, d=3, sigma_color=15, sigma_space=15)
                print("  → Pre-denoise bilateral (d=3, σ=15)")
            else:
                work = image

            # 2.2. Zero-DCE++ — с сохранением пропорций!
            target = pick_ai_size(w_orig, h_orig)
            if target >= max(w_orig, h_orig):
                result = self.zero_dce.enhance(work)
                print(f"  → Zero-DCE (native {w_orig}×{h_orig})")
            else:
                # Пропорциональный ресайз (не квадрат!)
                scale = target / max(w_orig, h_orig)
                target_w = int(w_orig * scale)
                target_h = int(h_orig * scale)
                # Кратность 8 для стабильности ONNX
                target_w = (target_w // 8) * 8
                target_h = (target_h // 8) * 8

                small = cv2.resize(work, (target_w, target_h),
                                   interpolation=cv2.INTER_AREA)
                result_small = self.zero_dce.enhance(small)
                result = cv2.resize(result_small, (w_orig, h_orig),
                                    interpolation=cv2.INTER_LANCZOS4)
                print(f"  → Zero-DCE ({target_w}×{target_h} → {w_orig}×{h_orig})")

            # 2.3. SCUNet
            if self.scunet.is_ready:
                result = self.scunet.denoise(result)
                print("  → SCUNet denoise")
                result = final_sharpen(result, amount=0.75, sigma=0.8)
                print("  → Post-SCUNet sharpen (0.75, σ=0.8)")
            else:
                print("  ⚠ SCUNet не загружен")

            # 2.4. Highlight rolloff
            result = highlight_rolloff(result, threshold=0.92, rolloff=0.75)

            # 2.5. Баланс белого
            result = correct_white_balance(result, strength=0.4)

            # 2.6. Гамма-доводка
            after_ai = analyze_frame(result)["mean_brightness"]
            if after_ai < 105:
                result = adaptive_gamma(result, target_mean=112.0)
                print(f"  → Гамма ({after_ai:.0f} → 112)")

            # 2.7. Post-гамма bilateral — только для очень тёмных
            if mean_bright < 15 and self.scunet.is_ready:
                result = bilateral_denoise(result, d=5,
                                           sigma_color=20, sigma_space=20)
                print("  → Post-гамма bilateral")

        else:
            result = self._basic_enhance(image, mean_bright, color_cast, report)
            print("  → OpenCV-усиление")

        # 3. СЕГМЕНТАЦИЯ
        masks = self.segmenter.build_masks(result)
        has_face = masks["found"]

        # 4. РЕТУШЬ
        if has_face:
            result = retouch_skin(
                result, masks["skin"],
                lift_shadows=1.08, smooth_amount=0.25,
            )
            result = process_background(
                result, masks["background"],
                darken=0.94, contrast_boost=1.05,
            )
            print("  → Ретушь: кожа + фон")

        # 5. КИНОГРЕЙДИНГ
        after_mean = analyze_frame(result)["mean_brightness"]
        if (not has_face) and diagnosis == "normal" and after_mean >= 100:
            result = cinematic_grade(
                result,
                curve_strength=0.30,
                toning_strength=0.05,
                vignette_strength=0.12,
            )
            print("  → Киногрейдинг")

        # 6. ФИНАЛИЗАЦИЯ
        if has_face:
            result = warm_tone(result, amount=0.06)

        if (not has_face) and diagnosis == "normal" and mean_bright >= 100:
            result = enhance_brightness_global(result, factor=1.03)

        # Микро-контраст
        if mean_bright >= 60:
            amount = 0.10 if has_face else 0.18
            result = micro_contrast(result, amount=amount)
        elif mean_bright >= 40:
            result = micro_contrast(result, amount=0.10)

        # Финальная резкость
        if mean_bright < 20:
            result = final_sharpen(result, amount=0.55, sigma=1.0)
            print("  → Финальная резкость (0.55)")
        elif mean_bright < 40:
            result = final_sharpen(result, amount=0.70, sigma=0.9)
            print("  → Финальная резкость (0.70)")
        elif mean_bright < 80:
            result = final_sharpen(result, amount=0.75, sigma=0.9)
            print("  → Финальная резкость (0.75)")
        else:
            result = final_sharpen(result, amount=0.80, sigma=0.9)
            print("  → Финальная резкость (0.80)")

        if after_mean > 180:
            result = prevent_overexposure(result, max_v=245)

        if result.shape[:2] != (h_orig, w_orig):
            result = cv2.resize(result, (w_orig, h_orig),
                                interpolation=cv2.INTER_LANCZOS4)

        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"  → Итого: {dt_ms:.0f} мс")

        return result

    def _basic_enhance(self, image, mean_bright, color_cast, report):
        result = image.copy()

        if mean_bright < 25:
            result = denoise(result, h=3)
        elif mean_bright < 50:
            result = denoise(result, h=2)

        if color_cast != "neutral":
            avg_diff = abs(report["color"]["avg_r"] - report["color"]["avg_b"])
            if avg_diff > 20:
                result = correct_white_balance(result, strength=0.28)

        if mean_bright < 100:
            result = adaptive_gamma(result, target_mean=115.0)

        if mean_bright < 60:
            result = suppress_color_noise(result)

        if mean_bright < 40:
            result = enhance_saturation(result, factor=1.20)
        elif mean_bright < 80:
            result = enhance_saturation(result, factor=1.10)

        return result

    @staticmethod
    def _validate(image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Некорректное изображение")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
        return image

    def process_with_masks(self, image: np.ndarray) -> tuple:
        image = self._validate(image)
        result = self.process(image)
        masks = self.segmenter.build_masks(result)
        return result, masks