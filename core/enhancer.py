"""Низкоуровневое усиление: шумодав, гамма, баланс белого, контраст."""
import cv2
import numpy as np


def denoise(image: np.ndarray, h: float = 5) -> np.ndarray:
    """Шумодав fastNlMeansColored на полном разрешении."""
    h = int(round(h))
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=h, hColor=h,
        templateWindowSize=7, searchWindowSize=21,
    )


def bilateral_denoise(image: np.ndarray, d: int = 5,
                      sigma_color: float = 30,
                      sigma_space: float = 30) -> np.ndarray:
    """Bilateral — быстрый, сохраняет края."""
    return cv2.bilateralFilter(image, d=d,
                               sigmaColor=sigma_color,
                               sigmaSpace=sigma_space)


def adaptive_gamma(image: np.ndarray, target_mean: float = 115.0) -> np.ndarray:
    """Адаптивная гамма по L-каналу в LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mean_l = float(np.mean(l))
    if mean_l < 1 or mean_l > 250:
        return image

    gamma = np.log(target_mean / 255.0) / np.log(mean_l / 255.0)
    gamma = float(np.clip(gamma, 0.3, 3.0))

    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    l = cv2.LUT(l, table)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def correct_white_balance(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gray World с контролируемой силой."""
    result = image.astype(np.float32)
    avg_b = float(np.mean(result[:, :, 0]))
    avg_g = float(np.mean(result[:, :, 1]))
    avg_r = float(np.mean(result[:, :, 2]))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    scale_b = 1.0 + strength * ((avg_gray / max(avg_b, 1e-6)) - 1.0)
    scale_g = 1.0 + strength * ((avg_gray / max(avg_g, 1e-6)) - 1.0)
    scale_r = 1.0 + strength * ((avg_gray / max(avg_r, 1e-6)) - 1.0)

    result[:, :, 0] *= scale_b
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_r

    return np.clip(result, 0, 255).astype(np.uint8)


def warm_tone(image: np.ndarray, amount: float = 0.06) -> np.ndarray:
    """Лёгкий тёплый тон для портретов."""
    result = image.astype(np.float32)
    result[:, :, 2] *= 1.0 + amount
    result[:, :, 0] *= 1.0 - amount * 0.5
    return np.clip(result, 0, 255).astype(np.uint8)


def suppress_color_noise(image: np.ndarray) -> np.ndarray:
    """Сглаживает только a,b в LAB. L не трогает."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.bilateralFilter(a, d=5, sigmaColor=15, sigmaSpace=15)
    b = cv2.bilateralFilter(b, d=5, sigmaColor=15, sigmaSpace=15)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def prevent_overexposure(image: np.ndarray, max_v: int = 245) -> np.ndarray:
    """Защита от выжженных бликов."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, max_v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_saturation(image: np.ndarray, factor: float = 1.1) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_brightness_global(image: np.ndarray, factor: float = 1.03) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def micro_contrast(image: np.ndarray, amount: float = 0.25) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    blurred = cv2.GaussianBlur(l, (0, 0), 3)
    l = cv2.addWeighted(l, 1.0 + amount, blurred, -amount, 0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def final_sharpen(image: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    blurred = cv2.GaussianBlur(l, (0, 0), sigma)
    l = cv2.addWeighted(l, 1.0 + amount, blurred, -amount, 0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def highlight_rolloff(image: np.ndarray, threshold: float = 0.9,
                      rolloff: float = 0.7) -> np.ndarray:
    """Мягкое сжатие светов."""
    result = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0
    mask = np.clip((l - threshold) / (1.0 - threshold + 1e-6), 0, 1)

    for c in range(3):
        result[:, :, c] = np.where(
            result[:, :, c] > threshold,
            threshold + (result[:, :, c] - threshold) * rolloff,
            result[:, :, c]
        )
        result[:, :, c] -= mask * (result[:, :, c] - threshold) * (1 - rolloff) * 0.5

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def add_grain(image: np.ndarray, strength: float = 3.0) -> np.ndarray:
    noise = np.random.normal(0, strength, image.shape).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)