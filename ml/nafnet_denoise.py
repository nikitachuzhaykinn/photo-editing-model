"""NAFNet ONNX-инференс для подавления шума (tile-based)."""
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nafnet_denoise.onnx")

TILE_SIZE = 256       # фиксированный размер входа модели
TILE_OVERLAP = 32     # перекрытие тайлов для бесшовной склейки


class NAFNetDenoise:
    """Обёртка над ONNX-моделью NAFNet с тайловым инференсом."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load(model_path)

    def _load(self, path: str):
        if not os.path.isfile(path):
            print(f"[NAFNet] Модель не найдена: {path}. Denoise пропущен.")
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                path, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[NAFNet] Загружен: {path}")
            print(f"[NAFNet] Вход: {self.input_name}, выход: {self.output_name}")
        except Exception as e:
            print(f"[NAFNet] Ошибка загрузки: {e}")
            self.session = None

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def _infer_tile(self, tile_bgr: np.ndarray) -> np.ndarray:
        """Прогон одного тайла 256×256 через NAFNet."""
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        out = self.session.run([self.output_name], {self.input_name: tensor})[0]

        if out.ndim == 4:
            out = out[0]
        if out.shape[0] == 3 and out.shape[-1] != 3:
            out = np.transpose(out, (1, 2, 0))

        out = np.clip(out, 0.0, 1.0) * 255.0
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _make_window(size: int, overlap: int) -> np.ndarray:
        """
        Весовая маска для тайла. Края затухают, центр = 1.
        При склейке перекрытий даёт бесшовный результат.
        """
        window = np.ones((size, size), dtype=np.float32)
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)

        # Плавное появление по краям
        window[:overlap, :] *= ramp[:, None]
        window[-overlap:, :] *= ramp[::-1, None]
        window[:, :overlap] *= ramp[None, :]
        window[:, -overlap:] *= ramp[::-1][None, :]

        return window

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        BGR uint8 → BGR uint8.
        Разбивает на тайлы 256×256 с перекрытием 32 и склеивает с усреднением.
        """
        if not self.is_ready:
            return image

        h, w = image.shape[:2]

        # Если изображение маленькое — обрабатываем как один тайл
        if h <= TILE_SIZE and w <= TILE_SIZE:
            padded = cv2.copyMakeBorder(
                image, 0, TILE_SIZE - h, 0, TILE_SIZE - w,
                cv2.BORDER_REFLECT
            )
            result = self._infer_tile(padded)
            return result[:h, :w]

        # Аккумулятор и веса
        acc = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        step = TILE_SIZE - TILE_OVERLAP
        window = self._make_window(TILE_SIZE, TILE_OVERLAP)

        # Позиции тайлов по X и Y
        ys = list(range(0, max(1, h - TILE_SIZE + 1), step))
        xs = list(range(0, max(1, w - TILE_SIZE + 1), step))

        # Гарантируем, что последний тайл упирается в правый/нижний край
        if ys[-1] + TILE_SIZE < h:
            ys.append(h - TILE_SIZE)
        if xs[-1] + TILE_SIZE < w:
            xs.append(w - TILE_SIZE)

        for y in ys:
            for x in xs:
                tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
                if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                    # На всякий случай — добиваем отражением
                    pad_h = TILE_SIZE - tile.shape[0]
                    pad_w = TILE_SIZE - tile.shape[1]
                    tile = cv2.copyMakeBorder(
                        tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )

                denoised = self._infer_tile(tile).astype(np.float32)

                acc[y:y + TILE_SIZE, x:x + TILE_SIZE] += denoised * window[..., None]
                weights[y:y + TILE_SIZE, x:x + TILE_SIZE] += window

        weights = np.maximum(weights, 1e-6)
        result = (acc / weights[..., None]).clip(0, 255).astype(np.uint8)
        return result