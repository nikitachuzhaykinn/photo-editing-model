"""SCUNet ONNX-инференс для подавления шума."""
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "scunet_color_real_psnr.onnx")

MAX_SIDE = 768  # ↑ с 640 — меньше апскейла, меньше мыла


class SCUNetDenoise:
    """Обёртка над ONNX-моделью SCUNet с downscale и паддингом."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load(model_path)

    def _load(self, path: str):
        if not os.path.isfile(path):
            print(f"[SCUNet] Модель не найдена: {path}. Denoise пропущен.")
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 6
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                path, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[SCUNet] Загружен: {path}")
        except Exception as e:
            print(f"[SCUNet] Ошибка: {e}")
            self.session = None

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    @staticmethod
    def _pad_to_multiple(image: np.ndarray, multiple: int = 64):
        h, w = image.shape[:2]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return image, 0, 0
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_REPLICATE)
        return padded, pad_h, pad_w

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            return image

        h_orig, w_orig = image.shape[:2]

        longest = max(h_orig, w_orig)
        if longest > MAX_SIDE:
            scale = MAX_SIDE / longest
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            work = cv2.resize(image, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
            need_upscale = True
        else:
            work = image
            need_upscale = False

        padded, pad_h, pad_w = self._pad_to_multiple(work, multiple=64)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]

        out = out[0] if out.ndim == 4 else out
        if out.shape[0] == 3 and out.shape[-1] != 3:
            out = np.transpose(out, (1, 2, 0))

        out = np.clip(out, 0.0, 1.0) * 255.0
        result = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = result[:work.shape[0], :work.shape[1]]

        if need_upscale:
            result = cv2.resize(result, (w_orig, h_orig),
                                interpolation=cv2.INTER_LANCZOS4)

        return result