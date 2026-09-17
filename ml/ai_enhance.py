"""Zero-DCE++ ONNX-инференс."""
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "zero_dce_plus.onnx")


class ZeroDCE:
    def __init__(self, model_path: str = MODEL_PATH):
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load(model_path)

    def _load(self, path):
        if not os.path.isfile(path):
            print(f"[ZeroDCE] Модель не найдена: {path}")
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
            print(f"[ZeroDCE] Загружен: {path}")
        except Exception as e:
            print(f"[ZeroDCE] Ошибка: {e}")
            self.session = None

    @property
    def is_ready(self):
        return self.session is not None

    def enhance(self, image):
        if not self.is_ready:
            return image

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        out = self.session.run([self.output_name], {self.input_name: tensor})[0]

        if out.ndim == 4:
            out = out[0]
        if out.shape[0] == 3 and out.shape[-1] != 3:
            out = np.transpose(out, (1, 2, 0))

        out = np.clip(out, 0.0, 1.0) * 255.0
        result = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
        return result