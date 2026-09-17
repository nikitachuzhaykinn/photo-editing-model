"""Тест ядра на одном фото."""
import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.processor import ImageProcessor
from core.analyzer import analyze_frame

INPUT = os.path.join(PROJECT_ROOT, "data", "dataset", "bad_light", "test.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "processed")


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(INPUT):
        print(f"❌ Файл не найден: {INPUT}")
        return

    image = cv2.imread(INPUT)
    if image is None:
        print(f"❌ Не удалось прочитать: {INPUT}")
        return

    print(f"📷 Фото: {INPUT}  ({image.shape})")

    before = analyze_frame(image)
    print(f"📊 До:    яркость={before['mean_brightness']:.1f}, "
          f"диагноз={before['diagnosis']}, "
          f"оттенок={before['color']['color_cast']}")

    processor = ImageProcessor(mode="auto")
    t0 = time.perf_counter()
    result = processor.process(image)
    dt_ms = (time.perf_counter() - t0) * 1000

    after = analyze_frame(result)
    print(f"📊 После: яркость={after['mean_brightness']:.1f}, "
          f"диагноз={after['diagnosis']}")
    print(f"⏱  Время: {dt_ms:.0f} мс")

    h = max(image.shape[0], result.shape[0])
    canvas = np.zeros((h, image.shape[1] + result.shape[1], 3), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    canvas[:result.shape[0], image.shape[1]:] = result

    cv2.imwrite(os.path.join(OUTPUT_DIR, "single_result.jpg"), result)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "single_compare.jpg"),
                np.hstack([_label(image, "BEFORE"), _label(result, "AFTER")]))

    print(f"✅ Результат: {OUTPUT_DIR}/single_compare.jpg")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)