"""Кинематографичный грейдинг."""
import cv2
import numpy as np


def apply_filmic_curve(image, strength=0.5):
    table = np.arange(256, dtype=np.float32) / 255.0
    s_curve = table + strength * (table * (1.0 - table) * (2.0 * table - 1.0) * 0.5)
    s_curve = s_curve * 0.95 + 0.05 * strength
    s_curve = np.where(s_curve > 0.9, 0.9 + (s_curve - 0.9) * 0.5, s_curve)
    lut = np.clip(s_curve * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_split_toning(image, highlight_color=(255, 180, 120),
                       shadow_color=(120, 160, 200),
                       balance=0.5, strength=0.15):
    result = image.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    shadow_mask = np.clip(1.0 - gray / balance, 0, 1)
    highlight_mask = np.clip((gray - balance) / (1.0 - balance), 0, 1)

    for c in range(3):
        result[:, :, c] += (
            shadow_mask * (shadow_color[c] - 128) * strength
            + highlight_mask * (highlight_color[c] - 128) * strength
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignette(image, strength=0.25):
    h, w = image.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    vignette = 1.0 - strength * np.clip(radius - 0.3, 0, 1) ** 2

    result = image.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= vignette
    return np.clip(result, 0, 255).astype(np.uint8)


def cinematic_grade(image, curve_strength=0.5, toning_strength=0.12,
                    vignette_strength=0.20):
    result = apply_filmic_curve(image, strength=curve_strength)
    result = apply_split_toning(result, strength=toning_strength)
    result = apply_vignette(result, strength=vignette_strength)
    return result