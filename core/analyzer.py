"""Анализ кадра: яркость, оттенок, гистограмма."""
import cv2
import numpy as np


def analyze_brightness(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return {
        "mean_brightness": mean_brightness,
        "is_dark": mean_brightness < 70,
        "is_normal": 70 <= mean_brightness <= 180,
        "is_bright": mean_brightness > 180,
    }


def analyze_color_cast(image: np.ndarray) -> dict:
    b, g, r = cv2.split(image)
    avg_b, avg_g, avg_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))

    if avg_b > avg_r + 15:
        color_cast = "blue"
    elif avg_r > avg_b + 15:
        color_cast = "yellow"
    else:
        color_cast = "neutral"

    return {
        "avg_b": avg_b, "avg_g": avg_g, "avg_r": avg_r,
        "color_cast": color_cast,
    }


def analyze_histogram(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()

    if total == 0:
        return {"diagnosis": "normal", "mean_brightness": 0.0}

    shadows = hist[0:64].sum() / total
    midtones = hist[64:192].sum() / total
    highlights = hist[192:256].sum() / total
    mean_bright = float(np.mean(gray))

    if mean_bright < 55 or shadows > 0.65:
        diagnosis = "underexposed"
    elif mean_bright > 190 or highlights > 0.6:
        diagnosis = "overexposed"
    elif midtones < 0.35:
        diagnosis = "low_contrast"
    else:
        diagnosis = "normal"

    return {
        "shadows": float(shadows),
        "midtones": float(midtones),
        "highlights": float(highlights),
        "mean_brightness": mean_bright,
        "diagnosis": diagnosis,
    }


def analyze_frame(image: np.ndarray) -> dict:
    brightness = analyze_brightness(image)
    histogram = analyze_histogram(image)
    color = analyze_color_cast(image)
    return {
        "brightness": brightness,
        "histogram": histogram,
        "color": color,
        "diagnosis": histogram["diagnosis"],
        "mean_brightness": brightness["mean_brightness"],
    }