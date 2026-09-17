"""Адресная ретушь кожи и фона."""
import cv2
import numpy as np


def retouch_skin(image, skin_mask, lift_shadows=1.08, smooth_amount=0.25):
    result = image.copy()
    mask = skin_mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)

    threshold = np.percentile(l_f, 35)
    shadow_mask = (l_f < threshold).astype(np.float32)
    l_f = l_f + (lift_shadows - 1.0) * l_f * shadow_mask
    l = np.clip(l_f, 0, 255).astype(np.uint8)

    l_smooth = cv2.bilateralFilter(l, d=9, sigmaColor=25, sigmaSpace=25)
    l_smooth = cv2.addWeighted(l, 1.0 - smooth_amount, l_smooth, smooth_amount, 0)

    lab = cv2.merge((l_smooth, a, b))
    retouched = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blended = (retouched.astype(np.float32) * mask_3ch +
               result.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_background(image, background_mask, darken=0.94, contrast_boost=1.05):
    result = image.copy()
    mask = background_mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    bg = result.copy()
    bg = np.clip(bg.astype(np.float32) * darken, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    l = clahe.apply(l)
    bg = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blended = (bg.astype(np.float32) * mask_3ch +
               result.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)