"""OCR helpers for U.S. currency."""
from __future__ import annotations
import os
import re
import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pytesseract

SERIAL_RE = re.compile(r"[A-Z0-9]{8,10}")
MAX_OCR_DIMENSION = 2400


def _configure_tesseract():
    """Find the standard Windows install when Tesseract is not on PATH."""
    if shutil.which("tesseract"):
        return

    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if not root:
            continue
        executable = Path(root) / "Tesseract-OCR" / "tesseract.exe"
        if executable.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(executable)
            return


_configure_tesseract()


def _valid(img):
    return isinstance(img, np.ndarray) and img.size > 0


def _gray(img):
    if not _valid(img):
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_variants(img):
    """Make OCR-friendly versions while preserving thin printed characters."""
    gray = _gray(img)
    if gray is None:
        return []
    # Enlarge small character crops, but cap large bill-wide inputs to avoid
    # creating multi-megapixel copies for every Tesseract pass.
    height, width = gray.shape[:2]
    scale = min(5.0, MAX_OCR_DIMENSION / max(height, width))
    if scale > 1.0:
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        up = gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(up)
    denoise = cv2.GaussianBlur(contrast, (3, 3), 0)
    otsu = cv2.threshold(denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 9
    )
    # A blackhat image is useful when dark printed text sits on a lighter note.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(contrast, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    return [up, contrast, denoise, otsu, adaptive, blackhat]


def preprocess(img):
    variants = preprocess_variants(img)
    return variants[1] if len(variants) > 1 else (variants[0] if variants else None)


def _ocr(image, whitelist, psm):
    if not _valid(image):
        return ""
    try:
        return pytesseract.image_to_string(
            image,
            config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}",
        ).strip()
    except Exception:
        return ""


def read_text(img, whitelist, psm=7):
    results = []
    variants = preprocess_variants(img)
    for index in (1, 3):
        if index >= len(variants):
            continue
        v = variants[index]
        text = _ocr(v, whitelist, psm)
        if text:
            results.append(text)
    return max(results, key=len) if results else ""


def _clean(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def read_denomination(img) -> Optional[int]:
    text = read_text(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$", 6).upper()
    words = {"ONE": 1, "FIVE": 5, "TEN": 10, "TWENTY": 20, "FIFTY": 50, "ONEHUNDRED": 100, "HUNDRED": 100}
    for word, value in words.items():
        if word in text.replace(" ", ""):
            return value
    for value in (100, 50, 20, 10, 5, 1):
        if str(value) in text.replace(" ", ""):
            return value
    return None


def read_fed_letter(img) -> Optional[str]:
    text = _clean(read_text(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10))
    return next((c for c in text if c in "ABCDEFGHIJKL"), None)


def read_fed_number(img) -> Optional[int]:
    text = read_text(img, "0123456789", 10)
    nums = [int(x) for x in re.findall(r"\d{1,2}", text)]
    nums = [x for x in nums if 1 <= x <= 12]
    return nums[0] if nums else None


def read_series(img) -> Optional[str]:
    text = _clean(read_text(img, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6))
    matches = re.findall(r"(?:19|20)\d{2}[A-Z]?", text)
    for m in matches:
        if 1900 <= int(m[:4]) <= 2035:
            return m
    return None


def _normalize_serial(text):
    return _clean(text)


def _serial_candidates(text):
    clean = _normalize_serial(text)
    out = list(SERIAL_RE.findall(clean))
    if 8 <= len(clean) <= 10:
        out.append(clean)
    return out


def _serial_score(s):
    if not s:
        return -1
    digits = sum(c.isdigit() for c in s)
    letters = len(s) - digits
    score = 0
    if len(s) == 8:
        score += 8
    elif len(s) in (9, 10):
        score += 5
    if digits >= 7:
        score += 6
    elif digits >= 6:
        score += 3
    if letters <= 2:
        score += 2
    if s[0].isalpha() or s[-1].isalpha():
        score += 1
    return score


def read_serial(img) -> Optional[str]:
    """Read a serial from a tightly cropped serial-number region."""
    candidates = []
    variants = preprocess_variants(img)
    for index in (1, 3):
        if index >= len(variants):
            continue
        for psm in (6, 7, 8):
            v = variants[index]
            text = _ocr(v, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", psm)
            candidates.extend(_serial_candidates(text))
    if not candidates:
        return None
    # Preserve the most frequent candidate, with score as tie-breaker.
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=lambda c: (counts[c], _serial_score(c)))


def serial_similarity(a, b):
    if not a or not b:
        return 0.0
    a, b = _normalize_serial(a), _normalize_serial(b)
    if a == b:
        return 1.0
    if len(a) != len(b):
        return 0.0
    corrections = {"O":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "G":"6", "B":"8"}
    matches = sum(corrections.get(x, x) == corrections.get(y, y) for x, y in zip(a, b))
    return matches / max(len(a), 1)
