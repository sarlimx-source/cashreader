
"""
Bill detection and perspective correction.

This module provides the detect_bill() function expected by pipeline.py.
It is designed for the phone-camera prototype and does not require a
machine-learning model yet.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    """
    Order four corner points as:

        top-left
        top-right
        bottom-right
        bottom-left
    """
    points = np.asarray(points, dtype=np.float32)

    if points.shape != (4, 2):
        raise ValueError("Expected exactly four 2D points")

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]          # top-left
    ordered[2] = points[np.argmax(sums)]          # bottom-right
    ordered[1] = points[np.argmin(differences)]   # top-right
    ordered[3] = points[np.argmax(differences)]   # bottom-left

    return ordered


def _find_bill_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Try to find a rectangular contour corresponding to the banknote.
    """

    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise while keeping the bill boundary.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect strong edges.
    edges = cv2.Canny(blurred, 50, 150)

    # Connect broken edges.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = image.shape[0] * image.shape[1]

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny objects.
        if area < image_area * 0.10:
            continue

        # Ignore objects that occupy almost the entire image.
        if area > image_area * 0.98:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) != 4:
            continue

        if not cv2.isContourConvex(approximation):
            continue

        points = approximation.reshape(4, 2).astype(np.float32)

        # Check the approximate banknote aspect ratio.
        ordered = _order_points(points)

        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])

        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])

        width = max(width_top, width_bottom)
        height = max(height_left, height_right)

        if width <= 0 or height <= 0:
            continue

        ratio = max(width, height) / min(width, height)

        # U.S. banknotes are roughly 1.64:1.
        if ratio < 1.20 or ratio > 2.20:
            continue

        candidates.append(
            (
                area,
                ordered,
                ratio,
            )
        )

    if not candidates:
        return None

    # Prefer the largest plausible rectangle.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


def _warp_bill(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Perspective-correct the detected bill.
    """

    corners = _order_points(corners)

    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])

    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])

    width = int(max(top_width, bottom_width))
    height = int(max(left_height, right_height))

    width = max(width, 100)
    height = max(height, 60)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destination,
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
    )

    return warped


def _detection_confidence(
    image: np.ndarray,
    corners: Optional[np.ndarray],
) -> float:
    """
    Produce a simple prototype confidence score.
    """

    if corners is None:
        return 0.0

    image_area = float(image.shape[0] * image.shape[1])

    area = abs(cv2.contourArea(corners.astype(np.float32)))

    if image_area <= 0:
        return 0.0

    coverage = area / image_area

    # A bill covering approximately 20–80% of the camera frame
    # is generally useful for this prototype.
    if coverage < 0.10:
        confidence = 0.30
    elif coverage < 0.20:
        confidence = 0.55
    elif coverage <= 0.80:
        confidence = 0.90
    else:
        confidence = 0.65

    return round(float(confidence), 3)


def detect_bill(
    image: np.ndarray,
    *,
    warp: bool = True,
) -> Dict[str, Any]:
    """
    Detect a U.S. banknote in a camera image.

    Parameters
    ----------
    image:
        OpenCV BGR image.

    warp:
        If True, perspective-correct the detected bill.

    Returns
    -------
    dict
        Contains:

        found
            Whether a bill-like rectangle was found.

        confidence
            Prototype detection confidence.

        corners
            Four detected corners.

        image
            Original image.

        warped
            Perspective-corrected bill image.

        width
            Detected bill width.

        height
            Detected bill height.
    """

    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Image is empty")

    if len(image.shape) != 3:
        raise ValueError("Expected a color image with 3 channels")

    corners = _find_bill_contour(image)

    if corners is None:
        height, width = image.shape[:2]
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        # Some submitted photos/crops contain the full bill almost edge to
        # edge, so the contour is rejected as occupying nearly the whole
        # frame. If the frame itself has a plausible banknote aspect ratio,
        # use it as a lower-confidence fallback instead of failing OCR.
        if not 2.0 <= aspect_ratio <= 2.6:
            return {
                "found": False,
                "confidence": 0.0,
                "corners": None,
                "image": image,
                "warped": image,
                "width": width,
                "height": height,
                "message": "No bill-shaped rectangle detected.",
            }

        corners = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        fallback_confidence = 0.45
        message = "Using full image as a bill-shaped crop."
    else:
        fallback_confidence = None
        message = "Bill detected."

    confidence = fallback_confidence or _detection_confidence(image, corners)

    ordered = _order_points(corners)

    width_top = np.linalg.norm(
        ordered[1] - ordered[0]
    )

    width_bottom = np.linalg.norm(
        ordered[2] - ordered[3]
    )

    height_left = np.linalg.norm(
        ordered[3] - ordered[0]
    )

    height_right = np.linalg.norm(
        ordered[2] - ordered[1]
    )

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    if warp and fallback_confidence is not None:
        warped = cv2.resize(
            image,
            (1500, 650),
            interpolation=cv2.INTER_AREA,
        )
    elif warp:
        warped = _warp_bill(
            image,
            ordered,
        )
    else:
        warped = image

    return {
        "found": True,
        "confidence": confidence,
        "corners": ordered.tolist(),
        "image": image,
        "warped": warped,
        "width": width,
        "height": height,
        "message": message,
    }


__all__ = [
    "detect_bill",
]
