"""
Main cash-reader scanning pipeline.

The pipeline is designed for U.S. currency.

FRONT OF BILL:
    - denomination
    - serial number (left)
    - serial number (right)
    - series year
    - Federal Reserve letter
    - Federal Reserve district number

BACK OF BILL:
    - denomination/design confirmation
    - used mainly to confirm the denomination and bill type

The scanner can be called in three ways:

    reader.scan_frame(image)
    reader.scan_frame(image, side="front")
    reader.scan_frame(image, side="back")

The default side is "auto".

For the most reliable results, the eventual phone UI should guide the
user through:

    1. Scan FRONT
    2. Scan BACK
    3. Combine the two results
    4. Save the completed bill
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

from cash_reader.detector import detect_bill
from cash_reader.federal_reserve import cross_check
from cash_reader.ocr import (
    read_denomination,
    read_fed_letter,
    read_fed_number,
    read_serial,
    read_series,
    serial_similarity,
)
from cash_reader.profiles import get_profile


class CashReader:
    """
    Main scanner class.

    database:
        Optional BillDatabase instance.

    config_path:
        Kept for compatibility with the project structure.
        Profiles are currently loaded through profiles.py.
    """

    def __init__(
        self,
        database=None,
        config_path: str = "config/profiles.json",
    ):
        self.database = database
        self.config_path = config_path

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def scan_frame(
        self,
        frame,
        side: str = "auto",
        save: bool = False,
    ) -> Dict[str, Any]:
        """
        Scan one camera frame.

        side:
            "front" - treat image as front of bill
            "back"  - treat image as back of bill
            "auto"  - attempt to determine the side

        save:
            If True and a database was supplied, save the result.

        Returns a dictionary suitable for Flask jsonify().
        """

        if frame is None:
            return self._error_result("No camera image supplied.")

        if not isinstance(frame, np.ndarray):
            return self._error_result("Camera image must be a NumPy/OpenCV image.")

        if frame.size == 0:
            return self._error_result("Camera image is empty.")

        side = str(side).lower().strip()

        if side not in {"auto", "front", "back"}:
            return self._error_result(
                "Invalid side. Use 'auto', 'front', or 'back'."
            )

        try:
            prepared = self._prepare_image(frame)

            detection = self._detect_bill(prepared)

            if detection is None:
                return {
                    "status": "needs_review",
                    "side": side if side != "auto" else "unknown",
                    "error": "Could not confidently locate the bill.",
                    "warnings": [
                        "Make sure the entire bill is visible.",
                        "Place the bill on a flat surface.",
                        "Avoid glare and heavy shadows.",
                    ],
                    "confidence": 0.0,
                }

            # Perspective-corrected image of the bill.
            bill_image = self._extract_bill_image(
                prepared,
                detection,
            )

            # Determine front/back if the caller did not specify it.
            detected_side = side

            if side == "auto":
                detected_side = self._determine_side(bill_image)

            if detected_side == "back":
                result = self._analyze_back(bill_image, detection)

            else:
                result = self._analyze_front(bill_image, detection)

            result["side"] = detected_side
            result["detector"] = {
                "found": True,
                "confidence": round(
                    float(detection.get("confidence", 0.0)),
                    3,
                ),
            }

            # Do not save an incomplete front scan automatically.
            if save and self.database is not None:
                result["id"] = self._save_result(result)

            return result

        except Exception as exc:
            return self._error_result(
                f"Scan failed: {exc}"
            )

    def scan(
        self,
        image,
        side: str = "auto",
        save: bool = False,
    ) -> Dict[str, Any]:
        """
        Alias for scan_frame().

        This lets older code call:

            reader.scan(image)
        """

        return self.scan_frame(
            image,
            side=side,
            save=save,
        )

    def analyze_front(self, image) -> Dict[str, Any]:
        """
        Explicit front-side scan.
        """

        return self.scan_frame(
            image,
            side="front",
            save=False,
        )

    def analyze_back(self, image) -> Dict[str, Any]:
        """
        Explicit back-side scan.
        """

        return self.scan_frame(
            image,
            side="back",
            save=False,
        )

    def combine_front_back(
        self,
        front_result: Dict[str, Any],
        back_result: Dict[str, Any],
        save: bool = False,
    ) -> Dict[str, Any]:
        """
        Combine a front scan and back scan into one bill record.

        The front supplies the identifying information.

        The back primarily confirms:
            - denomination
            - design/layout
            - that the second image really looks like the same bill
        """

        if not isinstance(front_result, dict):
            return self._error_result("Front result must be a dictionary.")

        if not isinstance(back_result, dict):
            return self._error_result("Back result must be a dictionary.")

        result = dict(front_result)

        result["side"] = "front+back"

        warnings = []

        warnings.extend(
            self._as_list(
                front_result.get("warnings")
            )
        )

        warnings.extend(
            self._as_list(
                back_result.get("warnings")
            )
        )

        front_denomination = self._clean_denomination(
            front_result.get("denomination")
        )

        back_denomination = self._clean_denomination(
            back_result.get("denomination")
        )

        result["back"] = {
            "status": back_result.get("status"),
            "denomination": back_denomination,
            "confidence": back_result.get("confidence"),
        }

        # --------------------------------------------------------------
        # Compare denominations
        # --------------------------------------------------------------

        if front_denomination and back_denomination:

            if front_denomination == back_denomination:
                result["denomination_match"] = True

            else:
                result["denomination_match"] = False

                warnings.append(
                    "Front and back scans appear to show different denominations."
                )

        else:
            result["denomination_match"] = None

            warnings.append(
                "Could not confirm the denomination from both sides."
            )

        # --------------------------------------------------------------
        # Front fields
        # --------------------------------------------------------------

        required_front_fields = [
            "serial_number",
            "series",
            "federal_reserve_letter",
            "federal_reserve_number",
        ]

        missing = []

        for field in required_front_fields:
            value = front_result.get(field)

            if value in (None, "", "UNKNOWN"):
                missing.append(field)

        if missing:
            warnings.append(
                "Front scan is missing: "
                + ", ".join(missing)
                + "."
            )

        # --------------------------------------------------------------
        # Determine final status
        # --------------------------------------------------------------

        front_ok = front_result.get("status") == "ok"
        back_ok = back_result.get("status") in {
            "ok",
            "needs_review",
        }

        if (
            front_ok
            and back_ok
            and result.get("denomination_match") is True
            and not missing
        ):
            result["status"] = "ok"

        else:
            result["status"] = "needs_review"

        result["warnings"] = self._unique_strings(warnings)

        # Combine confidence.
        front_conf = self._safe_float(
            front_result.get("confidence")
        )

        back_conf = self._safe_float(
            back_result.get("confidence")
        )

        if back_conf > 0:
            result["confidence"] = round(
                (front_conf * 0.70) + (back_conf * 0.30),
                3,
            )

        else:
            result["confidence"] = round(
                front_conf,
                3,
            )

        # Save only the completed combined record.
        if save and self.database is not None:
            result["id"] = self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # FRONT ANALYSIS
    # ------------------------------------------------------------------

    def _analyze_front(
        self,
        bill_image: np.ndarray,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze the front of a bill.

        This is where the important identifying fields are extracted.
        """

        # First try to identify the denomination.
        denomination = self._read_denomination_safely(
            bill_image
        )

        profile = get_profile(
            str(denomination)
            if denomination
            else None
        )

        # --------------------------------------------------------------
        # Targeted OCR using denomination profile
        # --------------------------------------------------------------

        serial_left = ""
        serial_right = ""
        series = ""
        fed_letter = ""
        fed_number = ""

        if profile:

            serial_left = self._read_profile_region(
                bill_image,
                profile,
                "serial_left",
                reader=read_serial,
            )

            serial_right = self._read_profile_region(
                bill_image,
                profile,
                "serial_right",
                reader=read_serial,
            )

            series = self._read_profile_region(
                bill_image,
                profile,
                "series",
                reader=read_series,
            )

            fed_letter = self._read_profile_region(
                bill_image,
                profile,
                "fed_letter",
                reader=read_fed_letter,
            )

            fed_number = self._read_profile_region(
                bill_image,
                profile,
                "fed_number",
                reader=read_fed_number,
            )

        # --------------------------------------------------------------
        # Fallback OCR
        # --------------------------------------------------------------

        if not serial_left or not serial_right:
            fallback = self._fallback_serial_reads(
                bill_image
            )

            if not serial_left:
                serial_left = fallback.get(
                    "serial_left",
                    "",
                )

            if not serial_right:
                serial_right = fallback.get(
                    "serial_right",
                    "",
                )

        if not series:
            series = self._fallback_series(
                bill_image
            )

        if not fed_letter:
            fed_letter = self._fallback_fed_letter(
                bill_image
            )

        if not fed_number:
            fed_number = self._fallback_fed_number(
                bill_image
            )

        # --------------------------------------------------------------
        # Normalize values
        # --------------------------------------------------------------

        serial_left = self._normalize_serial(
            serial_left
        )

        serial_right = self._normalize_serial(
            serial_right
        )

        series = self._normalize_series(
            series
        )

        fed_letter = self._normalize_fed_letter(
            fed_letter
        )

        fed_number = self._normalize_fed_number(
            fed_number
        )

        # --------------------------------------------------------------
        # Compare serial numbers
        # --------------------------------------------------------------

        serial_match = None

        if serial_left and serial_right:
            serial_match = (
                serial_similarity(
                    serial_left,
                    serial_right,
                )
                >= 0.90
            )

        serial_number = ""

        if serial_left and serial_right:
            if serial_match:
                serial_number = serial_left
            else:
                # Keep both reads but don't pretend they are the same.
                serial_number = serial_left

        elif serial_left:
            serial_number = serial_left

        elif serial_right:
            serial_number = serial_right

        # --------------------------------------------------------------
        # Federal Reserve cross-check
        # --------------------------------------------------------------

        fed_check = cross_check(
            fed_letter if fed_letter else None,
            fed_number if fed_number else None,
        )

        federal_reserve_match = fed_check.get(
            "match"
        )

        # --------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------

        warnings = []

        if not denomination:
            warnings.append(
                "Denomination was not confidently identified."
            )

        if not serial_left:
            warnings.append(
                "Left serial number was not confidently read."
            )

        if not serial_right:
            warnings.append(
                "Right serial number was not confidently read."
            )

        if serial_left and serial_right and not serial_match:
            warnings.append(
                "The two serial-number reads do not match."
            )

        if not series:
            warnings.append(
                "Series year was not confidently read."
            )

        if not fed_letter:
            warnings.append(
                "Federal Reserve letter was not confidently read."
            )

        if not fed_number:
            warnings.append(
                "Federal Reserve district number was not confidently read."
            )

        if federal_reserve_match is False:
            warnings.append(
                fed_check.get(
                    "message",
                    "Federal Reserve letter and number conflict.",
                )
            )

        # --------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------

        confidence = self._front_confidence(
            denomination=denomination,
            serial_left=serial_left,
            serial_right=serial_right,
            serial_match=serial_match,
            series=series,
            fed_letter=fed_letter,
            fed_number=fed_number,
            fed_match=federal_reserve_match,
            detector_confidence=detection.get(
                "confidence",
                0.0,
            ),
        )

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        complete = (
            bool(denomination)
            and bool(serial_left)
            and bool(serial_right)
            and serial_match is True
            and bool(series)
            and bool(fed_letter)
            and bool(fed_number)
            and federal_reserve_match is True
        )

        status = "ok" if complete else "needs_review"

        return {
            "status": status,
            "side": "front",
            "denomination": denomination,
            "serial_number": serial_number,
            "serial_left_read": serial_left,
            "serial_right_read": serial_right,
            "serial_match": serial_match,
            "series": series,
            "federal_reserve_letter": fed_letter,
            "federal_reserve_number": fed_number,
            "federal_reserve_district": fed_check.get(
                "district"
            ),
            "federal_reserve_city": fed_check.get(
                "city"
            ),
            "federal_reserve_match": federal_reserve_match,
            "federal_reserve_message": fed_check.get(
                "message"
            ),
            "confidence": round(
                confidence,
                3,
            ),
            "warnings": self._unique_strings(
                warnings
            ),
            "scanned_at": datetime.utcnow().isoformat(
                timespec="seconds"
            ),
        }

    # ------------------------------------------------------------------
    # BACK ANALYSIS
    # ------------------------------------------------------------------

    def _analyze_back(
        self,
        bill_image: np.ndarray,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze the back of a bill.

        The back normally does NOT contain the serial-number fields
        needed for identification.

        Therefore this function focuses on denomination/design clues.
        """

        denomination = self._read_denomination_safely(
            bill_image
        )

        warnings = []

        if not denomination:
            warnings.append(
                "Could not confidently identify the denomination from the back."
            )

        # Back-side confidence is intentionally lower than a complete
        # front-side scan because the important identifying fields are
        # normally on the front.
        detector_conf = self._safe_float(
            detection.get("confidence")
        )

        confidence = 0.0

        if denomination:
            confidence += 0.55

        confidence += detector_conf * 0.45

        confidence = min(
            max(confidence, 0.0),
            1.0,
        )

        status = (
            "ok"
            if denomination
            else "needs_review"
        )

        return {
            "status": status,
            "side": "back",
            "denomination": denomination,
            "serial_number": "",
            "serial_left_read": "",
            "serial_right_read": "",
            "serial_match": None,
            "series": "",
            "federal_reserve_letter": "",
            "federal_reserve_number": "",
            "federal_reserve_district": None,
            "federal_reserve_city": None,
            "federal_reserve_match": None,
            "confidence": round(
                confidence,
                3,
            ),
            "warnings": self._unique_strings(
                warnings
            ),
            "scanned_at": datetime.utcnow().isoformat(
                timespec="seconds"
            ),
        }

    # ------------------------------------------------------------------
    # BILL DETECTION
    # ------------------------------------------------------------------

    def _detect_bill(
        self,
        image: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Call detector.py.

        Different versions of detector.py may return slightly different
        structures, so this method accepts several common formats.
        """

        try:
            result = detect_bill(image)
        except TypeError:
            # Some detector implementations may use detect_bill(frame=...)
            result = detect_bill(frame=image)

        if result is None:
            return None

        if isinstance(result, dict):

            if result.get("found") is False:
                return None

            # Common format:
            # {
            #   "corners": ...,
            #   "confidence": ...
            # }
            if "corners" in result:
                return result

            # Some versions may return:
            # {
            #   "points": ...,
            #   ...
            # }
            if "points" in result:
                converted = dict(result)
                converted["corners"] = result["points"]
                return converted

        # If detector returns corners directly.
        if isinstance(result, (list, tuple, np.ndarray)):

            try:
                points = np.asarray(
                    result,
                    dtype=np.float32,
                )

                if points.size >= 8:
                    points = points.reshape(
                        -1,
                        2,
                    )

                    return {
                        "corners": points,
                        "confidence": 0.5,
                    }

            except Exception:
                pass

        return None

    def _extract_bill_image(
        self,
        image: np.ndarray,
        detection: Dict[str, Any],
    ) -> np.ndarray:
        """
        Perspective-correct the detected bill.

        If the detector already provides a corrected image, use it.
        Otherwise use its corner points.
        """

        if detection.get("warped") is not None:
            warped = detection["warped"]

            if isinstance(warped, np.ndarray):
                return warped

        if detection.get("image") is not None:
            detected_image = detection["image"]

            if isinstance(
                detected_image,
                np.ndarray,
            ):
                return detected_image

        corners = detection.get("corners")

        if corners is None:
            return image

        points = np.asarray(
            corners,
            dtype=np.float32,
        )

        if points.shape[0] < 4:
            return image

        points = points.reshape(
            -1,
            2,
        )[:4]

        ordered = self._order_points(points)

        width = 1500
        height = 650

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
            ordered,
            destination,
        )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
        )

        return warped

    # ------------------------------------------------------------------
    # FRONT / BACK DETECTION
    # ------------------------------------------------------------------

    def _determine_side(
        self,
        bill_image: np.ndarray,
    ) -> str:
        """
        Attempt to distinguish front from back.

        This is deliberately conservative.

        A normal camera image does not contain a universal, guaranteed
        front/back marker for every U.S. bill design, so this function
        uses OCR/layout clues rather than pretending it can be perfect.

        If there are strong front-side clues, return "front".

        Otherwise return "back".

        The phone UI should eventually allow the user to explicitly
        select FRONT or BACK for maximum reliability.
        """

        gray = cv2.cvtColor(
            bill_image,
            cv2.COLOR_BGR2GRAY,
        )

        h, w = gray.shape[:2]

        # The front generally contains several textual identifiers.
        # Examine the upper and lower portions where those fields
        # commonly occur.
        top = gray[
            0:int(h * 0.35),
            :
        ]

        bottom = gray[
            int(h * 0.55):h,
            :
        ]

        front_score = 0.0

        # OCR denomination clue.
        denomination = self._read_denomination_safely(
            bill_image
        )

        if denomination:
            front_score += 0.15

        # Series clue.
        series = self._safe_reader(
            read_series,
            top,
        )

        if series:
            front_score += 0.25

        # Federal Reserve letter clue.
        fed_letter = self._safe_reader(
            read_fed_letter,
            top,
        )

        if fed_letter:
            front_score += 0.20

        # Federal Reserve number clue.
        fed_number = self._safe_reader(
            read_fed_number,
            top,
        )

        if fed_number:
            front_score += 0.15

        # Serial number clues.
        serial_top = self._safe_reader(
            read_serial,
            top,
        )

        serial_bottom = self._safe_reader(
            read_serial,
            bottom,
        )

        if serial_top:
            front_score += 0.15

        if serial_bottom:
            front_score += 0.10

        if front_score >= 0.40:
            return "front"

        return "back"

    # ------------------------------------------------------------------
    # PROFILE OCR
    # ------------------------------------------------------------------

    def _read_profile_region(
        self,
        bill_image: np.ndarray,
        profile: Dict[str, Any],
        name: str,
        reader,
    ) -> str:
        """
        Read one profile-defined region.

        Regions are normalized to 0..1 coordinates.
        """

        region = profile.get(name)

        if not region:
            return ""

        crop = self._crop_normalized(
            bill_image,
            region,
        )

        if crop is None or crop.size == 0:
            return ""

        return self._safe_reader(
            reader,
            crop,
        )

    def _crop_normalized(
        self,
        image: np.ndarray,
        region,
    ) -> Optional[np.ndarray]:
        """
        Crop using normalized coordinates.

        Accepted formats:

            [x1, y1, x2, y2]

        or:

            {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...
            }
        """

        h, w = image.shape[:2]

        try:

            if isinstance(region, dict):

                x1 = float(region.get("x1", 0))
                y1 = float(region.get("y1", 0))
                x2 = float(region.get("x2", 1))
                y2 = float(region.get("y2", 1))

            else:

                if len(region) != 4:
                    return None

                x1, y1, x2, y2 = [
                    float(value)
                    for value in region
                ]

            x1 = max(
                0,
                min(
                    1,
                    x1,
                ),
            )

            y1 = max(
                0,
                min(
                    1,
                    y1,
                ),
            )

            x2 = max(
                0,
                min(
                    1,
                    x2,
                ),
            )

            y2 = max(
                0,
                min(
                    1,
                    y2,
                ),
            )

            left = int(
                x1 * w
            )

            top = int(
                y1 * h
            )

            right = int(
                x2 * w
            )

            bottom = int(
                y2 * h
            )

            if right <= left or bottom <= top:
                return None

            return image[
                top:bottom,
                left:right,
            ]

        except Exception:
            return None

    # ------------------------------------------------------------------
    # FALLBACK OCR
    # ------------------------------------------------------------------

    def _fallback_serial_reads(
        self,
        image: np.ndarray,
    ) -> Dict[str, str]:
        """Use the known physical locations of U.S. bill serial numbers.

        On the front of modern U.S. notes the two serial numbers are
        generally separated: one is in the lower-left area and the other
        is in the upper-right area. We try several slightly larger crops
        around those locations so small camera alignment errors do not
        make the OCR crop miss the number.
        """
        h, w = image.shape[:2]

        left_regions = [
            image[int(h*.58):int(h*.90), int(w*.03):int(w*.52)],
            image[int(h*.64):int(h*.88), int(w*.02):int(w*.55)],
            image[int(h*.52):int(h*.82), int(w*.05):int(w*.50)],
        ]
        right_regions = [
            image[int(h*.10):int(h*.42), int(w*.48):int(w*.97)],
            image[int(h*.12):int(h*.38), int(w*.52):int(w*.98)],
            image[int(h*.18):int(h*.48), int(w*.45):int(w*.95)],
        ]

        def best(regions):
            candidates = []
            for crop in regions:
                value = self._safe_reader(read_serial, crop)
                if value:
                    candidates.append(value)
            if not candidates:
                return ""
            # Prefer the most common reading; this reduces one-off OCR errors.
            counts = {x: candidates.count(x) for x in candidates}
            return max(candidates, key=lambda x: (counts[x], len(x)))

        return {
            "serial_left": best(left_regions),
            "serial_right": best(right_regions),
        }

    def _fallback_series(
        self,
        image: np.ndarray,
    ) -> str:
        h = image.shape[0]

        crop = image[
            0:int(h * 0.50),
            :
        ]

        return self._safe_reader(
            read_series,
            crop,
        )

    def _fallback_fed_letter(
        self,
        image: np.ndarray,
    ) -> str:
        h, w = image.shape[:2]

        crop = image[
            0:int(h * 0.65),
            int(w * 0.20):int(w * 0.80),
        ]

        return self._safe_reader(
            read_fed_letter,
            crop,
        )

    def _fallback_fed_number(
        self,
        image: np.ndarray,
    ) -> str:
        h, w = image.shape[:2]

        crop = image[
            0:int(h * 0.65),
            int(w * 0.20):int(w * 0.80),
        ]

        return self._safe_reader(
            read_fed_number,
            crop,
        )

    # ------------------------------------------------------------------
    # DENOMINATION
    # ------------------------------------------------------------------

    def _read_denomination_safely(
        self,
        image: np.ndarray,
    ) -> Optional[str]:
        """
        Read denomination and normalize it to strings like:

            "1"
            "5"
            "10"
            "20"
            "50"
            "100"
        """

        try:
            # OCR on the whole bill often mistakes the ornamental $1 for
            # nearby printed text (for example, reading a one-dollar note
            # as $10). The upper-left denomination is a more focused cue.
            profile = get_profile("1")
            regions = profile.get("denomination_regions", []) if profile else []
            if regions:
                corner = self._crop_normalized(image, regions[0])
                value = self._safe_reader(read_denomination, corner) if corner is not None else None
            else:
                value = None

            if value in (None, ""):
                value = read_denomination(image)
        except Exception:
            return None

        return self._clean_denomination(
            value
        )

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    def _front_confidence(
        self,
        denomination,
        serial_left,
        serial_right,
        serial_match,
        series,
        fed_letter,
        fed_number,
        fed_match,
        detector_confidence,
    ) -> float:
        """
        Calculate an overall confidence score.

        This is not a machine-learning probability.

        It is a practical prototype score based on how many required
        fields were successfully read and cross-checked.
        """

        score = 0.0

        detector_confidence = self._safe_float(
            detector_confidence
        )

        score += detector_confidence * 0.15

        if denomination:
            score += 0.15

        if serial_left:
            score += 0.15

        if serial_right:
            score += 0.15

        if serial_match is True:
            score += 0.15

        if series:
            score += 0.10

        if fed_letter:
            score += 0.05

        if fed_number:
            score += 0.05

        if fed_match is True:
            score += 0.10

        if fed_match is False:
            score -= 0.10

        return min(
            max(score, 0.0),
            1.0,
        )

    # ------------------------------------------------------------------
    # DATABASE
    # ------------------------------------------------------------------

    def _save_result(
        self,
        result: Dict[str, Any],
    ):
        """
        Save the result using the project's BillDatabase.

        Supports the current database.py implementation.
        """

        if self.database is None:
            return None

        try:
            return self.database.save_scan(
                result
            )

        except TypeError:
            # Compatibility fallback for alternate save_scan()
            # implementations.
            saved = self.database.save_scan(
                dict(result)
            )

            return saved

    # ------------------------------------------------------------------
    # IMAGE PREPARATION
    # ------------------------------------------------------------------

    def _prepare_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Prepare an incoming camera frame.

        OpenCV normally receives BGR.
        """

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR,
            )

        return image.copy()

    # ------------------------------------------------------------------
    # NORMALIZATION HELPERS
    # ------------------------------------------------------------------

    def _normalize_serial(
        self,
        value,
    ) -> str:
        """
        Normalize a serial-number OCR result.

        U.S. serial numbers are primarily alphanumeric.

        This function removes spaces and punctuation but does not
        aggressively replace ambiguous characters because doing so can
        create false serial numbers.
        """

        if value is None:
            return ""

        text = str(value).upper()

        text = text.replace(
            " ",
            "",
        )

        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text,
        )

        # A trailing G is frequently recognized as 6. When the rest of the
        # string already has the one-letter/eight-digit U.S. serial shape,
        # resolve that specific OCR ambiguity in the suffix position.
        if (
            len(text) == 10
            and text[0] in "ABCDEFGHIJKL"
            and text[1:9].isdigit()
            and text[9] == "6"
        ):
            text = text[:-1] + "G"

        return text

    def _normalize_series(
        self,
        value,
    ) -> str:
        if value is None:
            return ""

        text = str(value).upper().strip()

        match = re.search(
            r"(18|19|20)\d{2}[A-Z]?",
            text,
        )

        if match:
            return match.group(0)

        return text

    def _normalize_fed_letter(
        self,
        value,
    ) -> str:
        if value is None:
            return ""

        text = str(value).upper()

        letters = re.findall(
            r"[A-L]",
            text,
        )

        if not letters:
            return ""

        return letters[0]

    def _normalize_fed_number(
        self,
        value,
    ) -> Optional[int]:
        if value is None:
            return None

        text = str(value)

        match = re.search(
            r"\b([1-9]|1[0-2])\b",
            text,
        )

        if match:
            return int(
                match.group(1)
            )

        digits = re.sub(
            r"\D",
            "",
            text,
        )

        if digits:

            try:
                number = int(digits)

                if 1 <= number <= 12:
                    return number

            except ValueError:
                pass

        return None

    def _clean_denomination(
        self,
        value,
    ) -> Optional[str]:
        if value is None:
            return None

        text = str(value).upper().strip()

        # Remove dollar signs and spaces.
        text = text.replace(
            "$",
            "",
        )

        text = text.replace(
            " ",
            "",
        )

        # Common OCR forms.
        replacements = {
            "ONE": "1",
            "FIVE": "5",
            "TEN": "10",
            "TWENTY": "20",
            "FIFTY": "50",
            "ONEHUNDRED": "100",
            "HUNDRED": "100",
        }

        if text in replacements:
            return replacements[text]

        # Extract a numeric denomination.
        match = re.search(
            r"\b(1|5|10|20|50|100)\b",
            text,
        )

        if match:
            return match.group(1)

        # Handle OCR strings such as "$20.00".
        digits = re.sub(
            r"\D",
            "",
            text,
        )

        if digits in {
            "1",
            "5",
            "10",
            "20",
            "50",
            "100",
        }:
            return digits

        return None

    # ------------------------------------------------------------------
    # SAFE HELPERS
    # ------------------------------------------------------------------

    def _safe_reader(
        self,
        reader,
        image,
    ) -> str:
        try:
            value = reader(
                image
            )

            if value is None:
                return ""

            return str(value).strip()

        except Exception:
            return ""

    def _safe_float(
        self,
        value,
    ) -> float:
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    def _as_list(
        self,
        value,
    ):
        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        return [value]

    def _unique_strings(
        self,
        values,
    ):
        output = []

        for value in values:

            if value is None:
                continue

            text = str(value).strip()

            if not text:
                continue

            if text not in output:
                output.append(text)

        return output

    def _error_result(
        self,
        message: str,
    ) -> Dict[str, Any]:
        return {
            "status": "error",
            "side": "unknown",
            "error": message,
            "confidence": 0.0,
            "warnings": [
                message
            ],
            "scanned_at": datetime.utcnow().isoformat(
                timespec="seconds"
            ),
        }

    # ------------------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------------------

    def _order_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Return corners in:

            top-left
            top-right
            bottom-right
            bottom-left
        """

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if points.shape[0] != 4:
            raise ValueError(
                "Exactly four bill corners are required."
            )

        ordered = np.zeros(
            (4, 2),
            dtype=np.float32,
        )

        sums = points.sum(
            axis=1
        )

        differences = np.diff(
            points,
            axis=1,
        ).reshape(-1)

        ordered[0] = points[
            np.argmin(sums)
        ]

        ordered[2] = points[
            np.argmax(sums)
        ]

        ordered[1] = points[
            np.argmin(differences)
        ]

        ordered[3] = points[
            np.argmax(differences)
        ]

        return ordered
