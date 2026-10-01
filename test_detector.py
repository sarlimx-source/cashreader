import numpy as np

from cash_reader.detector import detect_bill
from cash_reader.pipeline import CashReader


def test_detect_bill_uses_full_frame_for_bill_shaped_crop():
    image = np.full((500, 1150, 3), 220, dtype=np.uint8)

    result = detect_bill(image)

    assert result["found"] is True
    assert result["confidence"] == 0.45
    assert result["message"] == "Using full image as a bill-shaped crop."
    assert result["warped"].shape[:2] == (650, 1500)


def test_normalize_serial_corrects_trailing_g_read_as_six():
    assert CashReader()._normalize_serial("L111809166") == "L11180916G"