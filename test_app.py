import base64

import cv2
import numpy as np

import app as app_module


def test_scan_post_decodes_image_and_returns_reader_result(monkeypatch):
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    received = []

    def fake_scan_frame(image, side="auto"):
        received.append((image, side))
        return {"status": "ok", "denomination": 1}

    monkeypatch.setattr(app_module.reader, "scan_frame", fake_scan_frame)
    monkeypatch.setattr(app_module, "_tesseract_version", lambda: "5.4.0")

    response = app_module.app.test_client().post(
        "/api/scan",
        json={"image": data_url, "side": "front"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "denomination": 1}
    assert len(received) == 1
    assert received[0][0].shape == (8, 12, 3)
    assert received[0][1] == "front"


def test_scan_post_reports_missing_ocr_engine(monkeypatch):
    monkeypatch.setattr(app_module, "_tesseract_version", lambda: None)

    response = app_module.app.test_client().post(
        "/api/scan",
        json={"image": "unused"},
    )

    assert response.status_code == 503
    assert "Tesseract OCR is not installed" in response.get_json()["error"]


def test_scan_post_rejects_missing_image(monkeypatch):
    monkeypatch.setattr(app_module, "_tesseract_version", lambda: "5.4.0")
    response = app_module.app.test_client().post(
        "/api/scan",
        json={"other": "value"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "No image field was supplied."