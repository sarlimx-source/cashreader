from flask import Flask, render_template, request, jsonify
import base64, os, cv2, numpy as np
import pytesseract
from cash_reader.database import BillDatabase
from cash_reader.pipeline import CashReader

app = Flask(__name__)

# Make sure the data directory exists.
os.makedirs("data", exist_ok=True)

db = BillDatabase("data/bills.db")
reader = CashReader(database=db)


def _tesseract_version():
    """Return the installed OCR engine version, or None if unavailable."""
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def decode_image(data_url):
    """
    Convert a browser camera data URL into an OpenCV image.
    """

    if not data_url:
        raise ValueError("No image was provided.")

    if "," in data_url:
        encoded = data_url.split(",", 1)[1]
    else:
        encoded = data_url

    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Invalid base64 image data.") from exc

    image_array = np.frombuffer(raw, dtype=np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("Could not decode camera image.")

    return image


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    """
    Simple endpoint used to verify that Flask is alive.
    """

    version = _tesseract_version()
    return jsonify({
        "status": "ok",
        "message": "Cash Reader server is running.",
        "ocr_available": version is not None,
        "ocr_version": version,
    })


@app.post("/api/scan")
def scan():
    """
    Receive a camera image and analyze it.
    """

    print("SCAN REQUEST: POST /api/scan received", flush=True)

    try:
        ocr_version = _tesseract_version()
        if ocr_version is None:
            return jsonify({
                "status": "error",
                "error": "Tesseract OCR is not installed or not on PATH. Install Tesseract OCR and restart Cash Reader.",
            }), 503

        payload = request.get_json(silent=True)

        if not payload:
            return jsonify(
                {
                    "status": "error",
                    "error": "Request did not contain valid JSON.",
                }
            ), 400

        if "image" not in payload:
            return jsonify(
                {
                    "status": "error",
                    "error": "No image field was supplied.",
                }
            ), 400

        image = decode_image(payload["image"])

        side = payload.get("side", "front")

        # Use the scan_frame method from CashReader.
        if hasattr(reader, "scan_frame"):
            result = reader.scan_frame(image, side=side)

        elif hasattr(reader, "scan"):
            result = reader.scan(image, side=side)

        else:
            raise RuntimeError(
                "CashReader has no scan_frame() or scan() method."
            )

        # Make absolutely sure Flask receives a dictionary.
        if result is None:
            result = {
                "status": "error",
                "error": "Scanner returned no result.",
            }

        if not isinstance(result, dict):
            result = {
                "status": "ok",
                "result": result,
            }

        return jsonify(result)

    except Exception as exc:
        print("SCAN ERROR:", repr(exc))

        return jsonify(
            {
                "status": "error",
                "error": str(exc),
            }
        ), 500


@app.get("/api/recent")
def recent():
    """
    Return recently scanned bills.
    """

    try:
        try:
            limit = int(
                request.args.get(
                    "limit",
                    25,
                )
            )
        except (TypeError, ValueError):
            limit = 25

        limit = min(
            max(limit, 1),
            100,
        )

        results = db.recent(limit)

        if results is None:
            results = []

        return jsonify(
            {
                "status": "ok",
                "scans": results,
            }
        )

    except Exception as exc:
        print("RECENT ERROR:", repr(exc))

        return jsonify(
            {
                "status": "error",
                "error": str(exc),
                "scans": [],
            }
        ), 500


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    print()
    print("=" * 50)
    print("CASH READER")
    print("=" * 50)
    print(f"Starting Flask on port {port}")
    print("Health check: /api/health")
    print("=" * 50)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=False,
    )
