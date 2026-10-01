# Cash Reader v2

The reader now checks four identifier groups:

1. **Denomination/design**
2. **Serial number** — reads both serial-number locations and checks whether they agree
3. **Series year/design identifier**
4. **Federal Reserve indicator** — reads the letter and district number, then cross-checks them

Pipeline:
camera -> straighten note -> identify denomination/design -> select design-specific ROIs
-> OCR serial #1 + serial #2 -> compare serials -> OCR series -> OCR Fed letter + number
-> cross-check Fed district -> return result/warnings.

IMPORTANT:
- ROI coordinates in `cash_reader/profiles.py` are placeholders and must be calibrated
  with real notes for each denomination/design.
- Serial-number formats and printed locations can vary by note series/design.
- This is identification/OCR support, not counterfeit authentication.
- For production accuracy, replace/augment Tesseract with a trained OCR/object-detection
  model and confidence scoring.


## Bill database

Every scan result can be stored automatically in a local SQLite database at
`data/bills.db`. The database stores denomination, both serial-number reads,
serial-match status, series, Federal Reserve letter/number/district, Fed
cross-check status, confidence, warnings, status, and scan time.

Examples:

```bash
# Scan a photo and save the result automatically
python -m cash_reader.cli --image path/to/bill.jpg

# Show the 25 most recent scans
python -m cash_reader.cli --recent 25

# Look up a serial number
python -m cash_reader.cli --find-serial ML12345678A
```

SQLite is built into Python, so there is no separate database server to install.
The database file is created automatically the first time the reader runs.


## Phone camera web app

Run `pip install -r requirements.txt` then `python app.py`. In GitHub Codespaces, open the forwarded port 5000 on your phone. Allow camera access and tap **Scan Bill**. Results are saved to `data/bills.db`.

### OCR engine requirement

`pytesseract` is only a Python wrapper. The Tesseract OCR executable must also be installed and available on `PATH`, or scans cannot read serials, series, and Federal Reserve fields. Install it before starting the app:

- **Windows:** install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki), then restart the terminal and Cash Reader.
- **macOS:** `brew install tesseract`
- **Debian/Ubuntu/Codespaces:** `sudo apt-get update && sudo apt-get install -y tesseract-ocr`

The web page reports when the server is reachable but its OCR engine is missing. Scans default to the **front** of the bill, where serial numbers and Federal Reserve identifiers are printed; choose **back** only for a back-side design check.

The current web version uses the phone camera. External LEDs, rollers, position sensors, and a dedicated scanner camera are not yet controlled by the web app.
