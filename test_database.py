from cash_reader.database import BillDatabase


def test_save_and_find_bill(tmp_path):
    db = BillDatabase(tmp_path / "bills.db")

    result = {
        "denomination": 20,
        "serial_number": "ML12345678A",
        "serial_left_read": "ML12345678A",
        "serial_right_read": "ML12345678A",
        "serial_match": True,
        "series": "2017A",
        "federal_reserve": {
            "letter": "L",
            "number": 12,
            "district": "San Francisco",
            "city": "San Francisco",
            "match": True,
        },
        "confidence": 0.97,
        "status": "ok",
        "warnings": [],
    }

    bill_id = db.save_scan(result)
    assert bill_id == 1
    assert db.count() == 1

    found = db.find_by_serial("ml12345678a")
    assert len(found) == 1
    assert found[0]["denomination"] == 20
    assert found[0]["series"] == "2017A"
    assert found[0]["federal_reserve_letter"] == "L"
    assert found[0]["federal_reserve_number"] == 12
