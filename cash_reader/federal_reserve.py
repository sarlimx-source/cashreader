FEDERAL_RESERVE_DISTRICTS = {
    "A": {"number": 1, "city": "Boston"},
    "B": {"number": 2, "city": "New York"},
    "C": {"number": 3, "city": "Philadelphia"},
    "D": {"number": 4, "city": "Cleveland"},
    "E": {"number": 5, "city": "Richmond"},
    "F": {"number": 6, "city": "Atlanta"},
    "G": {"number": 7, "city": "Chicago"},
    "H": {"number": 8, "city": "St. Louis"},
    "I": {"number": 9, "city": "Minneapolis"},
    "J": {"number": 10, "city": "Kansas City"},
    "K": {"number": 11, "city": "Dallas"},
    "L": {"number": 12, "city": "San Francisco"},
}
NUMBER_TO_LETTER = {v["number"]: k for k, v in FEDERAL_RESERVE_DISTRICTS.items()}

def cross_check(letter, number):
    out = {"letter": letter, "number": number, "match": None,
           "district": None, "city": None, "message": ""}
    if letter in FEDERAL_RESERVE_DISTRICTS:
        info = FEDERAL_RESERVE_DISTRICTS[letter]
        out["district"], out["city"] = info["number"], info["city"]
    if letter in FEDERAL_RESERVE_DISTRICTS and number in NUMBER_TO_LETTER:
        out["match"] = FEDERAL_RESERVE_DISTRICTS[letter]["number"] == number
        if out["match"]:
            out["message"] = f"{letter}/{number} agrees — {out['city']}."
        else:
            out["message"] = f"Conflict: {letter} should be district {out['district']}, not {number}."
    elif letter in FEDERAL_RESERVE_DISTRICTS:
        out["message"] = f"Read {letter}; expected district number {out['district']}."
    elif number in NUMBER_TO_LETTER:
        out["message"] = f"Read district {number}; expected letter {NUMBER_TO_LETTER[number]}."
    else:
        out["message"] = "Federal Reserve indicator not confidently identified."
    return out
