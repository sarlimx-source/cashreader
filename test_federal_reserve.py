from cash_reader.federal_reserve import cross_check
def test_valid(): assert cross_check("L",12)["match"] is True
def test_conflict(): assert cross_check("L",2)["match"] is False
