from msci_rebalance.events import name_similarity, normalize_company_name


def test_normalize_company_name_removes_corporate_suffixes():
    assert normalize_company_name("Affirm Holdings, Inc. Class A") == "AFFIRM"


def test_name_similarity_handles_abbreviations():
    assert name_similarity("INTERACTIVE BKRS GRP A", "Interactive Brokers Group, Inc.") > 0.8

