def test_dummy_math():
    # A trivial smoke test
    assert 1 + 1 == 2


def test_imports_work():
    import enums
    assert hasattr(enums, "DataSplit")
