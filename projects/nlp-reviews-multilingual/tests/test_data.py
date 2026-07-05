"""Smoke tests for data loading."""

from src.data import LABELS, LABEL_TO_ID


def test_labels_contract():
    assert LABELS == ["negative", "neutral", "positive"]
    assert len(LABEL_TO_ID) == 3
    for i, name in enumerate(LABELS):
        assert LABEL_TO_ID[name] == i


def test_star_to_label_boundary_cases():
    from data_download import star_to_label
    assert star_to_label(1) == "negative"
    assert star_to_label(2) == "negative"
    assert star_to_label(3) == "neutral"
    assert star_to_label(4) == "positive"
    assert star_to_label(5) == "positive"
