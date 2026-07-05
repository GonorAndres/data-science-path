"""Metric definition tests."""

from src.evaluate import ndcg_at_k, recall_at_k


def test_recall_hit():
    assert recall_at_k([1, 2, 3, 4, 5], 3, k=5) == 1.0


def test_recall_miss():
    assert recall_at_k([1, 2, 3, 4, 5], 99, k=5) == 0.0


def test_ndcg_rank_1():
    assert ndcg_at_k([3, 1, 2], 3, k=3) == 1.0


def test_ndcg_rank_2():
    expected = 1.0 / 1.585  # 1 / log2(3)
    assert abs(ndcg_at_k([1, 3, 2], 3, k=3) - expected) < 0.01
