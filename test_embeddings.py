import numpy as np

from embeddings import normalize, top_k

NAMES = ["cat", "dog", "car", "truck", "pizza", "pasta"]

VECTORS = np.array([
    [0.90, 0.10, 0.00],   # cat
    [0.80, 0.20, 0.00],   # dog
    [0.10, 0.90, 0.00],   # car
    [0.00, 0.95, 0.05],   # truck
    [0.00, 0.05, 0.90],   # pizza
    [0.05, 0.00, 0.85],   # pasta
])


def test_normalize_gives_length_one():
    lengths = np.linalg.norm(normalize(VECTORS), axis=1)
    assert np.allclose(lengths, 1.0)


def test_top_k_finds_animals_for_puppy():
    puppy = np.array([0.85, 0.10, 0.05])
    result = top_k(puppy, VECTORS, k=2)
    assert [NAMES[index] for index, _ in result] == ["cat", "dog"]


def test_top_k_orders_best_first_and_handles_big_k():
    van = np.array([0.05, 0.90, 0.05])
    result = top_k(van, VECTORS, k=100)       # more than the 6 that exist
    assert len(result) == 6
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)
    assert [NAMES[index] for index, _ in result][:2] == ["truck", "car"]