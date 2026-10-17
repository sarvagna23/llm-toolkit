import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    # divide each vector by its own length, so every length becomes 1
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / lengths


def top_k(query: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    if k <= 0:
        return []

    query = normalize(query)
    vectors = normalize(vectors)

    scores = vectors @ query                  # one cosine score per stored vector
    k = min(k, len(scores))                   # asking for more than exist is fine

    # argpartition finds the smallest values, so negate to get the largest
    candidate_indexes = np.argpartition(-scores, k - 1)[:k]

    # the candidates come back unordered, so sort just those k, best first
    best_first = candidate_indexes[np.argsort(-scores[candidate_indexes])]

    return [(int(index), float(scores[index])) for index in best_first]