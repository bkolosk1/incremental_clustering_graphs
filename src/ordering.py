"""Document orderings for the incremental neighborhood-graph construction.

Each strategy returns an int permutation array of length N. The first k_seed entries become the receptive-field seeds in Algorithm 1; later entries are inserted one at a time and can only connect back to already-inserted nodes.

Strategies:
  random    -- uniform random permutation (paper default).
  centroid  -- first k_seed = nodes closest to embedding centroid (R1.2-b: tight ball worst case).
  class     -- first k_seed all drawn from a single ground-truth class (information-leak worst case).
  temporal  -- sort by an external timestamp array passed via --dates-file (R2.4).
"""
import numpy as np


def random_ordering(*, N, k_seed, embeddings, labels, rng, **kwargs):
    return rng.permutation(N)


def centroid_ordering(*, N, k_seed, embeddings, labels, rng, **kwargs):
    """First k_seed = closest to mean(embeddings); remaining shuffled.

    Forces the seed receptive field into a tight ball, the inverse of what the algorithm wants. If V-measure barely moves vs random, the method is robust to bad seeds.
    """
    if embeddings is None:
        raise ValueError("centroid ordering requires embeddings")
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    seed_idx = np.argsort(dists)[:k_seed]
    seed_set = set(int(i) for i in seed_idx.tolist())
    rest = np.array([i for i in range(N) if i not in seed_set], dtype=np.int64)
    rng.shuffle(rest)
    return np.concatenate([seed_idx.astype(np.int64), rest])


def class_ordering(*, N, k_seed, embeddings, labels, rng, **kwargs):
    """First k_seed all drawn from a single random class (information-leak worst case).

    Picks a class large enough to provide k_seed members; falls back to random if no such class exists (very small dataset, very large k).
    """
    if labels is None:
        raise ValueError("class ordering requires labels")
    labels_arr = np.asarray(labels)
    classes = list({int(c) if hasattr(c, "__int__") else c for c in labels_arr.tolist()})
    rng.shuffle(classes)
    seed_idx = None
    for c in classes:
        members = np.where(labels_arr == c)[0]
        if len(members) >= k_seed:
            seed_idx = rng.choice(members, size=k_seed, replace=False)
            break
    if seed_idx is None:
        return rng.permutation(N)
    seed_set = set(int(i) for i in seed_idx.tolist())
    rest = np.array([i for i in range(N) if i not in seed_set], dtype=np.int64)
    rng.shuffle(rest)
    return np.concatenate([seed_idx.astype(np.int64), rest])


def temporal_ordering(*, N, k_seed, embeddings, labels, rng, dates=None, **kwargs):
    """Sort by ascending timestamp. Caller must pass `dates` (length N)."""
    if dates is None:
        raise ValueError("temporal ordering requires a `dates` array (use --dates-file)")
    dates_arr = np.asarray(dates)
    if dates_arr.shape[0] != N:
        raise ValueError(f"dates length {dates_arr.shape[0]} != N {N}")
    return np.argsort(dates_arr).astype(np.int64)


ORDERINGS = {
    "random": random_ordering,
    "centroid": centroid_ordering,
    "class": class_ordering,
    "temporal": temporal_ordering,
}


def get_ordering(name, **kwargs):
    if name not in ORDERINGS:
        raise ValueError(f"unknown ordering {name!r}; options: {sorted(ORDERINGS)}")
    return ORDERINGS[name](**kwargs)
