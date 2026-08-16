"""
ts_smote.py
-----------
A minimal, dependency-free (sklearn-only) re-implementation of SMOTE, adapted
to be safe for time-series classification:

  1. Oversampling is only ever applied *inside* a single walk-forward training
     fold, strictly AFTER the temporal train/test split has been made. The
     test fold is never touched, and no synthetic point is ever generated
     using a test-fold neighbour, so nothing about the future leaks into
     training.
  2. Synthetic points are convex combinations of a minority-class sample and
     one of its k nearest minority-class neighbours *within the same training
     fold*, exactly as in the original SMOTE (Chawla et al., 2002); we do not
     use calendar time as a feature, so a synthetic point has no fixed
     'time' of its own; it is simply an additional training example.
  3. An ADASYN-style variant weights how many synthetic neighbours to
     generate per minority sample by how hard that sample is to classify
     (density of majority-class neighbours around it), focusing synthetic
     samples on the minority points closest to the decision boundary.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X, y, minority_label=1, k_neighbors=5, target_ratio=0.3, random_state=0):
    """Standard SMOTE. target_ratio = desired minority_count / total_count
    after oversampling (only increases minority count, never removes majority
    rows)."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    min_idx = np.where(y == minority_label)[0]
    maj_idx = np.where(y != minority_label)[0]
    n_min, n_maj = len(min_idx), len(maj_idx)
    if n_min < 2 or n_min >= n_maj:
        return X, y
    n_target_min = int(target_ratio / (1 - target_ratio) * n_maj)
    n_to_generate = max(n_target_min - n_min, 0)
    if n_to_generate == 0:
        return X, y

    k = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X[min_idx])
    _, neigh = nn.kneighbors(X[min_idx])

    synth = np.zeros((n_to_generate, X.shape[1]))
    for t in range(n_to_generate):
        i = rng.integers(0, n_min)
        j = neigh[i, rng.integers(1, k + 1)]  # skip self (col 0)
        lam = rng.uniform(0, 1)
        synth[t] = X[min_idx[i]] + lam * (X[min_idx[j]] - X[min_idx[i]])

    X_out = np.vstack([X, synth])
    y_out = np.concatenate([y, np.full(n_to_generate, minority_label)])
    return X_out, y_out


def adasyn_oversample(X, y, minority_label=1, k_neighbors=5, target_ratio=0.3, random_state=0,
                       max_pool=40_000):
    """ADASYN-style oversampling: allocate more synthetic samples to minority
    points that have more majority-class neighbours (i.e. are harder / closer
    to the boundary). For large datasets, the difficulty score r_i is
    estimated using a random subsample of the majority class (capped at
    `max_pool` majority rows, combined with all minority rows) purely for
    computational tractability -- this does not touch the test fold and does
    not change which fold the search pool comes from, so it introduces no
    additional leakage risk beyond ordinary SMOTE/ADASYN."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    min_idx = np.where(y == minority_label)[0]
    maj_idx = np.where(y != minority_label)[0]
    n_min, n_maj = len(min_idx), len(maj_idx)
    if n_min < 2 or n_min >= n_maj:
        return X, y
    n_target_min = int(target_ratio / (1 - target_ratio) * n_maj)
    n_to_generate = max(n_target_min - n_min, 0)
    if n_to_generate == 0:
        return X, y

    k = min(k_neighbors, n_min - 1)
    if n_maj > max_pool:
        maj_pool = rng.choice(maj_idx, size=max_pool, replace=False)
    else:
        maj_pool = maj_idx
    pool_idx = np.concatenate([min_idx, maj_pool])
    nn_all = NearestNeighbors(n_neighbors=k).fit(X[pool_idx])
    _, neigh_all = nn_all.kneighbors(X[min_idx])  # indices into pool_idx
    maj_pool_set = set(range(n_min, len(pool_idx)))  # positions of majority rows within pool_idx
    r_i = np.array([sum(1 for nb in neigh_all[i] if nb in maj_pool_set) / k for i in range(n_min)])
    if r_i.sum() == 0:
        weights = np.full(n_min, 1.0 / n_min)
    else:
        weights = r_i / r_i.sum()
    g_i = np.round(weights * n_to_generate).astype(int)

    nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X[min_idx])
    _, neigh_min = nn_min.kneighbors(X[min_idx])

    synth_list = []
    for i in range(n_min):
        for _ in range(g_i[i]):
            j = neigh_min[i, rng.integers(1, k + 1)]
            lam = rng.uniform(0, 1)
            synth_list.append(X[min_idx[i]] + lam * (X[min_idx[j]] - X[min_idx[i]]))
    if not synth_list:
        return X, y
    synth = np.vstack(synth_list)
    X_out = np.vstack([X, synth])
    y_out = np.concatenate([y, np.full(len(synth_list), minority_label)])
    return X_out, y_out
