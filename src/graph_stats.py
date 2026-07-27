"""Descriptive graph statistics for the incremental kNN graphs (R1.2-a + R2.5).

Operates on a scipy.sparse adjacency. Returns a flat dict that the runner serializes alongside the V-measure result. Heavy networkx metrics (transitivity, assortativity, average clustering) are skipped when the graph exceeds `heavy_max_nodes`; their keys are still emitted with value `None` so downstream CSV/JSON consumers see a stable schema.
"""
import math
import time
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components


def compute(S, *, heavy_max_nodes=20000):
    """Return descriptive stats for sparse adjacency `S`.

    `S` is expected to be the same matrix fed to spectral_embedding (already mst-augmented if applicable). We symmetrize via element-wise max before computing undirected stats; original asymmetry is preserved in `nnz`.
    """
    S = S.tocsr()
    N = S.shape[0]
    t0 = time.monotonic()
    out = {"n_nodes": int(N), "nnz": int(S.nnz)}

    Sym = S.maximum(S.T).tocsr()
    n_und = int(Sym.nnz // 2)
    out["n_edges_undirected"] = n_und
    out["density"] = float((2.0 * n_und) / (N * (N - 1))) if N > 1 else 0.0

    deg = np.asarray(Sym.astype(bool).sum(axis=1)).ravel()
    out["max_degree"] = int(deg.max()) if N > 0 else 0
    out["avg_degree"] = float(deg.mean()) if N > 0 else 0.0
    out["std_degree"] = float(deg.std()) if N > 0 else 0.0

    nc, comp_labels = connected_components(Sym, directed=False)
    out["n_connected_components"] = int(nc)
    if nc > 0:
        _, counts = np.unique(comp_labels, return_counts=True)
        out["largest_component_size"] = int(counts.max())
    else:
        out["largest_component_size"] = 0

    if N <= heavy_max_nodes:
        G = nx.from_scipy_sparse_array(Sym)
        out["transitivity"] = _safe(nx.transitivity, G)
        out["avg_clustering"] = _safe(nx.average_clustering, G)
        out["assortativity"] = _safe(nx.degree_assortativity_coefficient, G)
    else:
        out["transitivity"] = None
        out["avg_clustering"] = None
        out["assortativity"] = None
        out["heavy_stats_skipped"] = True

    out["stats_time_s"] = round(time.monotonic() - t0, 3)
    return out


def homophily(S, labels):
    """Fraction of edges (i,j) with labels[i] == labels[j]. Sparse-only.

    R1.2-a explicitly asks for homophily across seeds. Cheap enough to compute always (no networkx, no triangle counting).
    """
    Sym = S.maximum(S.T).tocoo()
    if Sym.nnz == 0:
        return 0.0
    lab = np.asarray(labels)
    same = int((lab[Sym.row] == lab[Sym.col]).sum())
    return float(same) / float(Sym.nnz)


def _safe(fn, *args, **kwargs):
    """Run `fn`, return a finite float or None.

    networkx's assortativity / clustering / transitivity routines return NaN on degenerate cases (regular graphs, isolated nodes). NaN is not valid per RFC 8259 and many JSONL consumers (jq strict mode, Spark) reject the literal `NaN` token. We coerce to None here so the runner's `json.dumps(..., allow_nan=False)` stays happy.
    """
    try:
        v = float(fn(*args, **kwargs))
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v
