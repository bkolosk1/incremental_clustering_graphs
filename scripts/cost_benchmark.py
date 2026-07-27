"""Fair computational-cost benchmark for the R2.5 cost table.

Runs ONE (partition, method, mst) configuration per process so that peak RSS is
clean, and times graph construction end-to-end from the embeddings, broken into
its components. Crucially, and unlike the per-run `prep_sim_time_s` logged by the
main driver, the minimum-spanning-tree construction is timed here and attributed
to the MST-augmented methods (in the main driver the MST is built once in the
evaluator constructor and therefore is not part of any per-run timer).

Fairness policy
---------------
* t_sim  (dense N x N cosine similarity) is charged identically to every method:
  all constructions need the pairwise similarities to select neighbors. This is
  conservative for our method, which in principle need not materialise the full
  matrix but does so in this implementation.
* t_mst  (dense distance matrix + minimum spanning tree + edge re-weighting) is
  charged only to the MST-augmented methods -- this is the cost the earlier table
  hid.
* t_build is the sparse graph construction (+ the sparse MST combine for the
  augmented methods).
* t_construct = t_sim + t_mst + t_build  is the total wall time to turn the
  embeddings into the final affinity matrix.

All stage timers report the minimum over repeats (standard microbenchmark
practice: the minimum is the least noisy estimate of the intrinsic cost).
The construction code is imported from src/main.py, so it is byte-for-byte the
implementation used in the experiments.
"""
import argparse, json, os, resource, sys, time
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import main as M  # noqa: E402  -- the real _prepare_sim_matrix / _i64


def _best(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.monotonic()
        fn()
        ts.append(time.monotonic() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embs", required=True, help="path to the cached .npy embedding matrix")
    ap.add_argument("--task", required=True)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--n-clusters", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["NN", "MD"])
    ap.add_argument("--mst", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    embs = np.load(a.embs).astype(np.float32)
    N = embs.shape[0]
    e = torch.from_numpy(embs)
    e = e / e.norm(dim=1).view(-1, 1)

    # ---- t_sim: dense similarity, shared by all methods (faithful to main.py __init__) ----
    def build_sim():
        s = e @ e.T
        s.fill_diagonal_(-1)
        return s
    t_sim = _best(build_sim, a.repeats)
    sim = build_sim()

    # ---- t_mst: MST construction, charged to MST methods (faithful to main.py __init__) ----
    t_mst = 0.0
    mst_obj = None
    if a.mst:
        def build_mst():
            max_val = e.max()
            dist = max_val - sim
            msp = minimum_spanning_tree(dist)
            rows, cols = msp.nonzero()
            msp.data = sim[rows, cols].numpy()
            return M._i64(msp)
        t_mst = _best(build_mst, a.repeats)
        mst_obj = build_mst()

    # ---- t_build: sparse graph build (+ sparse MST combine, corrected/identity order) ----
    def build_graph():
        S = M._prepare_sim_matrix(sim, a.method, a.k, N)
        if a.mst:
            S = S.maximum(mst_obj).maximum(mst_obj.T)
        return S
    t_build = _best(build_graph, a.repeats)
    S = build_graph()
    nnz = int(S.nnz)
    n_comp = int(connected_components(S, directed=False, return_labels=False))

    # ---- t_spectral: the real spectral solve; skip if the graph is disconnected ----
    t_spectral = None
    spectral_note = None
    if n_comp == 1:
        try:
            import sklearn.manifold, sklearn.cluster
            km = sklearn.cluster.KMeans(n_clusters=a.n_clusters, n_init="auto", random_state=0)

            def spec():
                maps = sklearn.manifold.spectral_embedding(S, n_components=a.n_clusters, eigen_solver="amg", drop_first=True)
                km.fit(maps)
            t_spectral = _best(spec, max(1, a.repeats // 2))
        except Exception as exc:  # e.g. pyamg missing outside the container
            spectral_note = f"{type(exc).__name__}: {exc}"
    else:
        spectral_note = "disconnected graph; spectral solve not applicable"

    rec = dict(
        task=a.task, partition=a.partition, N=N, n_clusters=a.n_clusters,
        method=a.method, mst=bool(a.mst), k=a.k,
        t_sim=round(t_sim, 3), t_mst=round(t_mst, 3), t_build=round(t_build, 3),
        t_construct=round(t_sim + t_mst + t_build, 3),
        t_spectral=(round(t_spectral, 3) if t_spectral is not None else None),
        n_components=n_comp, connected=(n_comp == 1), spectral_note=spectral_note,
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        nnz=nnz,
    )
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
