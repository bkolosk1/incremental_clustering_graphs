"""Incremental spectral clustering of text embeddings -- MLJ rebuttal driver.

Usage (typical rebuttal sweep, drive via container/run.sh):
  container/run.sh src/main.py \
    --models all-MiniLM-L6-v2 all-MiniLM-L12-v2 all-mpnet-base-v2 BAAI/bge-base-en-v1.5 thenlper/gte-large \
    --datasets TwentyNewsgroupsClustering StackExchangeClustering StackExchangeClusteringP2P RedditClustering RedditClusteringP2P ArxivClusteringS2S ArxivClusteringP2P BiorxivClusteringS2S BiorxivClusteringP2P MedrxivClusteringS2S MedrxivClusteringP2P \
    --methods MD NN \
    --ks 1 2 3 6 8 10 15 20 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --ordering random \
    --mst both \
    --output-dir results/rebuttal \
    --embed-cache $HOME/clustering_graphs/cache/embs

What changed vs the original submission's main.py (R2.6 prerequisites):
  * No hardcoded paths. Replaces `path = "/cache/boshkok/graphs_marko/"`.
  * Full argparse CLI. Replaces module-top literal-flag config.
  * `--seeds` list drives the variance grid in-script (was `for i in range(1):`).
  * `--mst {off,on,both}` covers the NN+MST baseline reviewer R2 asked for.
  * Embedding cache keyed by (model, task, partition, n) -- skip re-encoding.
  * Graph statistics (density, assortativity, transitivity, ...) + wall time + peak RSS go into every per-run JSON line for R1.2-a + R2.5.
  * The bare `except: pass` that swallowed sweep failures is replaced with a logged error and a failure record so a sweep can never "succeed" with zero rows.
  * Pickling of the sparse graph is gated by `--save-graphs <dir>`; off by default.
"""
import argparse
import hashlib
import json
import logging
import os
import pathlib
import resource
import sys
import time
import traceback

# Make `src/` importable when this file is run directly without PYTHONPATH set (R2.6: fresh clone + `python src/main.py` outside the SIF must work). Inside container/run.sh, PYTHONPATH is already /workspace/src so this is a no-op.
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import networkx as nx
import numpy as np
import sklearn
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics.cluster
import torch
import tqdm
import mteb  # noqa: F401  -- imported for version pinning + side-effect-free
import mteb.tasks  # noqa: F401  -- REQUIRED: registers concrete AbsTaskClustering subclasses; without this `AbsTaskClustering.__subclasses__()` is empty and the dataset filter matches nothing.
from mteb import MTEB
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.evaluation.evaluators.Evaluator import Evaluator
from scipy.linalg import qr, svd
from scipy.sparse import csr_matrix, lil_matrix, csgraph
from scipy.sparse.csgraph import minimum_spanning_tree
from sentence_transformers import SentenceTransformer

# Eager import: sklearn.manifold.spectral_embedding(eigen_solver='amg') silently degrades to a much slower arpack solver if pyamg is missing. Surface that here, loudly, instead of after the sweep has burned half an hour on a doomed run.
import pyamg  # noqa: F401

from ordering import get_ordering
import graph_stats

logger = logging.getLogger("clustgraphs")


def cluster_qr(vectors):
    """QR/SVD discretization step (kept for parity with the submitted code; unused in the default path)."""
    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors[:, 1:].argmax(axis=1)


def _i64(M):
    """Coerce a sparse matrix's index arrays to int64.

    scipy computes intermediate flat indices (row*ncols + col) in the dtype of
    the matrix's `indices`/`indptr` arrays. For a large graph those default to
    int32, and once ncols*nrows crosses 2**31 (i.e. N > ~46,340) the arithmetic
    overflows -> garbage indices -> a ValueError in the COO check, or worse a
    heap corruption / core dump (observed on RedditClusteringP2P partition 9,
    N=62,261). Forcing int64 indices makes every downstream binary op (+, .T,
    .maximum) use 64-bit arithmetic and removes the overflow.
    """
    M = M.tocsr()
    M.indptr = M.indptr.astype(np.int64, copy=False)
    M.indices = M.indices.astype(np.int64, copy=False)
    return M


def _prepare_sim_matrix(sim, method, n_neighbors, size):
    """Build the sparse affinity from a dense similarity tensor `sim` (size x size).

    The five constructors match the paper:
      NN  -- symmetric full kNN: every node keeps its top-k, then S = 0.5*(S + S.T)
      ND  -- weighted symmetric: edge weight = (sim+1)/2 in (0,1]
      MD  -- incremental (paper's main contribution): for i >= k, only top-k from sim[i, :i]
      DD  -- exp-weighted incremental: weight = exp(2*sim - 2)
      MY  -- weighted incremental: weight = (sim+1)/2 on the MD support
    """
    S = lil_matrix((size, size))
    if method == "NN":
        for i in range(size):
            ttk = (sim[i]).topk(n_neighbors)
            indices = ttk.indices.numpy().tolist()
            S[i, indices] = [1] * n_neighbors
        S = _i64(S.tocsr())
        S = 0.5 * (S + S.T)
    elif method == "ND":
        for i in range(size):
            ttk = (sim[i]).topk(n_neighbors)
            indices = ttk.indices.numpy().tolist()
            S[i, indices] = S[indices, i] = ((ttk.values + 1) / 2).numpy().tolist()
        S = S.tocsr()
    elif method == "MD":
        for i in range(n_neighbors, size):
            ttk = (sim[i, :i]).topk(n_neighbors)
            indices = ttk.indices.numpy().tolist()
            S[i, indices] = S[indices, i] = [1] * n_neighbors
        S = S.tocsr()
    elif method == "DD":
        for i in range(n_neighbors, size):
            ttk = (sim[i, :i]).topk(n_neighbors)
            indices = ttk.indices.numpy().tolist()
            S[i, indices] = S[indices, i] = np.exp(2 * ttk.values.numpy() - 2).tolist()
        S = S.tocsr()
    elif method == "MY":
        for i in range(n_neighbors, size):
            ttk = (sim[i, :i]).topk(n_neighbors)
            indices = ttk.indices.numpy().tolist()
            S[i, indices] = S[indices, i] = ((ttk.values + 1) / 2).numpy().tolist()
        S = S.tocsr()
    else:
        raise ValueError(f"unknown method {method!r}; expected one of NN, ND, MD, DD, MY")
    return _i64(S)


class ClusteringEvaluator(Evaluator):
    """One MTEB cluster_set -> one ClusteringEvaluator -> many (method, k, mst, seed) runs."""

    def __init__(
        self,
        sentences,
        labels,
        corpus_embeddings,
        data_name=0,
        compute_mst=False,
        limit=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
            corpus_embeddings = corpus_embeddings[:limit]
        self.corpus_embeddings = corpus_embeddings
        self.sentences = sentences
        self.labels = labels
        embs = torch.from_numpy(np.asarray(self.corpus_embeddings, dtype=np.float32))
        embs = embs / embs.norm(dim=1).view(-1, 1)
        self.sim = embs @ embs.T
        self.mst = None
        if compute_mst:
            max_val = embs.max()
            dist = max_val - self.sim
            mst_sparse = minimum_spanning_tree(dist)
            rows, cols = mst_sparse.nonzero()
            mst_sparse.data = self.sim[rows, cols].numpy()
            self.mst = _i64(mst_sparse)  # int64 indices so the .maximum() combine with a large permuted S doesn't overflow (see _i64)
        self.data_name = data_name

    def __call__(self, *args, **kwargs):
        # MTEB's `Evaluator` base class declares `__call__` abstract; we satisfy that here. The original submitted code used `__call__` as the work method, but the new keyword-only signature is cleaner as `run(...)`. Keep `__call__` as a thin alias so the ABC is satisfied AND legacy `evaluator(...)` invocations continue to work.
        return self.run(*args, **kwargs)

    def run(self, *, method, n_neighbors, mst, mst_mode, ordering_name, seed, dates, save_graph_path, heavy_max_nodes, emit_preds, stats_only=False):
        size = len(self.corpus_embeddings)
        n_clusters = len(set(self.labels))
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init="auto", random_state=int(seed))
        rng = np.random.default_rng(int(seed))

        order = get_ordering(
            ordering_name,
            N=size,
            k_seed=n_neighbors,
            embeddings=np.asarray(self.corpus_embeddings),
            labels=self.labels,
            rng=rng,
            dates=dates,
        )

        sim = self.sim[np.ix_(order, order)]
        sim.fill_diagonal_(-1)

        t_prep_0 = time.monotonic()
        S = _prepare_sim_matrix(sim, method, n_neighbors, size)
        if mst:
            if self.mst is None:
                raise RuntimeError("mst=True but ClusteringEvaluator(compute_mst=False); pass --mst on/both at top level")
            if mst_mode == "paper-faithful":
                # PAPER-FAITHFUL: the original submitted code combined `S` (in permuted index space) with `self.mst` (in ORIGINAL index space) via element-wise max. That's a mathematical mismatch, but it is what produced tables 4/5/6 in the submission. We preserve it as the default so the rebuttal additions (NN+MST etc.) are commensurable with the paper's published MD+MST numbers. The discrepancy is documented in the rebuttal cover letter.
                S = S.maximum(self.mst).maximum(self.mst.T)
            elif mst_mode == "corrected":
                # CORRECTED: permute the MST into the same index space as S before combining. Mathematically sound but produces different numbers from the published tables. Only use this if you intend to re-publish the MD+MST and NN+MST rows.
                mst_perm = _permute_csr(self.mst.tocsr(), order)
                S = S.maximum(mst_perm).maximum(mst_perm.T)
            else:
                raise ValueError(f"unknown mst_mode {mst_mode!r}")
        t_prep = time.monotonic() - t_prep_0

        # STATS-ONLY (R1.2-a): compute graph statistics on S and return WITHOUT the
        # expensive spectral_embedding + KMeans. The graph is deterministic given
        # (method, k, seed, ordering), so this yields the per-seed graph-property
        # variance the reviewer asked for at a fraction of the cost of a full run.
        if stats_only:
            true_labels_perm = np.array(self.labels)[order]
            stats = graph_stats.compute(S, heavy_max_nodes=heavy_max_nodes)
            stats["homophily"] = graph_stats.homophily(S, true_labels_perm)
            return {
                "method": method, "n_neighbors": int(n_neighbors), "n_clusters": int(n_clusters),
                "mst": bool(mst), "mst_mode": mst_mode if mst else None, "ordering": ordering_name,
                "seed": int(seed), "partition_idx": int(self.data_name),
                "graph_stats": stats, "prep_sim_time_s": round(t_prep, 3), "stats_only": True,
            }

        t_spec_0 = time.monotonic()
        maps = sklearn.manifold.spectral_embedding(S, n_components=n_clusters, eigen_solver="amg", drop_first=True)
        kmeans.fit(maps)
        cluster_assignment = kmeans.labels_
        t_spec = time.monotonic() - t_spec_0

        true_labels_perm = np.array(self.labels)[order]
        H, C, V = sklearn.metrics.cluster.homogeneity_completeness_v_measure(true_labels_perm, cluster_assignment)

        stats = graph_stats.compute(S, heavy_max_nodes=heavy_max_nodes)
        stats["homophily"] = graph_stats.homophily(S, true_labels_perm)

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if save_graph_path is not None:
            save_graph_path.parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(save_graph_path, "wb") as f:
                pickle.dump(S, f)

        rec = {
            "homogeneity": float(H),
            "completeness": float(C),
            "v_measure": float(V),
            "method": method,
            "n_neighbors": int(n_neighbors),
            "n_clusters": int(n_clusters),
            "mst": bool(mst),
            "mst_mode": mst_mode if mst else None,
            "ordering": ordering_name,
            "seed": int(seed),
            "partition_idx": int(self.data_name),
            "graph_stats": stats,
            "prep_sim_time_s": round(t_prep, 3),
            "spectral_time_s": round(t_spec, 3),
            "peak_rss_kb": int(rss_kb),
            "graph_path": str(save_graph_path) if save_graph_path is not None else None,
        }
        if emit_preds:
            rec["labels"] = true_labels_perm.tolist()
            rec["preds"] = cluster_assignment.tolist()
        return rec


def _permute_csr(M, order):
    """Return M[order, :][:, order] efficiently for a CSR-friendly path."""
    M = M.tocsr()[order, :].tocsc()[:, order].tocsr()
    return M


class AbsSpectralTaskClustering(AbsTask):
    """MTEB task wrapper that replaces the default classify-then-cluster path with our spectral pipeline.

    Important: this class is mixed into each concrete `AbsTaskClustering` subclass at runtime by `build_spectralized_tasks`. Do not instantiate directly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(
        self,
        model,
        split="test",
        *,
        model_name,
        methods,
        ks,
        mst_modes,
        mst_combine_mode,
        seeds,
        ordering_name,
        dates,
        compute_mst,
        embed_cache_dir,
        limit,
        save_graphs_dir,
        heavy_max_nodes,
        emit_preds,
        result_writer,
        num_shards=1,
        shard_id=0,
        encode_only=False,
        require_cache=False,
        max_partition_nodes=None,
        stats_only=False,
        **kwargs,
    ):
        if not self.data_loaded:
            self.load_data()

        task_name = self.description["name"]
        res = {m: {k: [] for k in ks} for m in methods}

        for partition_idx, cluster_set in enumerate(tqdm.tqdm(self.dataset[split], desc=task_name)):
            # Partition-level sharding: each shard processes 1/num_shards of the partitions per task.
            # All shards see the same task list; they interleave on partition_idx modulo num_shards.
            # Result files carry the PID suffix so shard outputs are naturally separated.
            if num_shards > 1 and (partition_idx % num_shards) != shard_id:
                continue
            try:
                sentences = cluster_set["sentences"]
                labels = cluster_set["labels"]
                if limit is not None:
                    sentences = sentences[:limit]
                    labels = labels[:limit]
                # Skip partitions too large for scipy's 32-bit sparse indices. Above N ~= 46,340 the
                # flat index row*N+col exceeds 2**31 and scipy's binary ops (add, .T, .maximum,
                # .nonzero->tocoo) overflow -> heap corruption / core dump. Coercing indices to int64
                # only moves the crash to the next op (whack-a-mole), and these giant graphs are also
                # computationally intractable for the full k x seed grid. We skip them and record a
                # marker; downstream the dataset mean is taken over the remaining partitions (documented).
                if max_partition_nodes is not None and len(sentences) > max_partition_nodes:
                    logger.warning("SKIP %s p%d: N=%d exceeds --max-partition-nodes=%d (int32 sparse limit); recording skip marker", task_name, partition_idx, len(sentences), max_partition_nodes)
                    result_writer({"task": task_name, "model": model_name, "partition_idx": partition_idx, "skipped": True, "reason": "exceeds_max_partition_nodes", "n_sentences": len(sentences)})
                    continue
                embs = _encode_with_cache(model, model_name, task_name, partition_idx, sentences, embed_cache_dir, require_cache=require_cache)
                if encode_only:
                    logger.info("encode-only: %s p%d cached (n=%d, dim=%d)", task_name, partition_idx, len(sentences), embs.shape[1])
                    continue
                evaluator = ClusteringEvaluator(
                    sentences, labels, corpus_embeddings=embs, data_name=partition_idx, compute_mst=compute_mst,
                )
                partition_dates = dates.get((task_name, partition_idx)) if dates is not None else None
                for method in methods:
                    for k in tqdm.tqdm(ks, leave=False, desc=f"{method} k"):
                        for mst_flag in mst_modes:
                            for seed in seeds:
                                save_path = None
                                if save_graphs_dir is not None:
                                    save_path = save_graphs_dir / model_name.replace("/", "_") / f"{task_name}__p{partition_idx}__{method}__k{k}__mst{int(mst_flag)}__s{seed}.pkl"
                                rec = evaluator.run(
                                    method=method,
                                    n_neighbors=k,
                                    mst=mst_flag,
                                    mst_mode=mst_combine_mode,
                                    ordering_name=ordering_name,
                                    seed=seed,
                                    dates=partition_dates,
                                    save_graph_path=save_path,
                                    heavy_max_nodes=heavy_max_nodes,
                                    emit_preds=emit_preds,
                                    stats_only=stats_only,
                                )
                                rec["task"] = task_name
                                rec["model"] = model_name
                                result_writer(rec)
                                res[method][k].append(rec)
            except CacheMissError:
                raise
            except Exception:
                logger.exception("cluster_set %d of task %s failed; recording failure marker and continuing", partition_idx, task_name)
                result_writer({
                    "task": task_name,
                    "model": model_name,
                    "partition_idx": partition_idx,
                    "error": traceback.format_exc(),
                })
        return res


class CacheMissError(RuntimeError):
    """Raised in --require-cache mode when a partition's embeddings are not in the cache. Deliberately NOT swallowed by the per-partition failure handler: a missing cache means the encode phase wasn't run (or ran with different --limit/datasets), and every subsequent partition would fail the same way — fail fast instead of burning CPU-hours."""


def _sentences_fingerprint(sentences):
    """SHA1 over the full text stream so a re-released dataset with the same row count but different text doesn't return stale embeddings.

    Streaming hash with `\\0` separators avoids edge cases where two different sentence lists happen to concatenate to the same string. Cost: O(total chars), still negligible vs the encode itself.
    """
    h = hashlib.sha1()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _embed_cache_key(model_name, task_name, partition_idx, sentences_fp, n_sentences):
    return hashlib.sha1(f"{model_name}|{task_name}|{partition_idx}|{n_sentences}|{sentences_fp}".encode("utf-8")).hexdigest()[:16]


def _encode_with_cache(model, model_name, task_name, partition_idx, sentences, cache_dir, require_cache=False):
    if cache_dir is None:
        if require_cache:
            raise CacheMissError("--require-cache set but no --embed-cache directory given")
        return np.asarray(model.encode(sentences))
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _sentences_fingerprint(sentences)
    key = _embed_cache_key(model_name, task_name, partition_idx, fp, len(sentences))
    npy_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"
    if npy_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if (meta.get("model") == model_name
                and meta.get("task") == task_name
                and meta.get("partition") == partition_idx
                and meta.get("n_sentences") == len(sentences)
                and meta.get("sentences_fp") == fp):
                embs = np.load(npy_path)
                logger.info("embed-cache HIT %s/%s/p%d (n=%d, dim=%d)", model_name, task_name, partition_idx, len(sentences), embs.shape[1])
                return embs
            logger.warning("embed-cache key collision at %s (meta mismatch); recomputing", npy_path)
        except Exception:
            logger.exception("embed-cache read failed at %s; recomputing", npy_path)
    if require_cache:
        raise CacheMissError(f"embeddings not cached for {model_name}/{task_name}/p{partition_idx} (n={len(sentences)}, fp={fp}); run the encode phase first (--encode-only) with the SAME --datasets/--models/--limit")
    embs = np.asarray(model.encode(sentences))
    np.save(npy_path, embs)
    meta_path.write_text(json.dumps({
        "model": model_name,
        "task": task_name,
        "partition": partition_idx,
        "n_sentences": len(sentences),
        "sentences_fp": fp,
        "dim": int(embs.shape[1]),
    }, ensure_ascii=False))
    logger.info("embed-cache MISS -> wrote %s (n=%d, dim=%d)", npy_path, len(sentences), embs.shape[1])
    return embs


def build_spectralized_tasks(name_filter):
    """Dynamically build a _Spectralized subclass for every concrete AbsTaskClustering and filter by name."""
    tasks = []
    for cls in AbsTaskClustering.__subclasses__():
        spect_cls = type("_Spectralized_" + cls.__name__, (AbsSpectralTaskClustering, cls), {})
        tasks.append(spect_cls())
    if name_filter is not None:
        keep = set(name_filter)
        tasks = [t for t in tasks if t.description["name"] in keep]
    return tasks


def load_dates_file(path):
    """Optional JSON file mapping {task_name: {partition_idx: [date, date, ...]}}.

    Dates may be ISO strings or unix epoch numbers -- np.argsort handles both as long as they are comparable.
    """
    if path is None:
        return None
    raw = json.loads(pathlib.Path(path).read_text())
    out = {}
    for task_name, parts in raw.items():
        for k, v in parts.items():
            out[(task_name, int(k))] = v
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Incremental spectral clustering -- MLJ rebuttal driver")
    p.add_argument("--models", nargs="+", required=True, help="SentenceTransformer model names or local paths")
    p.add_argument("--datasets", nargs="+", required=True, help="MTEB clustering task names (e.g. TwentyNewsgroupsClustering)")
    p.add_argument("--methods", nargs="+", default=["MD"], choices=["NN", "ND", "MD", "DD", "MY"])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 6, 8, 10, 15, 20])
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="One run per seed; paper used 10 (use --seeds 0 1 2 3 4 5 6 7 8 9)")
    p.add_argument("--ordering", default="random", choices=["random", "centroid", "class", "temporal"])
    p.add_argument("--dates-file", default=None, help="JSON file with per-(task,partition) date arrays; required for --ordering temporal")
    p.add_argument("--mst", default="off", choices=["off", "on", "both"], help="off=no MST, on=MST only, both=run with and without")
    p.add_argument("--mst-mode", default="paper-faithful", choices=["paper-faithful", "corrected"], help="paper-faithful: combine S (permuted index space) with un-permuted MST (matches submitted tables 4/5/6 exactly, mathematically asymmetric); corrected: permute the MST into S's index space first (mathematically clean, will produce different numbers from the published tables)")
    p.add_argument("--output-dir", default="./results", help="Where result JSONL + manifest go")
    p.add_argument("--embed-cache", default=None, help="Optional directory to cache per-(model,task,partition,content) embeddings")
    p.add_argument("--save-graphs", default=None, help="Optional directory to pickle each sparse S (off by default; ~MB per graph)")
    p.add_argument("--limit", type=int, default=None, help="Optional truncation of sentences per partition (for smoke tests)")
    p.add_argument("--heavy-stats-max-nodes", type=int, default=20000, help="Skip networkx-heavy stats (transitivity, assortativity, clustering) above this size")
    p.add_argument("--emit-preds-labels", action="store_true", help="Include per-record `labels` and `preds` arrays in results.jsonl. Off by default because they balloon the file to multi-GB on large sweeps; H/C/V/graph_stats are always emitted regardless.")
    p.add_argument("--results-filename", default=None, help="Override the results JSONL filename. Default: results.<pid>.jsonl so parallel sweeps writing to the same --output-dir don't interleave.")
    p.add_argument("--num-shards", type=int, default=1, help="Split each task's partitions across N parallel shards; each shard processes partition_idx %% num_shards == shard_id. Use with a DP-N launcher that pins CUDA_VISIBLE_DEVICES.")
    p.add_argument("--shard-id", type=int, default=0, help="This shard's index in [0, num_shards).")
    p.add_argument("--encode-only", action="store_true", help="PHASE 1: encode every (dataset, partition) into --embed-cache and exit without running any clustering. Run this once on a GPU node; then run the experiments with --require-cache on cheap CPU nodes.")
    p.add_argument("--require-cache", action="store_true", help="PHASE 2: fail fast (CacheMissError) if any partition's embeddings are not already cached, instead of silently encoding on the fly. Guarantees experiment jobs never need a GPU. Must be paired with the same --datasets/--models/--limit the encode phase used.")
    p.add_argument("--max-partition-nodes", type=int, default=None, help="Skip partitions with more than N documents (records a skip marker). Set to 46340 to avoid scipy's 32-bit sparse-index overflow on very large graphs (the 5 giant RedditClusteringP2P partitions, 62k-92k nodes). Dataset means are then taken over the remaining partitions.")
    p.add_argument("--stats-only", action="store_true", help="R1.2-a: build each graph and compute descriptive statistics (density, assortativity, transitivity, avg clustering, homophily, ...) then SKIP spectral clustering. Cheap way to get per-seed graph-property variance without redoing the full sweep. Pair with a high --heavy-stats-max-nodes to enable the networkx-heavy stats.")
    p.add_argument("--split", default="test")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--mteb-err-log", default="./err.txt")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filename = args.results_filename or f"results.{os.getpid()}.jsonl"
    result_jsonl = output_dir / results_filename
    manifest_path = output_dir / f"manifest.{os.getpid()}.json"

    mst_modes = {"off": [False], "on": [True], "both": [False, True]}[args.mst]
    compute_mst = any(mst_modes)
    save_graphs_dir = pathlib.Path(args.save_graphs) if args.save_graphs else None
    dates = load_dates_file(args.dates_file)
    embed_cache_dir = pathlib.Path(args.embed_cache) if args.embed_cache else None

    manifest = {
        "argv": list(sys.argv) if argv is None else ["main.py", *argv],
        "args": vars(args),
        "pid": os.getpid(),
        "mteb_version": mteb.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    available = sorted({t.description["name"] for t in build_spectralized_tasks(None)})
    sanity_tasks = build_spectralized_tasks(args.datasets)
    if not sanity_tasks:
        raise SystemExit(f"No MTEB tasks matched --datasets {args.datasets}. Available ({len(available)} total): {available[:30]}{'...' if len(available) > 30 else ''}")
    logger.info("Will sweep %d task(s) across %d model(s): %s", len(sanity_tasks), len(args.models), [t.description["name"] for t in sanity_tasks])

    with open(result_jsonl, "a", encoding="utf-8") as fout:
        def writer(rec):
            # allow_nan=False so any leftover NaN/Inf surfaces here, not as invalid JSON tokens downstream.
            fout.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
            fout.flush()

        for model_name in args.models:
            logger.info("Loading model %s", model_name)
            model = SentenceTransformer(model_name)
            # Rebuild tasks per model: MTEB.run pops from self.tasks as it iterates, so a shared list gets drained across models.
            tasks = build_spectralized_tasks(args.datasets)
            evaluation = MTEB(tasks=tasks, err_logs_path=args.mteb_err_log)
            # NOTE: do NOT pass split=... here -- MTEB.run iterates splits internally and calls task.evaluate(model, split, **kwargs); a split= in kwargs would crash with "got multiple values for argument 'split'". Use eval_splits=[args.split] instead.
            evaluation.run(
                model,
                output_folder=str(output_dir / f"mteb_{model_name.replace('/', '_')}"),
                eval_splits=[args.split],
                methods=args.methods,
                verbosity=2,
                ks=args.ks,
                model_name=model_name,
                mst_modes=mst_modes,
                mst_combine_mode=args.mst_mode,
                seeds=args.seeds,
                ordering_name=args.ordering,
                dates=dates,
                compute_mst=compute_mst,
                embed_cache_dir=embed_cache_dir,
                limit=args.limit,
                save_graphs_dir=save_graphs_dir,
                heavy_max_nodes=args.heavy_stats_max_nodes,
                emit_preds=args.emit_preds_labels,
                result_writer=writer,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
                encode_only=args.encode_only,
                require_cache=args.require_cache,
                max_partition_nodes=args.max_partition_nodes,
                stats_only=args.stats_only,
            )

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.encode_only and embed_cache_dir is not None:
        log_cache_summary(embed_cache_dir)
    logger.info("Done. Results -> %s, manifest -> %s", result_jsonl, manifest_path)


def log_cache_summary(cache_dir):
    """Per-(model, task) partition counts in the embed cache — the encode phase's receipt."""
    from collections import Counter
    counts = Counter()
    for meta_path in pathlib.Path(cache_dir).glob("*.json"):
        try:
            m = json.loads(meta_path.read_text())
            counts[(m.get("model"), m.get("task"))] += 1
        except Exception:
            continue
    logger.info("embed-cache summary: %d cached partitions total", sum(counts.values()))
    for (model_name, task_name), n in sorted(counts.items()):
        logger.info("  %-45s %-32s %3d partitions", model_name or "?", task_name or "?", n)


if __name__ == "__main__":
    main()
