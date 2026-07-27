"""Regenerate every data-derived LaTeX table body in the manuscript.

One authoritative generator, reading the per-run JSONL records under
``results_arnes/`` (produced by ``src/main.py``) and the cost benchmark output
(``results_arnes/r25_cost/cost.jsonl`` from ``scripts/cost_benchmark.py``), and
writing the ``\\input``-ed table bodies under ``submission/``.

    python scripts/make_tables.py all          # regenerate everything
    python scripts/make_tables.py r23 r12b      # regenerate a subset

Tables and their sources
------------------------
    r23   -> tab_r23_body.tex           Table 7  (k-NN / k-NN+MST / Ours / Ours+MST)
    r25   -> tab_r25_body.tex           Table 8  (fair computational cost)
    r12b  -> tab_r12b_body.tex          Table 9  (ordering vs. clustering quality)
    r12a  -> tab_r12a_body.tex          Table 10 (graph statistics across orderings, k=1,6)
             tab_r12a_appendix_s2s.tex  Table B1 (all k, S2S)
             tab_r12a_appendix_p2p.tex  Table B2 (all k, P2P)

Frozen k-NN numbers (the no-MST baseline block of Table 7) are the published L12
V1 scores transcribed from the submitted manuscript's main results table; every
other block is aggregated from the ARNES runs. RedditClusteringP2P means over the
recomputed configurations exclude the five partitions above the sparse-solver
index limit; this is stated in the manuscript captions.
"""
import collections
import glob
import json
import math
import statistics
import sys

ROOT = __file__.rsplit("/scripts/", 1)[0]
RES = f"{ROOT}/results_arnes"
SUB = f"{ROOT}/submission"

COLS = ["ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S", "RedditClustering", "StackExchangeClustering", "TwentyNewsgroupsClustering", "ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P", "RedditClusteringP2P", "StackExchangeClusteringP2P"]
SHORT = {"ArxivClusteringS2S": "Arx-S2S", "BiorxivClusteringS2S": "Bio-S2S", "MedrxivClusteringS2S": "Med-S2S", "RedditClustering": "Red-S2S", "StackExchangeClustering": "SE-S2S", "TwentyNewsgroupsClustering": "20NG", "ArxivClusteringP2P": "Arx-P2P", "BiorxivClusteringP2P": "Bio-P2P", "MedrxivClusteringP2P": "Med-P2P", "RedditClusteringP2P": "Red-P2P", "StackExchangeClusteringP2P": "SE-P2P"}
KS = [1, 2, 3, 6, 8, 10, 15, 20]

# Frozen published L12 V1 (x100), submitted manuscript main-results table.
KNN_FROZEN = {
    1: [9.33, 3.24, 15.07, 3.43, 3.31, 7.19, 10.23, 3.63, 15.55, 6.67, 30.17],
    2: [21.73, 12.91, 26.09, 5.11, 38.83, 10.85, 28.49, 14.13, 28.18, 25.06, 35.49],
    3: [37.93, 26.43, 29.51, 26.99, 56.67, 14.67, 45.61, 30.16, 32.37, 54.03, 36.35],
    6: [38.85, 29.03, 31.71, 45.67, 60.48, 19.10, 46.60, 32.89, 34.22, 64.86, 37.01],
    8: [39.44, 30.14, 32.52, 49.05, 61.22, 23.10, 46.67, 33.50, 34.86, 65.08, 37.18],
    10: [39.45, 30.14, 32.80, 50.77, 61.36, 29.22, 46.69, 33.71, 35.07, 63.99, 37.19],
    15: [39.51, 30.47, 33.06, 53.79, 61.49, 38.32, 46.90, 34.11, 35.74, 63.12, 37.24],
    20: [39.46, 31.18, 33.43, 54.83, 61.45, 42.07, 46.83, 34.35, 35.76, 62.80, 37.27],
}


def _load_vmeasure(pattern):
    """(task, k) -> mean V1 (x100) over seeds/partitions, skipping skip-markers."""
    agg = collections.defaultdict(list)
    for f in glob.glob(pattern):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("skipped") or "v_measure" not in d:
                continue
            agg[(d["task"], d["n_neighbors"])].append(d["v_measure"])
    return {key: 100 * statistics.mean(v) for key, v in agg.items()}


def _write(name, lines):
    path = f"{SUB}/{name}"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def make_r23():
    frozen = {(c, k): KNN_FROZEN[k][i] for k in KS for i, c in enumerate(COLS)}
    nnmst = _load_vmeasure(f"{RES}/r23_nnmst_l12_corrected_arnes/shard*/results.*.jsonl")
    ours = _load_vmeasure(f"{RES}/r23_ours_md_l12_arnes/shard*/results.*.jsonl")
    oursm = _load_vmeasure(f"{RES}/r23_mdmst_l12_corrected_arnes/shard*/results.*.jsonl")
    blocks = [("k-NN", frozen), ("k-NN+MST", nnmst), ("Ours", ours), ("Ours+MST", oursm)]
    L = ["\\begin{tabular}{cccccccc|ccccc}", "\\hline",
         "{} & {} & \\multicolumn{6}{c|}{S2S} & \\multicolumn{5}{|c}{P2P} \\\\ \\hline",
         "{} & {k} & Arx & Bio & Med & Red & SE & 20NG & Arx & Bio & Med & Red & SE \\\\ \\hline"]
    for name, src in blocks:
        L.append(f"\\multirow{{8}}{{*}}{{\\rot{{{name}}}}}")
        for j, k in enumerate(KS):
            cells = []
            for c in COLS:
                v = src.get((c, k))
                cells.append(f"{v:.1f}" if v is not None else "--")
            lead = "" if j == 0 else "{}"
            L.append(f"{lead} & {k} & " + " & ".join(cells) + " \\\\")
        L[-1] += " \\hline"
    L.append("\\end{tabular}")
    _write("tab_r23_body.tex", L)


def make_r12b():
    def _load_homophily(pattern):
        agg = collections.defaultdict(list)
        for f in glob.glob(pattern):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("skipped"):
                    continue
                h = d.get("graph_stats", {}).get("homophily")
                if h is not None:
                    agg[(d["task"], d["n_neighbors"])].append(h)
        return {key: statistics.mean(v) for key, v in agg.items()}

    srcs = {n: _load_vmeasure(f"{RES}/r12b_{n}_l12_arnes/results.*.jsonl") for n in ("random", "centroid", "class")}
    homs = {n: _load_homophily(f"{RES}/r12b_{n}_l12_arnes/results.*.jsonl") for n in ("random", "centroid", "class")}
    order = ["BiorxivClusteringP2P", "BiorxivClusteringS2S", "MedrxivClusteringP2P", "MedrxivClusteringS2S", "StackExchangeClustering", "TwentyNewsgroupsClustering"]
    ks = [1, 2, 3, 6]
    L = ["\\begin{tabular}{llcccc|cccc}", "\\hline",
         "{} & {} & \\multicolumn{4}{c|}{$V_1$} & \\multicolumn{4}{c}{label homophily} \\\\",
         "Dataset & Ordering & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}6$ & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}6$ \\\\ \\hline"]
    for t in order:
        for i, o in enumerate(("random", "centroid", "class")):
            lead = f"\\multirow{{3}}{{*}}{{{SHORT[t]}}}" if i == 0 else ""
            vcells = [f"{srcs[o][(t, k)]:.1f}" if (t, k) in srcs[o] else "--" for k in ks]
            hcells = [f"{homs[o][(t, k)]:.3f}" if (t, k) in homs[o] else "--" for k in ks]
            L.append(f"{lead} & {o} & " + " & ".join(vcells) + " & " + " & ".join(hcells) + " \\\\")
        L.append("\\hline")
    L.append("\\end{tabular}")
    _write("tab_r12b_body.tex", L)


def make_r12a():
    STATS = ["transitivity", "avg_clustering", "assortativity", "homophily"]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    dens = collections.defaultdict(set)
    for f in glob.glob(f"{RES}/r12a_graphstats_l12_arnes/shard*/results.*.jsonl"):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("skipped"):
                continue
            gs = d.get("graph_stats", {})
            key = (d["task"], d.get("partition_idx"), d["n_neighbors"])
            for s in STATS:
                v = gs.get(s)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    cells[key][s].append(v)
            if gs.get("nnz") is not None:
                dens[key].add(gs["nnz"])
    violations = sum(1 for v in dens.values() if len(v) > 1)
    print(f"  [r12a] edge-count invariance: {len(dens) - violations}/{len(dens)} cells identical across orderings")

    rows = {}
    agg = collections.defaultdict(list)
    for (task, part, k), st in cells.items():
        for s, vals in st.items():
            if len(vals) >= 8:
                agg[(task, k, s)].append((statistics.mean(vals), statistics.stdev(vals)))
    for key, mv in agg.items():
        rows[key] = (statistics.mean(m for m, _ in mv), statistics.mean(sd for _, sd in mv))

    def cell(task, k, s):
        if (task, k, s) not in rows:
            return "--"
        m, sd = rows[(task, k, s)]
        # std as subscript (no ±) to save horizontal space
        return f"{m:.3f}$_{{{sd:.3f}}}$"

    # main body: k=1 and k=6
    ks_main = [1, 6]
    hdr = " & ".join(["Dataset"] + [f"\\multicolumn{{4}}{{c{'|' if i == 0 else ''}}}{{$k{{=}}{k}$}}" for i, k in enumerate(ks_main)])
    sub = " & ".join([""] + ["trans. & avg.\\ clust. & assort. & homoph."] * len(ks_main))
    L = ["\\begin{tabular}{l|cccc|cccc}", "\\hline", hdr + " \\\\", sub + " \\\\ \\hline"]
    for t in COLS:
        L.append(" & ".join([SHORT[t]] + [cell(t, k, s) for k in ks_main for s in STATS]) + " \\\\")
    L += ["\\hline", "\\end{tabular}"]
    _write("tab_r12a_body.tex", L)

    # appendix: all k, split S2S / P2P
    S2S = COLS[:6]
    P2P = COLS[6:]
    for name, group in [("s2s", S2S), ("p2p", P2P)]:
        A = ["\\begin{tabular}{llcccc}", "\\hline",
             "Dataset & $k$ & transitivity & avg.\\ clustering & assortativity & homophily \\\\ \\hline"]
        for t in group:
            for i, k in enumerate(KS):
                lead = f"\\multirow{{{len(KS)}}}{{*}}{{{SHORT[t]}}}" if i == 0 else ""
                A.append(f"{lead} & {k} & " + " & ".join(cell(t, k, s) for s in STATS) + " \\\\")
            A.append("\\hline")
        A.append("\\end{tabular}")
        _write(f"tab_r12a_appendix_{name}.tex", A)


def make_r25():
    recs = []
    for f in glob.glob(f"{RES}/r25_cost/cost.jsonl"):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        print("  [r25] no cost.jsonl yet -- run scripts/r25_cost_arnes.sbatch first; skipping")
        return
    label = {("MD", False): "Ours", ("MD", True): "Ours+MST", ("NN", False): "k-NN", ("NN", True): "k-NN+MST"}
    row_order = [("MD", False), ("MD", True), ("NN", False), ("NN", True)]
    by_task = collections.defaultdict(dict)
    Ns = {}
    for r in recs:
        by_task[r["task"]][(r["method"], r["mst"])] = r
        Ns[r["task"]] = r["N"]
    disp = {"StackExchangeClustering": "StackExchange (S2S)", "ArxivClusteringP2P": "Arxiv (P2P)"}
    tasks = [t for t in ["StackExchangeClustering", "ArxivClusteringP2P"] if t in by_task]
    L = ["\\begin{tabular}{llrrrr}", "\\hline",
         "Dataset & Method & construct (s) & spectral (s) & peak RSS (GB) & nnz \\\\ \\hline"]
    for t in tasks:
        name = f"{disp.get(t, t)}{{,}} $N\\!\\approx\\!{round(Ns[t] / 1000)}$k"
        for i, key in enumerate(row_order):
            r = by_task[t].get(key)
            if r is None:
                continue
            lead = f"\\multirow{{4}}{{*}}{{{name}}}" if i == 0 else ""
            spec = f"{r['t_spectral']:.1f}" if r.get("t_spectral") is not None else "n/a"
            nnz = f"{r['nnz']:,}".replace(",", "{,}")
            L.append(f"{lead} & {label[key]} & {r['t_construct']:.1f} & {spec} & {r['peak_rss_gb']:.1f} & {nnz} \\\\")
        L.append("\\hline")
    L.append("\\end{tabular}")
    _write("tab_r25_body.tex", L)


def make_bayes():
    """tab_bayes_body.tex — Bayesian signed-rank Ours vs Ours+MST (corrected), ROPE=0.01.

    Recomputes the submitted manuscript's tab:bayes on the corrected MST
    combination. Paired samples per task: mean V-measure over the ten node
    orderings for every (partition, k) cell present in both configurations;
    probabilities from the Bayesian signed-rank test (Benavoli et al., 2017)
    via baycomp, ROPE of 0.01 on the [0,1] V-measure scale.
    """
    import numpy as np
    import baycomp

    def _paired(pattern):
        # one sample per (task, partition, k): mean over the ten orderings; the
        # test below is run separately per k, so its paired samples are
        # independent partitions (no repeated measurements across k)
        agg = collections.defaultdict(list)
        for f in glob.glob(pattern):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("skipped") or "v_measure" not in d:
                    continue
                agg[(d["task"], d.get("partition_idx"), d["n_neighbors"])].append(d["v_measure"])
        return {key: statistics.mean(v) for key, v in agg.items()}

    ours = _paired(f"{RES}/r23_ours_md_l12_arnes/shard*/results.*.jsonl")
    oursm = _paired(f"{RES}/r23_mdmst_l12_corrected_arnes/shard*/results.*.jsonl")
    KSLICES = [1, 6]

    def probs(t, kk):
        keys = sorted(key for key in ours if key[0] == t and key[2] == kk and key in oursm)
        x = np.array([ours[key] for key in keys])
        y = np.array([oursm[key] for key in keys])
        np.random.seed(0)  # deterministic Monte-Carlo posterior
        return baycomp.two_on_multiple(x, y, rope=0.01), len(keys)

    L = ["\\begin{tabular}{lrrr|rrr}", "\\toprule",
         "{} & \\multicolumn{3}{c|}{$k=1$} & \\multicolumn{3}{c}{$k=6$} \\\\",
         "Dataset & P(Ours) & P(ROPE) & P(+MST) & P(Ours) & P(ROPE) & P(+MST) \\\\ \\midrule"]
    for t in sorted({key[0] for key in ours}):
        cells = []
        for kk in KSLICES:
            (pl, prope, pr), n = probs(t, kk)
            row = [f"{p:.3f}" for p in (pl, prope, pr)]
            row[int(np.argmax([pl, prope, pr]))] = f"\\textbf{{{max(pl, prope, pr):.3f}}}"
            cells += row
            print(f"  [bayes] {t} k={kk}: n={n} P(ours)={pl:.3f} P(rope)={prope:.3f} P(+mst)={pr:.3f}")
        L.append(f"{t} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    _write("tab_bayes_body.tex", L)


TABLES = {"r23": make_r23, "r25": make_r25, "r12b": make_r12b, "r12a": make_r12a, "bayes": make_bayes}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    todo = list(TABLES) if which == ["all"] else which
    for name in todo:
        if name not in TABLES:
            raise SystemExit(f"unknown table {name!r}; choose from {sorted(TABLES)} or 'all'")
        print(f"[{name}]")
        TABLES[name]()
