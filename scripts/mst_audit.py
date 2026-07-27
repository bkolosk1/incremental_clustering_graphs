"""Paired benchmark of `--mst-mode paper-faithful` vs `corrected`.

Joins two results.*.jsonl dumps on (task, partition_idx, method, n_neighbors, seed) and reports per-cell V-measure delta + paired-Wilcoxon p + how often the MD-vs-NN ranking flips between modes.

The question this script answers:
  "If we switch from the published (index-mismatched) MST combination to the mathematically correct (permuted) one, does the paper's headline claim -- MD + MST does not improve over MD -- still hold? And by how much does each cell move?"

Run:
  python scripts/mst_audit.py \
    --paper results/mst_audit_paper-faithful \
    --corrected results/mst_audit_corrected \
    --out results/mst_audit_report
"""
import argparse
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
import scipy.stats


def load_results(dir_path):
    """Return a flat list of dicts (one per run) from every results.*.jsonl in `dir_path`."""
    dir_path = pathlib.Path(dir_path)
    out = []
    for jp in sorted(dir_path.glob("results.*.jsonl")):
        for line in jp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            out.append(rec)
    return out


def cell_key(rec):
    return (rec["task"], rec["partition_idx"], rec["method"], rec["n_neighbors"], rec["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="output-dir from the --mst-mode paper-faithful run")
    ap.add_argument("--corrected", required=True, help="output-dir from the --mst-mode corrected run")
    ap.add_argument("--out", required=True, help="output directory for the report (.json + .tsv)")
    args = ap.parse_args()

    paper = {cell_key(r): r for r in load_results(args.paper) if r.get("mst") is True}
    corr = {cell_key(r): r for r in load_results(args.corrected) if r.get("mst") is True}

    common = sorted(set(paper) & set(corr))
    only_paper = set(paper) - set(corr)
    only_corr = set(corr) - set(paper)
    print(f"loaded {len(paper)} paper-faithful runs, {len(corr)} corrected runs")
    print(f"paired cells: {len(common)} | only-paper: {len(only_paper)} | only-corrected: {len(only_corr)}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-cell delta and graph-structural diagnostics.
    rows = []
    for k in common:
        p = paper[k]
        c = corr[k]
        task, pidx, method, kk, seed = k
        d_v = c["v_measure"] - p["v_measure"]
        d_h = c["homogeneity"] - p["homogeneity"]
        d_c = c["completeness"] - p["completeness"]
        d_nnz = c["graph_stats"]["nnz"] - p["graph_stats"]["nnz"]
        d_homo = (c["graph_stats"].get("homophily") or 0.0) - (p["graph_stats"].get("homophily") or 0.0)
        d_density = (c["graph_stats"].get("density") or 0.0) - (p["graph_stats"].get("density") or 0.0)
        rows.append({
            "task": task, "partition": pidx, "method": method, "k": kk, "seed": seed,
            "v_paper": p["v_measure"], "v_corr": c["v_measure"], "delta_v": d_v,
            "delta_h": d_h, "delta_c": d_c,
            "nnz_paper": p["graph_stats"]["nnz"], "nnz_corr": c["graph_stats"]["nnz"], "delta_nnz": d_nnz,
            "delta_homophily": d_homo, "delta_density": d_density,
        })

    # Overall paired stats.
    deltas = np.array([r["delta_v"] for r in rows])
    overall = {
        "n_pairs": len(rows),
        "mean_delta_v": float(deltas.mean()) if len(deltas) else None,
        "median_delta_v": float(np.median(deltas)) if len(deltas) else None,
        "std_delta_v": float(deltas.std()) if len(deltas) else None,
        "abs_mean_delta_v": float(np.abs(deltas).mean()) if len(deltas) else None,
        "max_abs_delta_v": float(np.abs(deltas).max()) if len(deltas) else None,
        "n_corrected_better": int((deltas > 0).sum()),
        "n_paper_better": int((deltas < 0).sum()),
        "n_tied": int((deltas == 0).sum()),
    }
    if len(deltas) >= 5:
        try:
            stat = scipy.stats.wilcoxon(deltas, alternative="two-sided", zero_method="zsplit")
            overall["wilcoxon_stat"] = float(stat.statistic)
            overall["wilcoxon_p"] = float(stat.pvalue)
        except Exception as e:
            overall["wilcoxon_error"] = repr(e)

    # Per-(dataset, method, k) breakdown.
    by_dmk = defaultdict(list)
    for r in rows:
        by_dmk[(r["task"], r["method"], r["k"])].append(r["delta_v"])
    breakdown = []
    for (task, method, k), ds in sorted(by_dmk.items()):
        arr = np.array(ds)
        row = {
            "task": task, "method": method, "k": k, "n": len(arr),
            "mean_delta_v": float(arr.mean()), "std_delta_v": float(arr.std()),
            "median_delta_v": float(np.median(arr)),
        }
        if len(arr) >= 5:
            try:
                s = scipy.stats.wilcoxon(arr, alternative="two-sided", zero_method="zsplit")
                row["wilcoxon_p"] = float(s.pvalue)
            except Exception:
                row["wilcoxon_p"] = None
        breakdown.append(row)

    # MD-vs-NN ranking sign-flip count: for each (task, partition, k, seed), does V(MD) > V(NN) under paper differ from corrected?
    by_cell = defaultdict(dict)
    for k, r in {**{k: ("paper", paper[k]) for k in common}}.items():
        by_cell[(k[0], k[1], k[3], k[4])]["paper_" + k[2]] = r[1]["v_measure"]
    for k in common:
        by_cell[(k[0], k[1], k[3], k[4])]["corr_" + k[2]] = corr[k]["v_measure"]

    flips = 0
    same = 0
    total = 0
    for cell, vs in by_cell.items():
        if not all(x in vs for x in ("paper_MD", "paper_NN", "corr_MD", "corr_NN")):
            continue
        total += 1
        sign_paper = np.sign(vs["paper_MD"] - vs["paper_NN"])
        sign_corr = np.sign(vs["corr_MD"] - vs["corr_NN"])
        if sign_paper != sign_corr:
            flips += 1
        else:
            same += 1
    ranking = {"n_cells_with_both_methods": total, "ranking_kept": same, "ranking_flipped": flips}

    # Persist.
    (out_dir / "pairs.tsv").write_text("\t".join(["task", "partition", "method", "k", "seed", "v_paper", "v_corr", "delta_v", "delta_h", "delta_c", "nnz_paper", "nnz_corr", "delta_nnz", "delta_homophily", "delta_density"]) + "\n" + "\n".join(["\t".join(str(r[c]) for c in ["task", "partition", "method", "k", "seed", "v_paper", "v_corr", "delta_v", "delta_h", "delta_c", "nnz_paper", "nnz_corr", "delta_nnz", "delta_homophily", "delta_density"]) for r in rows]), encoding="utf-8")
    (out_dir / "breakdown.tsv").write_text("\t".join(["task", "method", "k", "n", "mean_delta_v", "std_delta_v", "median_delta_v", "wilcoxon_p"]) + "\n" + "\n".join(["\t".join(str(r.get(c, "")) for c in ["task", "method", "k", "n", "mean_delta_v", "std_delta_v", "median_delta_v", "wilcoxon_p"]) for r in breakdown]), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "overall": overall,
        "ranking_flip": ranking,
        "only_paper_count": len(only_paper),
        "only_corrected_count": len(only_corr),
        "breakdown_n_rows": len(breakdown),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stdout summary.
    print()
    print("=== OVERALL paired Δ V-measure (corrected − paper-faithful) ===")
    for k, v in overall.items():
        print(f"  {k:<26} {v}")
    print()
    print("=== MD vs NN ranking flip ===")
    for k, v in ranking.items():
        print(f"  {k:<32} {v}")
    print()
    print(f"Wrote: {out_dir}/pairs.tsv, {out_dir}/breakdown.tsv, {out_dir}/report.json")


if __name__ == "__main__":
    main()
