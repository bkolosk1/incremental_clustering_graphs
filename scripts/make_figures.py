"""Regenerate Figures 2 and 3 (adapt_models_orig.pdf / adapt_models.pdf) with corrected labels.

Data: the original per-(strategy, model, k, dataset) V-measures from
cluster_results/inter_data.csv (strategy 'base' = the incremental construction),
plus the frozen standard-kNN L12 numbers of the main results table (identical to
scripts/make_tables.py KNN_FROZEN). Fixes over the submitted figures: legend
'k-NNN' -> 'k-NN', legend swatch colors match the curves, y-axis 'HCV-measure'
-> 'V-measure'. Layout, colors, and values are unchanged.
"""
import csv
import collections
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = __file__.rsplit("/scripts/", 1)[0] + "/data/inter_data.csv"
OUT = __file__.rsplit("/scripts/", 1)[0] + "/figures"
KS = [1, 2, 3, 6, 8, 10, 15, 20]
os.makedirs(__file__.rsplit("/scripts/", 1)[0] + "/figures", exist_ok=True)

COLS_FIG2 = ["ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S",
             "RedditClustering", "StackExchangeClustering", "TwentyNewsgroupsClustering",
             "ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P",
             "StackExchangeClusteringP2P", "RedditClusteringP2P"]
COLS_FIG3 = ["TwentyNewsgroupsClustering", "ArxivClusteringP2P", "RedditClustering",
             "BiorxivClusteringS2S", "MedrxivClusteringP2P", "BiorxivClusteringP2P",
             "ArxivClusteringS2S", "StackExchangeClustering", "MedrxivClusteringS2S",
             "StackExchangeClusteringP2P", "RedditClusteringP2P"]
MODELS = ["bge-base-en-v1.5", "all-MiniLM-L12-v2", "gte-large", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
PAL = {"bge-base-en-v1.5": "tab:blue", "all-MiniLM-L12-v2": "tab:orange", "gte-large": "tab:green",
       "all-MiniLM-L6-v2": "tab:red", "all-mpnet-base-v2": "tab:purple"}

# frozen standard-kNN L12 V1 (x100), main-results table, column order = COLS of make_tables.py
TAB_COLS = ["ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S", "RedditClustering",
            "StackExchangeClustering", "TwentyNewsgroupsClustering", "ArxivClusteringP2P",
            "BiorxivClusteringP2P", "MedrxivClusteringP2P", "RedditClusteringP2P", "StackExchangeClusteringP2P"]
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
OURS_FROZEN = {
    1: [25.85, 18.85, 25.93, 33.02, 32.48, 32.89, 34.09, 22.74, 28.02, 44.28, 35.18],
    2: [36.27, 26.23, 29.15, 48.22, 55.04, 40.79, 44.39, 30.41, 31.75, 60.37, 35.74],
    3: [37.71, 28.23, 30.62, 50.82, 58.03, 43.20, 45.48, 32.27, 33.06, 61.16, 36.28],
    6: [38.74, 29.70, 31.89, 53.70, 59.75, 45.72, 46.32, 33.69, 34.22, 61.36, 36.79],
    8: [38.92, 30.22, 32.28, 54.34, 59.94, 46.25, 46.40, 34.14, 34.58, 61.35, 36.90],
    10: [38.99, 30.74, 32.43, 54.64, 60.05, 46.49, 46.55, 34.34, 34.65, 60.91, 36.96],
    15: [39.01, 31.22, 32.72, 54.83, 59.93, 46.82, 46.52, 34.77, 34.81, 60.20, 36.98],
    20: [38.95, 31.52, 32.75, 54.96, 59.60, 46.99, 46.55, 35.01, 34.67, 59.55, 36.93],
}
knn = {(t, k): KNN_FROZEN[k][i] for k in KS for i, t in enumerate(TAB_COLS)}
ours = {(t, k): OURS_FROZEN[k][i] for k in KS for i, t in enumerate(TAB_COLS)}

base = {}   # (model, task, k) -> v (fraction)
for r in csv.DictReader(open(CSV)):
    if r["strategy"] == "base":
        base[(r["model"], r["dataset"], int(r["k"]))] = float(r["v-measure"])

def grid(names):
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 13.5))
    plt.rcParams.update({"font.size": 10})
    return fig, axes.ravel()

# ---------------- Figure 2: k-NN vs Ours (L12), percentages ----------------
fig, axs = grid(COLS_FIG2)
for i, t in enumerate(COLS_FIG2):
    ax = axs[i]
    ax.plot(range(len(KS)), [knn[(t, k)] for k in KS], marker=".", color="tab:blue", label="k-NN")
    ax.plot(range(len(KS)), [ours[(t, k)] for k in KS], marker=".", color="tab:orange", label="Ours")
    ax.set_title(t, fontsize=11)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
    ax.grid(alpha=0.3)
    if i % 3 == 0:
        ax.set_ylabel("V-measure")
    if i >= 8:
        ax.set_xlabel("k")
axs[11].axis("off")
h, l = axs[0].get_legend_handles_labels()
axs[11].legend(h, l, loc="center", fontsize=13, title="graph construction method", title_fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/adapt_models_orig.pdf")
print("wrote adapt_models_orig.pdf (Figure 2)")

# ---------------- Figure 3: 5 models, base strategy, fractions ----------------
fig, axs = grid(COLS_FIG3)
for i, t in enumerate(COLS_FIG3):
    ax = axs[i]
    for m in MODELS:
        ax.plot(range(len(KS)), [base[(m, t, k)] for k in KS], marker=".", color=PAL[m], label=m, linewidth=1.4)
    ax.set_title(t, fontsize=11)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
    ax.grid(alpha=0.3)
    if i % 3 == 0:
        ax.set_ylabel("V-measure")
    if i >= 8:
        ax.set_xlabel("k")
axs[11].axis("off")
h, l = axs[0].get_legend_handles_labels()
axs[11].legend(h, l, loc="center", fontsize=11, title="embedding model", title_fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/adapt_models.pdf")
print("wrote adapt_models.pdf (Figure 3)")

# sanity: Ours(L12) at k=1 ArxivS2S should match Table 4 (25.85)
print("check Ours L12 ArxivS2S k=1:", round(100 * base[("all-MiniLM-L12-v2", "ArxivClusteringS2S", 1)], 2))
