# Running clustering_graphs on VEGA — two-phase pipeline

**Design:** the sweep is CPU-bound (spectral_embedding + KMeans + graph prep); the GPU is only ever needed for sentence encoding. So we split:

| phase | job | hardware | what |
| --- | --- | --- | --- |
| 1 | `encode_vega.sbatch` | **1 CPU node** (16 cores, OMP=16), ~1–2 h | encode all (dataset, partition) into `cache/embs`, exit |
| 2a | `r23_nnmst_vega.sbatch` | **16 CPU shards**, ~1–3 h | R2.3 NN+MST sweep, `--require-cache` |
| 2b | `r23_orderings_vega.sbatch` | 2 CPU shards, ~1 h | R1.2-b centroid/class orderings, `--require-cache` |

**Everything is CPU.** Vega's GPU nodes are heterogeneous — several lack the `nvidia-container-cli` binary, and the site apptainer config force-enables an nvccli mode that `--nvccli=false` does not reliably override (FATAL observed on jobs 39820844 and 39823728). Since the only GPU use was the ~30 s/partition encode of a 22M-param model, we do that on CPU too (~1–2 h once) and the whole pipeline becomes node-agnostic. Do **not** re-add `--nv` unless you pin `--nodelist` to a GPU node you've confirmed has `nvidia-container-cli`.

Benefits: no GPU queue contention for the long part, 16-way parallelism on the plentiful `cpu` partition, and phase 2 **never touches the NVIDIA container stack** — the `nvidia-container-cli: executable not found` failure class (job 39820844 shard 0) cannot occur.

`--require-cache` makes phase 2 fail immediately and loudly if the cache is cold, instead of silently encoding on CPU for minutes per partition.

## Deploy (from local repo root)

```bash
bash scripts/deploy_vega.sh          # code + SIF (first time)
bash scripts/deploy_vega.sh --code   # code only (after edits)
```

## Submit everything (on Vega, from ~/clustering_graphs)

```bash
bash scripts/submit_all_vega.sh
```

That submits phase 1, then phase 2a/2b with `--dependency=afterok:<encode-job>` — SLURM holds the experiments until the cache is warm, and never starts them if encoding failed.

Variants:

```bash
ENCODE_ARRAY=0-4 bash scripts/submit_all_vega.sh   # warm all 5 backbone models
bash scripts/submit_all_vega.sh --skip-encode      # cache already warm → experiments only
```

## Monitor

```bash
squeue -u $USER          # PD (Dependency) on phase 2 until encode finishes = expected
tail -f logs/encode_*_0.out
tail -f logs/r23_*_0.out

# records per shard
for d in results/r23_nnmst_minilm_vega/shard*; do echo "$d: $(cat $d/results.*.jsonl 2>/dev/null | wc -l)"; done
```

## Merge + pull back

```bash
# on Vega
cat results/r23_nnmst_minilm_vega/shard*/results.*.jsonl > results/r23_nnmst_minilm_vega/all_results.jsonl

# from local
rsync -avz bkoloski@login.vega.izum.si:~/clustering_graphs/results/ /home/boshkokoloski/work/clustering_graphs/results_vega/
```

## Caveats

- **Same `--datasets` / `--models` / `--limit` in both phases.** The cache key fingerprints the actual sentences, so a phase-2 run with a different `--limit` than the encode run = guaranteed `CacheMissError`.
- **OMP=4 in phase 2, OMP=16 in phase 1.** The AMG/LOBPCG spectral solver oversubscribes past 4 threads (OMP=16 measured ~5× slower), but the transformer encode in phase 1 *does* benefit from many threads — hence the split.
- **Vega memory ratio: 2 GB/core.** The submit plugin rejects `--mem` that exceeds 2000 MB × cores ("memory too high, memory per core: …"). Use `--mem-per-cpu=2000M` and buy RAM by adding cores. Peak RSS ~24 GiB/shard on ArxivP2P (dense MST distance matrix) → 24 cores.
- **nvccli is force-disabled** (`APPTAINER_NVCCLI=false` + `--nvccli=false` on the GPU job). Vega's compute nodes are heterogeneous; library-binding `--nv` works on all of them.
- **Old interleaved job:** if a pre-refactor array (e.g. 39820844) is still running, `scancel` it before submitting the new pipeline — its cache writes are kept and reused, but its result records would partially overlap the 16-shard layout and complicate the merge.
- **Leonardo Booster:** don't use it for this — the sweep needs CPUs, not H100s.
