# Reproducing `contrastive-decoding-on-HELMET`

This fork adds a training-free contrastive-decoding intervention (CAD + two novel variants) to HELMET. This file documents how to reproduce both the **vanilla HELMET baseline** (Phase A) and the **full CD matrix** (Phases B–C).

## Upstream pin

- Upstream: `princeton-nlp/HELMET` at SHA `af609c4` (latest at fork time).
- This project lives on the `cd` branch of `erik-d123/HELMET`.
- Do `git log --oneline cd..main` to see any uncommitted deltas from upstream.

## Environment

Tested on Princeton Neuronic with 1× A100-80GB. Python 3.11.

```bash
git clone -b cd https://github.com/erik-d123/HELMET.git helmet-cd
cd helmet-cd

python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Flash-attn: see https://github.com/Dao-AILab/flash-attention
pip install flash-attn --no-build-isolation
```

## Data

```bash
bash scripts/download_data.sh    # ~34 GB download, extracts to ./data/
```

## Phase A — reproduce vanilla Llama-3.1-8B-Instruct RAG baseline

**Goal:** confirm we can reproduce HELMET's published Llama-3.1-8B-Instruct RAG accuracy within ~2 pts before touching anything. If this gate fails, do not proceed to Phase B.

```bash
bash scripts/reproduce_phase_a.sh
```

This runs RAG (4 KILT datasets: NQ / TriviaQA / HotpotQA / PopQA) at **32K** input length on **n=50** samples per dataset, with the HF path (no vLLM), greedy deterministic.

Results land in `output/Llama-3.1-8B-Instruct/*.json.score`.

**Expected:** substring-EM within ~2 pts of HELMET's published Llama-3.1-8B-Instruct RAG number (see HELMET's public [results spreadsheet](https://docs.google.com/spreadsheets/d/1LBt6dP4UwZwU_CjoYhyAd_rjKhQLvo0Gq4cYUnpi_CA/edit?usp=sharing)).

## Phase B — CAD implementation + α sweep + CAD vs vanilla @ 32K

### Wrapper no-op regression (sanity check)

Run the baseline with `--cd_mode off` through the wrapper path to confirm the wrapper introduces no regression on the same examples:
```bash
python eval.py --config configs/rag.yaml \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --tag phaseB_nop --input_max_length 32768 --max_test_samples 50 \
    --cd_mode off --use_chat_template False
```
Expected: same metric scores (within float noise) as Phase A.

### α sweep (pick α for main runs)

```bash
sbatch scripts/run_cd_alpha_sweep.sh    # CAD on RAG @ 32K, n=100, α ∈ {0.5, 1.0, 1.5}
```

### Main Phase B gate: CAD vs vanilla on RAG @ 32K, full test set

```bash
ALPHA=1.0 sbatch scripts/run_cd_phaseB.sh
```

**Gate:** bootstrap 95% CI for (CAD − vanilla) at 32K does not cross zero (direction irrelevant — even a significantly-negative effect is publishable as a null-result study).

### Trace-log smoke test

```bash
python eval.py --config configs/cd_rag.yaml \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --tag phaseB_smoke --input_max_length 32768 --max_test_samples 10 \
    --cd_mode cad --cd_alpha 1.0 --cd_log_trace --use_chat_template False
```
Then spot-check `output/Llama-3.1-8B-Instruct/cd_trace.jsonl`:
- `logits_A_top ≠ logits_B_top` on most steps (non-trivial contrast)
- At least one example differs from the vanilla output

### Variant differentiation

```bash
python eval.py --config configs/cd_rag.yaml --max_test_samples 10 \
    --cd_mode shuffled --cd_log_trace --tag phaseB_shuf_smoke --input_max_length 32768 \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --use_chat_template False
```
`logits_B_top` from shuffled mode should differ from CAD mode on matched steps.

## Phases C–D

Added as those phases start — see the project plan for the expected file layout.

## COS484 Local-Window Extension

The `codex/cos484-local-window-cd` branch adds the full class-project harness for:

- vanilla **Recall** + **RAG** reproduction at `8K`, `32K`, and `64K`
- RAG-only decoding/control methods:
  - `vanilla`
  - `cad`
  - `local_window` with `--cd_window_tokens {2000,8000}`
  - `truncate_last` with `--context_window_tokens {2000,8000}`
  - `oracle_passages_only`
  - `shuffled` / `reversed` at `32K`

### New configs

- `configs/cos484_recall_{8k,32k,64k}.yaml`
- `configs/cos484_rag_{8k,32k,64k}.yaml`

These pin the exact HELMET Recall/RAG datasets for the three required lengths.

### New CLI options

- `--cd_mode local_window`
- `--cd_window_tokens N`
- `--context_mode {off,truncate_last,oracle_passages_only}`
- `--context_window_tokens N`
- `--analysis_window_tokens 2000,8000`

`local_window` uses the **visible Pass-A context after normal HELMET truncation**, then keeps only the last `N` context tokens for Pass B.

### New scripts

- `bash scripts/cos484_smoke.sh`
  - 32K RAG smoke run for vanilla / CAD / local-window / truncation / oracle
- `bash scripts/cos484_position_diag.sh`
  - vanilla 32K RAG pilot + diagnostic summary
- `bash scripts/cos484_alpha_sweep.sh`
  - per-method alpha sweep for `cad`, `local_window_2k`, `local_window_8k`
- `bash scripts/cos484_recall_vanilla.sh`
  - vanilla Recall curve at 8K / 32K / 64K
- `bash scripts/cos484_rag_matrix.sh`
  - full RAG matrix across lengths and methods
- `python scripts/cos484_analyze.py --input_dir output/Llama-3.1-8B-Instruct --output_dir analysis/cos484`
  - macro-average summaries with bootstrap 95% CI
  - delta-vs-vanilla summaries
  - 32K evidence-position diagnostic
  - plots when `matplotlib` is available

### Per-example metadata

RAG outputs now log:

- `dataset`
- `method`
- `length`
- `primary_metric_name`
- `primary_score`
- `first_answer_passage_rank`
- `answer_position_bucket`
- `gold_inside_window_2000`
- `gold_inside_window_8000`

This is the metadata consumed by `scripts/cos484_analyze.py`.

### Hardware note

- `64K` CD runs should use an `A100-80GB` / `gpu80` node when possible.
- If `gpu80` is unavailable, keep the three-point vanilla Recall/RAG reproduction curve and drop `64K` CD methods first.
