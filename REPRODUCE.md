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

## Phases B–D

Will be added as those phases start. The `cd_wrapper.py` module, `configs/cd_*.yaml`, and `scripts/run_cd_slurm.sh` are not present yet — see the [project plan](../rustling-bouncing-sunbeam.md) for the expected file layout.
