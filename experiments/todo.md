# Experiments Plan

## Goal

Evaluate leave-one-domain-out SNIPS slot extraction with instruction-tuned causal LMs.

## Current Setup

- Data source: `data/snips/`
- LODO folds: `data/snips_lodo/`
- Instruction data: `data/snips_lodo_llama/`
- MRC baseline data: `data/snips_lodo_mrc/`
- Held-out protocol: train on 6 domains, test on 1 held-out domain
- Prompt input: domain, allowed slot names, utterance
- Target format: sparse JSON object with only present slots
- Generation: deterministic decoding

## Metrics

- Primary: exact span-plus-slot micro-F1 on held-out test
- Also report: precision, recall, exact match, per-slot F1
- Test views:
  - `test_all`
  - `test_seen_slots`
  - `test_unseen_slots`

## Current Pilot

- Model: `/data/public_model/qwen3-4b`
- Training script: `scripts/train_llama_sft.py`
- Method: QLoRA SFT
- Seed: `42`
- Selection metric: dev micro-F1
- Early stopping: 1 epoch without dev micro-F1 improvement

## Current Hyperparameters

- Epochs: `6`
- Learning rate: `2e-4`
- Warmup ratio: `0.03`
- Scheduler: `cosine`
- Max seq length: `512`
- Max new tokens: `64`
- Train batch size: `4`
- Eval batch size: `16`
- Gradient accumulation: `2`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Precision: `bf16`

## Priority Runs

1. Re-run the `DeBERTa-v3` MRC sweep on the corrected source-domain question protocol.
2. Compare corrected MRC vs Qwen on seen and unseen subsets.
3. Keep the Qwen adapter release as the current main generative reference.

## Notes

- `RateBook` has no seen-slot held-out subset.
- `SearchCreativeWork` has no unseen-slot held-out subset.
- Log every completed run in `experiments/results.md`.
