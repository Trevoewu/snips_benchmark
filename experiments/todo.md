# Experiments Plan

## Goal

Fine-tune instruction-following LLMs such as Llama 3 8B for cross-domain zero-shot slot filling on SNIPS using a leave-one-domain-out protocol. The model sees only slot names at inference time and must extract exact slot spans as JSON.

## Fixed Setup

- Dataset source: `data/snips/`
- Regenerated LODO folds: `data/snips_lodo/`
- LLM-ready instruction data: `data/snips_lodo_llama/`
- Fold construction: hold out one SNIPS domain, train on the other six domains, test on the held-out domain
- Prompt style: domain name plus allowed slot names only, no slot descriptions
- Output format: `{"slot_name_1": "slot_value" | null, "slot_name_2": "slot_value" | null, ...}`
- Deduplication: enabled during fold and prompt-data generation

## Main Hypothesis

An instruction-tuned LLM fine-tuned on the six source domains can transfer slot extraction behavior to a held-out SNIPS domain when given only the held-out domain name and slot names, but performance will vary sharply with the proportion of unseen slot types in that domain.

## Evaluation

- Primary metric: exact span-plus-slot micro-F1 on held-out test data
- Secondary metrics: precision, recall, per-slot F1, and utterance-level exact match
- Report all three held-out views for each fold:
  - `test_all.jsonl`
  - `test_seen_slots.jsonl`
  - `test_unseen_slots.jsonl`
- Aggregate across all seven folds with mean and standard deviation over seeds
- Run at least 3 random seeds for each main model configuration

## Known Edge Cases

- `RateBook` has no seen-slot held-out subset
- `SearchCreativeWork` has no unseen-slot held-out subset
- These folds should still be included in overall reporting, with empty subset metrics marked as not applicable

## Experiment Sequence

### Phase 1: Evaluation Pipeline

1. Build a scorer for model JSON outputs against gold spans in `data/snips_lodo_llama/`
2. Validate the scorer on gold targets to confirm perfect recovery
3. Add strict parsing rules for malformed JSON, duplicate spans, and invalid slot names

### Phase 2: Baselines

1. Majority or empty-output baseline to set a lower bound
2. Prompt-only zero-shot Llama baseline without fine-tuning
3. `DeBERTa-v3` token classifier as the main discriminative baseline
4. `BERT` token classifier if `DeBERTa-v3` is too heavy or unstable
5. `BiLSTM-CRF` as the classic sequence-labeling baseline
6. `T5-base` as the smaller generative baseline for comparison with Llama-style instruction tuning

### Phase 3: Main Fine-Tuning Run

1. Fine-tune `Llama3 8B` with LoRA or QLoRA on each LODO fold
2. Train on `train.jsonl`, select checkpoints on `dev.jsonl`
3. Evaluate on `test_all`, `test_seen_slots`, and `test_unseen_slots`
4. Save raw predictions for later error analysis

### Baseline Notes

1. `DeBERTa-v3` or `BERT`
   - Train on the same six source domains for each fold
   - Evaluate with the same held-out `test_all`, `test_seen_slots`, and `test_unseen_slots` subsets
   - Use token-level BIO tagging and convert predictions to spans before scoring
   - Treat this as the strongest non-generative reference point
2. `BiLSTM-CRF`
   - Use GloVe or fastText initialization if available, otherwise learned embeddings
   - Keep it as a lightweight classical baseline to show how much transfer comes from modern pretrained encoders vs sequence modeling alone
3. `T5-base`
   - Use the same JSON extraction target format as the Llama experiments
   - Keep prompts close to the Llama setup so the comparison isolates model scale and architecture more than prompt design

### Phase 4: Ablations

1. Slot names only vs slot names plus domain name removed
2. Smaller vs larger max output length
3. Training with and without examples that contain only source-domain-unique slots
4. Optional comparison between full fine-tuning surrogate model and LoRA

### Phase 5: Analysis

1. Compare folds by unseen-slot ratio and slot inventory size
2. Break down failures into wrong slot label, wrong span boundary, missing span, and hallucinated span
3. Inspect whether performance drops are concentrated on domain-unique slots or also affect shared slots

## Run Order

1. Implement and verify the evaluator
2. Run cheap lower-bound baselines: empty-output and prompt-only zero-shot Llama
3. Run `DeBERTa-v3` on all folds as the main strong baseline
4. Run `BiLSTM-CRF` on all folds if compute is limited, or in parallel with `DeBERTa-v3`
5. Run `T5-base` on all folds as the smaller generative baseline
6. Run one-seed QLoRA pilot on two folds:
   - `AddToPlaylist` as a mixed seen/unseen case
   - `BookRestaurant` or `RateBook` as a hard unseen-heavy case
7. If the pilot is stable, launch all seven folds
8. Repeat best configuration for additional seeds

## Immediate SFT Pilot Configuration

- Current pilot model while Llama finishes downloading: `/data/public_model/qwen3-4b`
- Planned main model after download: `/data/public_model/Meta-Llama-3.1-8B-Instruct`
- Primary baseline model path for later comparison: `/data/public_model/roberta-base`
- Training data: `data/snips_lodo_llama/<heldout_domain>/train.jsonl`
- Dev data: `data/snips_lodo_llama/<heldout_domain>/dev.jsonl`
- Test data: `data/snips_lodo_llama/<heldout_domain>/test_all.jsonl`, `test_seen_slots.jsonl`, `test_unseen_slots.jsonl`
- Fine-tuning method: start with QLoRA SFT
- Random seed: `42`
- Validation selection metric: dev exact span-plus-slot micro-F1 from generated JSON outputs
- Early stopping rule: stop when one full epoch finishes without improving dev micro-F1
- Checkpoint rule: keep the checkpoint with the best dev micro-F1; if tied, prefer lower dev loss
- Decoding for evaluation: deterministic decoding, no sampling
- Pilot folds:
  - `AddToPlaylist`
  - `BookRestaurant` or `RateBook`

## Initial SFT Hyperparameters

- Epoch cap: `5-8`
- Learning rate: `2e-4`
- Warmup ratio: `0.03`
- Scheduler: cosine or linear decay
- Max input length: `512`
- Max generation length: `128-192`
- LoRA rank: `16` or `32`
- LoRA alpha: `32` or `64`
- LoRA dropout: `0.05`
- Precision: prefer `bf16`

## Success Criteria

- The model beats the prompt-only baseline on held-out `test_all`
- The main Llama setup is competitive with or better than `DeBERTa-v3` on at least some unseen-slot-heavy folds
- The model shows non-trivial F1 on `test_unseen_slots`
- Results are stable across seeds and reproducible from saved fold artifacts

## Logging Requirements

- Save hyperparameters, seed, training loss, dev metrics, and held-out test metrics for every run
- Save per-fold prediction files and aggregate results table
- Update `research/lab_notebook.md` after each experiment batch
