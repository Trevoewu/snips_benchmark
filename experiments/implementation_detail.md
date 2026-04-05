# Implementation Details

## Task and Data

We evaluate leave-one-domain-out (LODO) slot filling on SNIPS. For each held-out domain, the model is trained on the other six domains and evaluated on the held-out domain. Derived data are regenerated from `data/snips/` with `scripts/prepare_datasets.py --dedupe`.

Generated data used in the current experiments:
- Generative SFT data: `data/snips_lodo_llama/`
- MRC baseline data: `data/snips_lodo_mrc/`

The LODO protocol contains two known edge cases:
- `RateBook` has no seen-slot held-out subset.
- `SearchCreativeWork` has no unseen-slot held-out subset.

## Prompting

### Generative SFT prompt

System prompt used for Qwen SFT:

```text
You extract slot values from user utterances. Use only the provided slot names. Return strict JSON as a single object from slot name to extracted text. Copy slot text spans exactly from the utterance. Omit missing slots. Return {} when no slots are present. Example format: {"slot_name": "exact text"}. Do not output markdown or extra text.
```

User prompt template:

```text
Domain: <domain>
Allowed slot names: <comma-separated slot names>
Utterance: <utterance>
```

Target format:

```json
{"slot_name": "slot value", "slot_name_2": "slot value_2"}
```

Only present slots are emitted.

### MRC prompt

The DeBERTa baseline reformulates slot filling as extractive QA. Each utterance is expanded into one question per slot. Questions are domain-specific and stored in `experiments/snips_mrc_templates.json`.

Example (`AddToPlaylist`):

```text
Question: what’s the playlist?
Context: add step to me to the 50 clásicos playlist
Answer: 50 clásicos
```

Missing slots are encoded as no-answer examples.

## Models

### Generative model

- Base model: `/data/public_model/qwen3-4b`
- Training script: `scripts/train_llama_sft.py`
- Public adapter release: `https://huggingface.co/l0ulan/qwen3-4b-snips-lodo`

### MRC baseline

- Base model: `/data/public_model/microsoft-deberta-v3-large`
- Training script: `scripts/train_mrc_slot_model.py`

## Training setup

### Qwen3-4B SFT

- Optimization: AdamW
- Method: LoRA / QLoRA-style SFT with PEFT
- Seed: `42`
- Epoch cap: `6`
- Early stopping patience: `1` epoch without dev improvement
- Learning rate: `2e-4`
- Weight decay: `0.0`
- Warmup ratio: `0.03`
- Scheduler: `cosine`
- Max sequence length: `512`
- Max generation length: `64`
- Train batch size: `4`
- Eval batch size: `16`
- Gradient accumulation: `2`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Precision: `bf16` when supported
- Quantization: `--auto-4bit`

Checkpoint selection:
- Dev predictions are generated after each epoch.
- The best checkpoint is selected by dev micro-F1.
- Ties are broken by lower dev loss.
- Held-out test evaluation is run from the saved best checkpoint.

Decoding:
- Deterministic decoding (`do_sample=False`)
- Left padding is used for decoder-only evaluation batches.

### DeBERTa-v3 MRC baseline

- Optimization: AdamW
- Seed: `42`
- Epoch cap: `8`
- Early stopping patience: `3`
- Learning rate: `3e-5`
- Weight decay: `0.0`
- Warmup ratio: `0.06`
- Scheduler: `linear`
- Train batch size: `16`
- Eval batch size: `32`
- Gradient accumulation: `1`
- Max sequence length: `384`
- Max answer length: `12`
- Document stride: `96`
- Precision: `bf16` when supported

Checkpoint selection:
- Dev QA predictions are aggregated back to utterance-level slot outputs after each epoch.
- A no-answer threshold is selected on the dev set by maximizing dev micro-F1, then dev exact match, then preferring the less aggressive threshold.
- The best checkpoint is selected by dev micro-F1, then dev exact match, then lower dev loss.
- A longer patience window is used for MRC than for Qwen SFT because thresholded dev micro-F1 is substantially noisier.

## Evaluation

All models are evaluated with the same span-based scorer in `scripts/evaluate_slot_json.py`.

Primary metric:
- Exact span-plus-slot micro-F1

Additional metrics:
- Precision
- Recall
- Utterance-level exact match
- Per-slot F1

Reported held-out views:
- `test_all`
- `test_seen_slots`
- `test_unseen_slots`

For the generative setup, tolerant JSON recovery is enabled in evaluation so that a prediction is still usable if it contains leading text before the first valid JSON object.

## Hardware and software

Hardware used in the current runs:
- 2 x NVIDIA L20 GPUs
- 46,068 MiB memory per GPU
- NVIDIA driver `580.105.08`

Key software versions in the training environment:
- `torch 2.10.0+cu128`
- `transformers 4.51.3`
- `peft 0.15.2`
- `bitsandbytes 0.49.2`
- `tokenizers 0.21.4`
- `sentencepiece 0.2.1`

## Current completed results referenced in this repo

- Qwen3-4B SFT: all seven folds completed; summarized in `experiments/results.md`
- DeBERTa-v3 MRC: `AddToPlaylist` completed; remaining folds are still being run or queued
