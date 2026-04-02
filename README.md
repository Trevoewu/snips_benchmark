# snips_benchmark

See `AGENT.md`, `experiments/todo.md`, and `research/lab_notebook.md` for the experiment context.

## Repo layout

- Raw SNIPS source data lives in `data/snips/`
- Regenerated artifacts are written to:
  - `data/snips_lodo/`
  - `data/snips_lodo_llama/`
  - `data/snips_lodo_mrc/`
  - `data/snips_lodo_t5/`
  - `data/snips_lodo_tokencls/`
  - `data/eval_reports/`

Those derived directories are intentionally ignored by git. Regenerate them after cloning.

## Dataset preparation

From the repo root, run:

```bash
python3 scripts/prepare_datasets.py --dedupe
```

This runs, in order:

1. `scripts/build_snips_lodo.py`
2. `scripts/build_llama_slot_data.py`
3. `scripts/build_mrc_slot_data.py`
4. `scripts/build_baseline_data.py`
5. `scripts/evaluate_slot_json.py` on a gold-vs-gold sanity check

## Generated artifacts

- Fold metadata: `data/snips_lodo/summary.json`
- Llama SFT data: `data/snips_lodo_llama/summary.json`
- MRC baseline data: `data/snips_lodo_mrc/`
- Token classification baseline data: `data/snips_lodo_tokencls/summary.json`
- T5 baseline data: `data/snips_lodo_t5/summary.json`
- Evaluator sanity check: `data/eval_reports/addtoplaylist_gold_eval.json`

## Qwen Release

- Hugging Face repo: `l0ulan/qwen3-4b-snips-lodo`
- Contents: one Qwen 3 4B LoRA adapter per held-out SNIPS fold under a fold subfolder
- Base model expected at load time: `Qwen/Qwen3-4B`

## Current SFT Setup

- Default config: `experiments/llama_sft_config.json`
- Training entrypoint: `scripts/train_llama_sft.py`
- Current completed seven-fold run: `/data/public_model/qwen3-4b`
- Other local instruct models available include:
  - `/data/public_model/Mistral-7B-Instruct-v0.3`
  - `/data/public_model/Meta-Llama-3.1-8B-Instruct`

Example pilot run:

```bash
python3 scripts/train_llama_sft.py \
  --fold AddToPlaylist \
  --model-path /data/public_model/qwen3-4b \
  --data-root data/snips_lodo_llama \
  --output-root outputs/qwen3_4b_sft \
  --seed 42 \
  --epochs 6 \
  --patience-epochs 1 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --scheduler cosine \
  --train-batch-size 4 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --max-seq-length 512 \
  --max-new-tokens 64 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --auto-4bit \
  --run-test-after-training
```

Batch launchers:

- `scripts/run_llama31_all_folds.sh` runs the same SFT settings across all seven folds and supports multi-GPU scheduling via `GPU_LIST`.

## MRC Baseline

- Template file: `experiments/snips_mrc_templates.json`
- MRC data builder: `scripts/build_mrc_slot_data.py`
- Training entrypoint: `scripts/train_mrc_slot_model.py`
- Current model path used for the pilot: `/data/public_model/microsoft-deberta-v3-large`

Example MRC pilot run:

```bash
python3 scripts/train_mrc_slot_model.py \
  --fold AddToPlaylist \
  --model-path /data/public_model/microsoft-deberta-v3-large \
  --data-root data/snips_lodo_mrc \
  --gold-root data/snips_lodo_llama \
  --output-root outputs/deberta_v3_mrc \
  --seed 42 \
  --epochs 6 \
  --patience-epochs 1 \
  --learning-rate 3e-5 \
  --warmup-ratio 0.06 \
  --scheduler linear \
  --train-batch-size 16 \
  --eval-batch-size 32 \
  --max-seq-length 384 \
  --max-answer-length 12 \
  --doc-stride 96 \
  --bf16 \
  --run-test-after-training
```

## Notes

- `bitsandbytes` is required for true 4-bit QLoRA; otherwise `--auto-4bit` falls back to regular LoRA.
- Model selection is based on dev JSON extraction micro-F1, not dev loss.
- Full experiment results live in `experiments/results.md`.
