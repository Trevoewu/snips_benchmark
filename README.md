# snips_benchmark

See `AGENT.md`, `experiments/todo.md`, and `research/lab_notebook.md` for the experiment context.

## Repo layout

- Raw SNIPS source data lives in `data/snips/`
- Regenerated artifacts are written to:
  - `data/snips_lodo/`
  - `data/snips_lodo_llama/`
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
3. `scripts/build_baseline_data.py`
4. `scripts/evaluate_slot_json.py` on a gold-vs-gold sanity check

## Generated artifacts

- Fold metadata: `data/snips_lodo/summary.json`
- Llama SFT data: `data/snips_lodo_llama/summary.json`
- Token classification baseline data: `data/snips_lodo_tokencls/summary.json`
- T5 baseline data: `data/snips_lodo_t5/summary.json`
- Evaluator sanity check: `data/eval_reports/addtoplaylist_gold_eval.json`

## Llama SFT

- Default config: `experiments/llama_sft_config.json`
- Training entrypoint: `scripts/train_llama_sft.py`
- Base model path expected by current setup: `/data/public_model/Meta-Llama-3.1-8B-Instruct`

Example pilot run:

```bash
python3 scripts/train_llama_sft.py \
  --fold AddToPlaylist \
  --model-path /data/public_model/Meta-Llama-3.1-8B-Instruct \
  --data-root data/snips_lodo_llama \
  --output-root outputs/llama_sft \
  --seed 42 \
  --epochs 6 \
  --patience-epochs 1 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --scheduler cosine \
  --train-batch-size 1 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512 \
  --max-new-tokens 160 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --auto-4bit \
  --run-test-after-training
```

## Notes

- `bitsandbytes` is required for true 4-bit QLoRA; otherwise `--auto-4bit` falls back to regular LoRA.
- Model selection is based on dev JSON extraction micro-F1, not dev loss.
