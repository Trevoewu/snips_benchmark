# Lab Notebook

## 2026-03-29

### What changed
- Added `scripts/build_snips_lodo.py` to regenerate leave-one-domain-out SNIPS test subsets from the full per-domain files in `data/snips/`.
- Generated fold artifacts in `data/snips_lodo/` with deterministic seen-slot and unseen-slot partitions based on slot-name overlap between the six training domains and the held-out domain.

### Why
- The repository-provided `seen_slots.txt` and `unseen_slots.txt` do not align cleanly with the full per-domain files, so they are not reliable as canonical subsets for a reproducible leave-one-domain-out benchmark.
- Regenerating the folds from the full domain files removes ambiguity and makes the evaluation protocol explicit.

### Outcome
- Produced validated fold metadata and test partitions for all seven held-out domains.
- Observed two expected edge cases: `RateBook` has no seen-slot test subset, and `SearchCreativeWork` has no unseen-slot test subset.
- Deduplicated exact repeated utterance-plus-tag examples before writing outputs.

### Follow-up
- Added `scripts/build_llama_slot_data.py` to convert each leave-one-domain-out fold into Llama-style instruction-tuning data with slot-name-only prompts.
- Wrote chat-format fold artifacts to `data/snips_lodo_llama/` with `train.jsonl`, `dev.jsonl`, and held-out `test_*.jsonl` files.
- The prompt format names the domain, lists allowed slot names, and asks for strict JSON span extraction with exact copied text.

### Tooling added
- Added `scripts/evaluate_slot_json.py` to score JSON slot extraction outputs against gold spans with micro-F1, precision, recall, exact match, per-slot metrics, and subset-level metrics.
- Validated the evaluator on gold targets for `data/snips_lodo_llama/AddToPlaylist/test_all.jsonl`, recovering perfect scores as expected.
- Added `scripts/build_baseline_data.py` to export the same LODO folds into `data/snips_lodo_tokencls/` for `DeBERTa-v3` or `BiLSTM-CRF` style token classification and `data/snips_lodo_t5/` for `T5-base` text-to-text training.
- The token-classification export uses a single global BIO label vocabulary saved to `data/snips_lodo_tokencls/label_list.json` so all folds share the same label indexing.

### Llama training scaffold
- Added `scripts/train_llama_sft.py` for fold-wise `Meta-Llama-3.1-8B-Instruct` SFT with LoRA or QLoRA, seed `42`, deterministic decoding, and epoch-level early stopping on dev micro-F1.
- The training loop evaluates generated dev outputs after every epoch, keeps the best checkpoint by dev micro-F1, breaks ties with lower dev loss, and can optionally run held-out test evaluation after training.
- Added a default config snapshot in `experiments/llama_sft_config.json` to document the current Llama pilot settings.
