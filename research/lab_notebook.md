# Lab Notebook

## 2026-04-02

### What changed
- Added `experiments/snips_mrc_templates.json` for domain-specific slot questions.
- Added `scripts/build_mrc_slot_data.py` to expand each fold into extractive QA examples under `data/snips_lodo_mrc/`.
- Added `scripts/train_mrc_slot_model.py` for fold-wise `microsoft-deberta-v3-large` MRC training with dev-threshold selection and utterance-level span scoring.

### Why
- A standard BIO tagger is not a fair primary baseline for truly unseen slot labels.
- An MRC-style baseline is closer to the pre-LLM zero-shot setting because it conditions on slot-specific questions at inference time.

### Outcome
- The MRC pipeline runs end-to-end on `AddToPlaylist` and writes comparable `test_all`, `test_seen_slots`, and `test_unseen_slots` reports.
- Current completed MRC pilot result on `AddToPlaylist`:
  - dev micro-F1 `0.1318`
  - test micro-F1 `0.5694`
  - seen micro-F1 `0.6829`
  - unseen micro-F1 `0.5294`
- The strongest current weakness on that fold is slot coverage for `entity_name` and `playlist_owner`, both at `0.0000` F1.

### Follow-up
- Finish the remaining six MRC folds and compare them directly against the Qwen release.
- If MRC remains weak on key unseen slots, inspect question wording and no-answer thresholding on dev only.

### Update
- The initial MRC sweep stopped too aggressively for several folds under `epochs=6` and `patience_epochs=1`.
- The default MRC configuration was updated to `epochs=8` and `patience_epochs=3` to better tolerate noisy thresholded dev micro-F1.

## 2026-03-30

### What changed
- Ran two `qwen3-4b` QLoRA pilot folds with `scripts/train_llama_sft.py` using the temporary local base model at `/data/public_model/qwen3-4b`:
  - `AddToPlaylist`
  - `BookRestaurant`
- Patched `scripts/evaluate_slot_json.py` so it can recover the first valid JSON object from model outputs that contain leading prose or `<think>` blocks before the final JSON answer.
- Patched `scripts/train_llama_sft.py` so generation prompts pre-close Qwen3's think block via `enable_thinking=False`, keeping inference prompts aligned with the assistant-side chat template used during SFT.

### Why
- The raw pilot outputs often contained correct slot JSON after extra reasoning text, but the original evaluator required the whole prediction string to be valid JSON and therefore reported near-zero dev/test F1.
- Qwen3's chat template inserts a closed think block before assistant answers during supervised formatting, so generation prompts should match that layout to reduce stray reasoning text and avoid training/inference mismatch.

### Outcome
- The original saved `test_reports/` from the pilot runs were misleadingly low because most predictions were counted as `invalid_json`.
- Re-scoring the saved predictions with the patched evaluator produced the following recovered held-out `test_all` results:
  - `AddToPlaylist`: micro-F1 `0.6743`, exact match `0.2329`
  - `BookRestaurant`: micro-F1 `0.7146`, exact match `0.3066`
- Subset breakdown from the reparsed reports:
  - `AddToPlaylist` seen slots: micro-F1 `0.8931`
  - `AddToPlaylist` unseen slots: micro-F1 `0.5926`
  - `BookRestaurant` seen slots: micro-F1 `0.8950`
  - `BookRestaurant` unseen slots: micro-F1 `0.7136`
- Saved repaired evaluation outputs under:
  - `outputs/qwen3_4b_sft/AddToPlaylist/seed_42/reparsed_test_reports/`
  - `outputs/qwen3_4b_sft/BookRestaurant/seed_42/reparsed_test_reports/`

### Follow-up
- Re-run the Qwen3 pilot folds with the patched trainer so dev selection and held-out decoding both reflect the corrected prompt format.
- If Qwen3 still emits occasional reasoning text, keep the tolerant evaluator path for robustness but treat the trainer-side prompt alignment as the primary fix.

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

### Pilot default update
- Switched the current default SFT pilot from Llama to `/data/public_model/qwen3-4b` because the Llama 3.1 checkpoint is still downloading locally.
- Kept the same data pipeline, JSON evaluation, and QLoRA-style settings so the cheaper Qwen run can serve as a smoke test before the larger Llama experiment.
