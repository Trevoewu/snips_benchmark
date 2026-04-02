---
license: other
base_model: qwen3-4b
tags:
- snips
- slot-filling
- information-extraction
- peft
- lora
- qwen
- zero-shot
language:
- en
library_name: peft
pipeline_tag: text-generation
---

# Qwen3-4B SNIPS LODO Slot Filling Adapters

This repository contains LoRA adapters for leave-one-domain-out SNIPS slot filling, trained on top of `qwen3-4b`.

Each subfolder corresponds to one held-out SNIPS domain:
- `AddToPlaylist`
- `BookRestaurant`
- `GetWeather`
- `PlayMusic`
- `RateBook`
- `SearchCreativeWork`
- `SearchScreeningEvent`

These adapters were trained for instruction-following slot extraction with a sparse JSON output format:

```json
{"slot_name": "slot value", "...": "..."}
```

Only present slots are returned.

## Task

Given:
- the domain
- the allowed slot names
- the user utterance

the model extracts slot values as exact text spans copied from the utterance.

## Training Setup

- Base model: `qwen3-4b`
- Method: QLoRA-style supervised fine-tuning
- Seed: `42`
- Max sequence length: `512`
- Max new tokens: `64`
- Train batch size: `4`
- Eval batch size: `16`
- Gradient accumulation: `2`
- Learning rate: `2e-4`
- Scheduler: `cosine`
- Early stopping: dev micro-F1

## Evaluation

Metric:
- exact span-plus-slot micro-F1

Reported views:
- `test_all`
- `test_seen_slots`
- `test_unseen_slots`

Integrated results from the 7-fold Qwen run:
- Seen-slot micro-F1: `0.8380`
- Unseen-slot micro-F1: `0.6120`

## Fold Results

| Fold | Dev micro-F1 | Test all | Seen | Unseen |
| --- | ---: | ---: | ---: | ---: |
| AddToPlaylist | 0.7127 | 0.7017 | 0.8660 | 0.6385 |
| BookRestaurant | 0.9759 | 0.7193 | 0.9556 | 0.7169 |
| GetWeather | 0.9714 | 0.8186 | 0.9496 | 0.7562 |
| PlayMusic | 0.9695 | 0.7588 | 0.7423 | 0.7624 |
| RateBook | 0.9524 | 0.4094 | n/a | 0.4094 |
| SearchCreativeWork | 0.9726 | 0.8345 | 0.8345 | n/a |
| SearchScreeningEvent | 0.9689 | 0.4564 | 0.1361 | 0.4694 |

Mean test micro-F1 across all 7 folds:
- `0.6712`

## Repository Layout

Each fold subfolder contains:
- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files
- `metrics.json`
- fold-specific `README.md`

## Usage

Load one fold adapter on top of the base model.

Example: `AddToPlaylist`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen3-4B"
repo_id = "Trevoewu/qwen3-4b-snips-lodo"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="AddToPlaylist")
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, repo_id, subfolder="AddToPlaylist")
```

## Notes

- These are adapter weights, not full base-model checkpoints.
- You must load them on top of the compatible Qwen 3 4B base model.
- `RateBook` has no seen-slot held-out subset.
- `SearchCreativeWork` has no unseen-slot held-out subset.
