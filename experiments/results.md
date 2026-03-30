# Experiment Results

## 2026-03-30

### Qwen3-4B Pilot

Setup
- Model: `/data/public_model/qwen3-4b`
- Method: QLoRA SFT
- Script: `scripts/train_llama_sft.py`
- Seed: `42`
- Train batch size: `4`
- Eval batch size: `16`
- Gradient accumulation: `2`
- Max seq length: `512`
- Max new tokens: `64`
- Selection metric: dev micro-F1

Completed folds

| Fold | Best epoch | Dev micro-F1 | Test all | Seen | Unseen | Exact match |
| --- | --- | --- | --- | --- | --- | --- |
| `AddToPlaylist` | `2` | `0.7127` | `0.7017` | `0.8660` | `0.6385` | `0.2457` |
| `BookRestaurant` | `6` | `0.9759` | `0.7193` | `0.9556` | `0.7169` | `0.2887` |
| `GetWeather` | `5` | `0.9714` | `0.8186` | `0.9496` | `0.7562` | `0.5930` |
| `PlayMusic` | `1` | `0.9695` | `0.7588` | `0.7423` | `0.7624` | `0.4969` |
| `RateBook` | `1` | `0.9524` | `0.4094` | `n/a` | `0.4094` | `0.0688` |
| `SearchCreativeWork` | `4` | `0.9726` | `0.8345` | `0.8345` | `n/a` | `0.7299` |
| `SearchScreeningEvent` | `4` | `0.9689` | `0.4564` | `0.1361` | `0.4694` | `0.2249` |

Aggregate across seven folds
- Mean test micro-F1: `0.6712`
- Std dev across folds: `0.1577`
- Strongest test result: `SearchCreativeWork` at `0.8345` micro-F1.
- Most difficult fold: `RateBook` at `0.4094` micro-F1.
- `RateBook` has no seen-slot subset.
- `SearchCreativeWork` has no unseen-slot subset.

Integrated seen vs unseen across all domains

| View | Folds with subset | Examples | Precision | Recall | Micro-F1 | Exact match |
| --- | --- | --- | --- | --- | --- | --- |
| Seen slots | `7` | `4143` | `0.8593` | `0.8178` | `0.8380` | `0.7190` |
| Unseen slots | `7` | `10074` | `0.7045` | `0.5410` | `0.6120` | `0.2424` |

Completed fold: `AddToPlaylist`
- Best epoch: `2`
- Dev micro-F1: `0.7127`
- Dev exact match: `0.5189`
- Test all micro-F1: `0.7017`
- Test all exact match: `0.2457`
- Seen-slot micro-F1: `0.8660`
- Unseen-slot micro-F1: `0.6385`
- Report path: `outputs/qwen3_4b_sft/AddToPlaylist/seed_42/test_reports/test_all.json`

Per-slot test F1 for `AddToPlaylist`
- `artist`: `0.8710`
- `entity_name`: `0.4641`
- `music_item`: `0.7827`
- `playlist`: `0.8633`
- `playlist_owner`: `0.0000`

Completed fold: `BookRestaurant`
- Best epoch: `6`
- Dev micro-F1: `0.9759`
- Test all micro-F1: `0.7193`
- Seen-slot micro-F1: `0.9556`
- Unseen-slot micro-F1: `0.7169`
- Report path: `outputs/qwen3_4b_sft/BookRestaurant/seed_42/test_reports/test_all.json`

Completed fold: `GetWeather`
- Best epoch: `5`
- Dev micro-F1: `0.9714`
- Test all micro-F1: `0.8186`
- Seen-slot micro-F1: `0.9496`
- Unseen-slot micro-F1: `0.7562`
- Report path: `outputs/qwen3_4b_sft/GetWeather/seed_42/test_reports/test_all.json`

Completed fold: `PlayMusic`
- Best epoch: `1`
- Dev micro-F1: `0.9695`
- Test all micro-F1: `0.7588`
- Seen-slot micro-F1: `0.7423`
- Unseen-slot micro-F1: `0.7624`
- Report path: `outputs/qwen3_4b_sft/PlayMusic/seed_42/test_reports/test_all.json`

Completed fold: `RateBook`
- Best epoch: `1`
- Dev micro-F1: `0.9524`
- Test all micro-F1: `0.4094`
- Unseen-slot micro-F1: `0.4094`
- Report path: `outputs/qwen3_4b_sft/RateBook/seed_42/test_reports/test_all.json`

Completed fold: `SearchCreativeWork`
- Best epoch: `4`
- Dev micro-F1: `0.9726`
- Test all micro-F1: `0.8345`
- Seen-slot micro-F1: `0.8345`
- Report path: `outputs/qwen3_4b_sft/SearchCreativeWork/seed_42/test_reports/test_all.json`

Completed fold: `SearchScreeningEvent`
- Best epoch: `4`
- Dev micro-F1: `0.9689`
- Test all micro-F1: `0.4564`
- Seen-slot micro-F1: `0.1361`
- Unseen-slot micro-F1: `0.4694`
- Report path: `outputs/qwen3_4b_sft/SearchScreeningEvent/seed_42/test_reports/test_all.json`

Hard failure modes observed on `RateBook`
- `object_select`: `0.0000` F1
- `best_rating`: `0.0925` F1
- `rating_unit`: `0.1226` F1

Worst per-slot failures across all seven folds
- `AddToPlaylist.playlist_owner`: `0.0000` F1
- `RateBook.object_select`: `0.0000` F1
- `SearchScreeningEvent.object_type`: `0.0018` F1
- `SearchScreeningEvent.movie_type`: `0.0093` F1
- `RateBook.best_rating`: `0.0925` F1
- `RateBook.rating_unit`: `0.1226` F1
- `PlayMusic.playlist`: `0.2957` F1
- `PlayMusic.track`: `0.2968` F1
- `SearchScreeningEvent.location_name`: `0.3312` F1
- `GetWeather.current_location`: `0.3425` F1

Notes
- Earlier pilot evaluations were sensitive to generation formatting and padding behavior.
- `AddToPlaylist` test reports were re-run with the patched eval path and now replace the older saved outputs.
