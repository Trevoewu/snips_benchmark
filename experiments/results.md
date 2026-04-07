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

Per-slot F1 by fold

`AddToPlaylist`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `artist` | `0.8710` | `784` |
| `entity_name` | `0.4641` | `608` |
| `music_item` | `0.7827` | `940` |
| `playlist` | `0.8633` | `2035` |
| `playlist_owner` | `0.0000` | `1161` |

`BookRestaurant`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `city` | `0.8105` | `546` |
| `country` | `0.8846` | `376` |
| `cuisine` | `0.4589` | `221` |
| `facility` | `0.3797` | `166` |
| `party_size_description` | `0.1277` | `329` |
| `party_size_number` | `0.8566` | `1079` |
| `poi` | `0.0373` | `149` |
| `restaurant_name` | `0.4679` | `359` |
| `restaurant_type` | `0.5387` | `1402` |
| `served_dish` | `0.6316` | `274` |
| `sort` | `0.8579` | `212` |
| `spatial_relation` | `0.8775` | `335` |
| `state` | `0.8983` | `544` |
| `timeRange` | `0.8927` | `708` |

`GetWeather`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `city` | `0.8372` | `889` |
| `condition_description` | `0.7367` | `475` |
| `condition_temperature` | `0.6464` | `496` |
| `country` | `0.9467` | `522` |
| `current_location` | `0.3425` | `280` |
| `geographic_poi` | `0.4085` | `306` |
| `spatial_relation` | `0.8703` | `220` |
| `state` | `0.9324` | `515` |
| `timeRange` | `0.9463` | `1106` |

`PlayMusic`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `album` | `0.3681` | `189` |
| `artist` | `0.9539` | `1232` |
| `genre` | `0.4093` | `146` |
| `music_item` | `0.6345` | `811` |
| `playlist` | `0.2957` | `158` |
| `service` | `0.9553` | `780` |
| `sort` | `0.6611` | `363` |
| `track` | `0.2968` | `217` |
| `year` | `0.8973` | `648` |

`RateBook`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `best_rating` | `0.0925` | `1072` |
| `object_name` | `0.6238` | `1024` |
| `object_part_of_series_type` | `0.4944` | `319` |
| `object_select` | `0.0000` | `972` |
| `object_type` | `0.6653` | `932` |
| `rating_unit` | `0.1226` | `1152` |
| `rating_value` | `0.4593` | `1991` |

`SearchCreativeWork`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `object_name` | `0.8256` | `2051` |
| `object_type` | `0.8474` | `1535` |

`SearchScreeningEvent`

| Slot | F1 | Support |
| --- | ---: | ---: |
| `location_name` | `0.3312` | `610` |
| `movie_name` | `0.7821` | `857` |
| `movie_type` | `0.0093` | `686` |
| `object_location_type` | `0.4520` | `477` |
| `object_type` | `0.0018` | `706` |
| `spatial_relation` | `0.6798` | `690` |
| `timeRange` | `0.7090` | `277` |


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

### DeBERTa-v3 MRC Baseline

Setup
- Model: `/data/public_model/microsoft-deberta-v3-large`
- Method: extractive MRC slot filling
- Script: `scripts/train_mrc_slot_model.py`
- Seed: `42`

Completed folds

| Fold | Best epoch | Dev micro-F1 | Test all | Seen | Unseen | Exact match |
| --- | --- | --- | --- | --- | --- | --- |
| `AddToPlaylist` | `3` | `0.1318` | `0.5694` | `0.6829` | `0.5294` | `0.1022` |

Notes
- Current completed MRC evidence is only for `AddToPlaylist`.
- Compared with Qwen on the same fold, DeBERTa MRC is lower on `test_all` (`0.5694` vs `0.7017`).
- Weakest `AddToPlaylist` slots in the MRC run: `entity_name` `0.0000`, `playlist_owner` `0.0000`.
