# cross-domain zeroshot learning for the slot-filling task (ZSSF) Protocol

## Goal

We use a leave-one-domain-out (LODO) evaluation protocol, similar to Bapna et al. (2017), in which the model is trained on six source domains and evaluated zero-shot on one held-out target domain.

## Dataset

We use the SNIPS dataset (Coucke et al., 2018), a widely used benchmark for intent detection and slot filling consisting of seven domains.

- SNIPS: contain 7 domains: 
  - `AddToPlaylist` contains 5 slots: `music_item`, `playlist_owner`, `entity_name`, `playlist`, `artist`
  - `BookRestaurant` contains 14 slots: `city`, `facility`, `timeRange`, `restaurant_name`, `country`, `cuisine`, `restaurant_type`, `served_dish`, `party_size_number`, `poi`, `sort`, `spatial_relation`, `state`, `party_size_description`
  - `GetWeather` contains 9 slots: `city`, `state`, `timeRange`, `current_location`, `country`, `spatial_relation`, `geographic_poi`, `condition_temperature`, `condition_description`
  - `PlayMusic` contains 9 slots: `genre`, `music_item`, `service`, `year`, `playlist`, `album`, `sort`, `track`, `artist`
  - `RateBook` contains 7 slots: `object_part_of_series_type`, `object_select`, `rating_value`, `object_name`, `object_type`, `rating_unit`, `best_rating`
  - `SearchCreativeWork` contains 2 slots: `object_name`, `object_type`
  - `SearchScreeningEvent` contains 7 slots: `timeRange`, `movie_type`, `object_location_type`, `object_type`, `location_name`, `spatial_relation`, `movie_name`

For each held-out fold:
- choose one SNIPS domain as the target domain
- train on the other six source domains
- do not use held-out-domain training utterances
- select checkpoints on a source-domain dev split drawn from the other six domains(optional)
- evaluate once on the held-out test set

### Seen and unseen slot

- For a held-out target domain, let `T` be the set of slot labels that appear in that target domain, and let `S` be the set of slot labels observed in the other six source domains.
- **Seen slots** are target-domain slot labels that also appear in the source-domain label inventory: `T ∩ S`.
- **Unseen slots** are target-domain slot labels that do not appear in any source domain: `T \ S`.
- This definition depends only on slot-label overlap. It does not depend on slot values, utterance wording, or semantic similarity.

Held-out test utterances are partitioned as follows:
- `test_seen_slots`: utterances whose annotated slots are non-empty and all belong to the seen-slot set.
- `test_unseen_slots`: utterances that contain at least one unseen-slot label.
- In SNIPS, every evaluation utterance has at least one slot annotation, so `test_all` is the union of `test_seen_slots` and `test_unseen_slots`.

**Unseen rate**: The unseen rate is the proportion of target-domain slot labels that are unseen under the leave-one-domain-out split, i.e. `|T \ S| / |T|`.

For SNIPS under this leave-one-domain-out setup:

| Held-out target domain | Source labels (raw sum over 6 domains) | Target labels (T) | Seen target labels (T ∩ S) | Unseen target labels (T \ S) | Unseen rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `AddToPlaylist` | 48 | 5 | 3 | 2 | 0.4000 |
| `BookRestaurant` | 39 | 14 | 6 | 8 | 0.5714 |
| `GetWeather` | 44 | 9 | 5 | 4 | 0.4444 |
| `PlayMusic` | 44 | 9 | 4 | 5 | 0.5556 |
| `RateBook` | 46 | 7 | 2 | 5 | 0.7143 |
| `SearchCreativeWork` | 51 | 2 | 2 | 0 | 0.0000 |
| `SearchScreeningEvent` | 46 | 7 | 3 | 4 | 0.5714 |

The source-label count in this table is the raw sum of per-domain label inventories across the six source domains. The unseen-rate calculation above still uses the distinct-label source set `S`.

## Evaluation views

For each held-out fold, report:
- `test_all`
- `test_seen_slots`
- `test_unseen_slots`

Primary metric:
- exact span-plus-slot micro-F1

Also report:
- precision
- recall
- exact match
- per-slot F1

```python
def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```


## Prompt-conditioned generative setting

This is the main LLM setting.

Prompt templates:

System prompt template:

```text
You extract slot values from user utterances. Use only the provided slot names. Return strict JSON as a single object from slot name to extracted text. Copy slot text spans exactly from the utterance. Omit missing slots. Return {} when no slots are present. Example format: {"slot_name": "exact text"}. Do not output markdown or extra text.
```

User prompt template:

```text
Domain: {domain}
Allowed slot names: {slot_1}, {slot_2}, ..., {slot_n}
Utterance: {utterance}
```

Interpretation:
- the model is allowed to condition on held-out-domain slot names at inference time
- the model is not allowed to see held-out-domain training utterances

## MRC baseline setting

The MRC baseline should follow the same zero-shot transfer logic.

The QA-style slot questions are taken from the questioning strategy used in Du et al. (2021), *QA-Driven Zero-shot Slot Filling with Weak Supervision Pretraining*.

Prompt templates:


input template:

```text
Question: {slot_question}
Context: {utterance}
```

Question templates are domain-specific and follow the paper's QA formulation. 

`AddToPlaylist`

| Slot | Question |
| --- | --- |
| `music_item` | `what’s the music item?` |
| `playlist_owner` | `who’s the owner?` |
| `entity_name` | `what’s the entity name?` |
| `playlist` | `what’s the playlist?` |
| `artist` | `who’s the artist?` |

`BookRestaurant`

| Slot | Question |
| --- | --- |
| `city` | `what’s the city?` |
| `facility` | `what’s the facility?` |
| `timeRange` | `when’s the time range?` |
| `restaurant_name` | `what’s the name?` |
| `country` | `what’s the country?` |
| `cuisine` | `what’s the cuisine?` |
| `restaurant_type` | `what’s the restaurant type?` |
| `served_dish` | `what’s the served dish?` |
| `party_size_number` | `how many people?` |
| `poi` | `where’s the location?` |
| `sort` | `what’s the type?` |
| `spatial_relation` | `what’s the spatial relation?` |
| `state` | `what’s the state?` |
| `party_size_description` | `who are the persons?` |

`GetWeather`

| Slot | Question |
| --- | --- |
| `city` | `what’s the city?` |
| `state` | `what’s the state?` |
| `timeRange` | `when’s the time range?` |
| `current_location` | `what’s the current location?` |
| `country` | `what’s the country?` |
| `spatial_relation` | `what’s the spatial relation?` |
| `geographic_poi` | `where’s the location?` |
| `condition_temperature` | `how is the temperature?` |
| `condition_description` | `how is the weather?` |

`PlayMusic`

| Slot | Question |
| --- | --- |
| `genre` | `what’s the genre?` |
| `music_item` | `what’s the music item?` |
| `service` | `what’s the service?` |
| `year` | `when’s the year?` |
| `playlist` | `what’s the playlist?` |
| `album` | `what’s the album?` |
| `sort` | `what’s the type?` |
| `track` | `what’s the track?` |
| `artist` | `who’s the artist?` |

`RateBook`

| Slot | Question |
| --- | --- |
| `object_part_of_series_type` | `what’s the series?` |
| `object_select` | `which to select?` |
| `rating_value` | `how many rating value?` |
| `object_name` | `what’s the object name?` |
| `object_type` | `what’s the object type?` |
| `rating_unit` | `what’s the rating unit?` |
| `best_rating` | `how many rating points in total?` |

`SearchCreativeWork`

| Slot | Question |
| --- | --- |
| `object_name` | `what’s the object name?` |
| `object_type` | `what’s the object type?` |

`SearchScreeningEvent`

| Slot | Question |
| --- | --- |
| `timeRange` | `when’s the time range?` |
| `movie_type` | `what’s the movie type?` |
| `object_location_type` | `what’s the location type?` |
| `object_type` | `what’s the object type?` |
| `location_name` | `where’s the location name?` |
| `spatial_relation` | `what’s the spatial relation?` |
| `movie_name` | `what’s the movie name?` |

### Train and dev

For source-domain utterances:
- ask only source-domain slot questions
- do not ask irrelevant held-out-domain questions on source-domain utterances

Examples:
- a `GetWeather` utterance should be expanded only with `GetWeather` slot questions
- a `PlayMusic` utterance should be expanded only with `PlayMusic` slot questions

This keeps train and dev supervision aligned with the utterance semantics.

Dev definition:
- the dev set is a validation split from the six source domains
- dev utterances must be paired with source-domain questions and source-domain answers
- held-out-domain questions must not be used on source-domain dev utterances

### Test

For held-out-domain test utterances:
- ask the held-out-domain slot questions
- aggregate slot-wise QA answers back into utterance-level slot predictions

Interpretation:
- the model learns QA-style slot extraction from source domains
- zero-shot transfer is tested by applying held-out-domain slot questions only at evaluation time

## What is not allowed

- no held-out-domain training utterances
- no tuning on held-out test predictions
- no post-processing rules designed from held-out test errors

## What is allowed

- use held-out-domain slot names or slot questions at inference time
- use a source-domain dev split for checkpoint selection when the dev supervision is aligned with the training formulation
- compare generative and non-generative baselines under the same held-out test views

## Rationale

This protocol measures cross-domain slot transfer, not in-domain supervised learning.

The key requirement is:
- source-domain utterances must be paired with source-domain supervision during training
- held-out-domain schema information may be used at inference time for zero-shot transfer

This avoids training on question-utterance pairs that are semantically unrelated, which would otherwise distort both learning and dev selection.

## references

- Bapna et al. (2017)
- Coucke et al. (2018)
- Du et al. (2021), *QA-Driven Zero-shot Slot Filling with Weak Supervision Pretraining*: https://aclanthology.org/2021.acl-short.83/
