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

## Prompt-conditioned generative setting

This is the main LLM setting.

Training examples:
- input contains the example domain, allowed slot names, and utterance
- target is a sparse JSON object containing only present slots

Dev examples:
- come only from the six source domains
- use the same source-domain prompt format and source-domain gold targets as training examples

At test time:
- the model receives the held-out-domain utterance
- the allowed slot names for that held-out domain are provided in the prompt
- the model must output the held-out-domain slot values as exact copied spans

Interpretation:
- the model is allowed to condition on held-out-domain slot names at inference time
- the model is not allowed to see held-out-domain training utterances

## MRC baseline setting

The MRC baseline should follow the same zero-shot transfer logic.

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
