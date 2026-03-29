#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


Span = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate JSON slot extraction predictions against gold spans."
    )
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSONL file.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction JSONL file with one record per example id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report JSON.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}: {exc}") from exc
    return records


def get_candidate_slots(record: Dict[str, object]) -> Optional[Set[str]]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        slot_names = metadata.get("candidate_slot_names")
        if isinstance(slot_names, list):
            return {str(name) for name in slot_names}
    return None


def parse_target_text(text: str, candidate_slots: Optional[Set[str]]) -> Dict[str, object]:
    issues = {
        "invalid_json": False,
        "invalid_schema": False,
        "invalid_slot_name_count": 0,
        "invalid_text_count": 0,
        "duplicate_prediction_count": 0,
    }
    valid_spans: List[Span] = []
    seen: Set[Span] = set()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        issues["invalid_json"] = True
        return {"spans": [], "issues": issues}

    if not isinstance(payload, dict) or not isinstance(payload.get("slots"), list):
        issues["invalid_schema"] = True
        return {"spans": [], "issues": issues}

    for item in payload["slots"]:
        if not isinstance(item, dict):
            issues["invalid_schema"] = True
            continue
        slot = item.get("slot")
        text_value = item.get("text")
        if not isinstance(slot, str) or not slot:
            issues["invalid_schema"] = True
            continue
        if candidate_slots is not None and slot not in candidate_slots:
            issues["invalid_slot_name_count"] += 1
            continue
        if not isinstance(text_value, str):
            issues["invalid_text_count"] += 1
            continue
        span = (slot, text_value)
        if span in seen:
            issues["duplicate_prediction_count"] += 1
            continue
        seen.add(span)
        valid_spans.append(span)

    return {"spans": valid_spans, "issues": issues}


def extract_prediction_text(record: Dict[str, object]) -> str:
    for key in ["prediction", "response", "output", "generated_text", "text", "target"]:
        value = record.get(key)
        if isinstance(value, str):
            return value

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content

    assistant = record.get("assistant")
    if isinstance(assistant, str):
        return assistant

    raise ValueError(
        f"Could not locate prediction text for record id={record.get('id')!r}."
    )


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    args = parse_args()
    gold_records = read_jsonl(args.gold)
    prediction_records = read_jsonl(args.predictions)

    gold_by_id = {str(record["id"]): record for record in gold_records}
    pred_by_id = {str(record["id"]): record for record in prediction_records}

    missing_ids = sorted(set(gold_by_id) - set(pred_by_id))
    extra_ids = sorted(set(pred_by_id) - set(gold_by_id))

    tp = fp = fn = 0
    exact_match_count = 0
    invalid_json_count = 0
    invalid_schema_count = 0
    invalid_slot_name_count = 0
    invalid_text_count = 0
    duplicate_prediction_count = 0
    per_slot_counts: Dict[str, Counter[str]] = {}
    per_subset_counts: Dict[str, Counter[str]] = {}

    for example_id, gold_record in gold_by_id.items():
        candidate_slots = get_candidate_slots(gold_record)
        gold_text = gold_record.get("target")
        if not isinstance(gold_text, str):
            raise ValueError(f"Gold record {example_id} is missing a string target field.")

        gold_parsed = parse_target_text(gold_text, candidate_slots)
        gold_spans = set(gold_parsed["spans"])

        pred_record = pred_by_id.get(example_id)
        if pred_record is None:
            pred_spans: Set[Span] = set()
            pred_issues = {
                "invalid_json": False,
                "invalid_schema": False,
                "invalid_slot_name_count": 0,
                "invalid_text_count": 0,
                "duplicate_prediction_count": 0,
            }
        else:
            pred_text = extract_prediction_text(pred_record)
            pred_parsed = parse_target_text(pred_text, candidate_slots)
            pred_spans = set(pred_parsed["spans"])
            pred_issues = pred_parsed["issues"]

        invalid_json_count += int(pred_issues["invalid_json"])
        invalid_schema_count += int(pred_issues["invalid_schema"])
        invalid_slot_name_count += int(pred_issues["invalid_slot_name_count"])
        invalid_text_count += int(pred_issues["invalid_text_count"])
        duplicate_prediction_count += int(pred_issues["duplicate_prediction_count"])

        example_tp = len(gold_spans & pred_spans)
        example_fp = len(pred_spans - gold_spans)
        example_fn = len(gold_spans - pred_spans)
        tp += example_tp
        fp += example_fp
        fn += example_fn
        exact_match_count += int(gold_spans == pred_spans)

        subset = "unknown"
        metadata = gold_record.get("metadata")
        if isinstance(metadata, dict):
            subset = str(metadata.get("test_subset") or gold_record.get("split") or "unknown")

        subset_counter = per_subset_counts.setdefault(subset, Counter())
        subset_counter.update(tp=example_tp, fp=example_fp, fn=example_fn, examples=1)
        subset_counter.update(exact_match=int(gold_spans == pred_spans))

        for slot, _ in gold_spans & pred_spans:
            per_slot_counts.setdefault(slot, Counter()).update(tp=1)
        for slot, _ in pred_spans - gold_spans:
            per_slot_counts.setdefault(slot, Counter()).update(fp=1)
        for slot, _ in gold_spans - pred_spans:
            per_slot_counts.setdefault(slot, Counter()).update(fn=1)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    report = {
        "gold_path": str(args.gold),
        "predictions_path": str(args.predictions),
        "counts": {
            "examples": len(gold_by_id),
            "predictions": len(pred_by_id),
            "missing_predictions": len(missing_ids),
            "extra_predictions": len(extra_ids),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_match": exact_match_count,
            "invalid_json": invalid_json_count,
            "invalid_schema": invalid_schema_count,
            "invalid_slot_name": invalid_slot_name_count,
            "invalid_text": invalid_text_count,
            "duplicate_prediction": duplicate_prediction_count,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "micro_f1": f1_score(tp, fp, fn),
            "exact_match": safe_div(exact_match_count, len(gold_by_id)),
        },
        "per_subset": {},
        "per_slot": {},
        "missing_prediction_ids": missing_ids,
        "extra_prediction_ids": extra_ids,
    }

    for subset, counter in sorted(per_subset_counts.items()):
        subset_tp = int(counter["tp"])
        subset_fp = int(counter["fp"])
        subset_fn = int(counter["fn"])
        subset_examples = int(counter["examples"])
        report["per_subset"][subset] = {
            "examples": subset_examples,
            "precision": safe_div(subset_tp, subset_tp + subset_fp),
            "recall": safe_div(subset_tp, subset_tp + subset_fn),
            "micro_f1": f1_score(subset_tp, subset_fp, subset_fn),
            "exact_match": safe_div(int(counter["exact_match"]), subset_examples),
        }

    for slot, counter in sorted(per_slot_counts.items()):
        slot_tp = int(counter["tp"])
        slot_fp = int(counter["fp"])
        slot_fn = int(counter["fn"])
        report["per_slot"][slot] = {
            "precision": safe_div(slot_tp, slot_tp + slot_fp),
            "recall": safe_div(slot_tp, slot_tp + slot_fn),
            "f1": f1_score(slot_tp, slot_fp, slot_fn),
            "support": slot_tp + slot_fn,
        }

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
