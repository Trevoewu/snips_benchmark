#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MRC-style SNIPS slot filling data from LODO folds."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/snips_lodo_llama"),
        help="Root directory containing per-fold LODO JSONL files with train/dev/test splits.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path("experiments/snips_mrc_templates.json"),
        help="JSON file containing per-domain slot question templates.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/snips_lodo_mrc"),
        help="Directory where per-fold MRC JSONL files are written.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_templates(path: Path) -> Dict[str, List[Dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Template JSON must contain a top-level object.")
    return payload


def answer_start(utterance: str, text: str) -> int:
    start = utterance.find(text)
    if start >= 0:
        return start

    pattern = re.escape(text).replace(r"\ ", r"\s+")
    match = re.search(pattern, utterance)
    if match is not None:
        return match.start()

    raise ValueError(f"Could not locate span text {text!r} in utterance {utterance!r}")


def span_text_from_tokens(record: Dict[str, object], span: Dict[str, object]) -> str:
    tokens = record.get("tokens")
    start = span.get("start")
    end = span.get("end")
    if (
        not isinstance(tokens, list)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        return str(span["text"])
    if start < 0 or end >= len(tokens) or start > end:
        return str(span["text"])
    return " ".join(str(token) for token in tokens[start : end + 1])


def build_examples(
    fold: str,
    split_name: str,
    records: Iterable[Dict[str, object]],
    templates_by_domain: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for record in records:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            utterance = str(metadata.get("utterance", record.get("utterance", "")))
            spans = metadata.get("spans", record.get("spans", []))
            tokens = metadata.get("tokens", record.get("tokens"))
            bio_tags = metadata.get("bio_tags", record.get("bio_tags"))
            domain = metadata.get("domain", record.get("domain"))
        else:
            utterance = str(record.get("utterance", ""))
            spans = record.get("spans", [])
            tokens = record.get("tokens")
            bio_tags = record.get("bio_tags")
            domain = record.get("domain")
        if not isinstance(spans, list):
            raise ValueError(f"Record {record.get('id')} has invalid spans field")

        span_by_slot: Dict[str, Dict[str, object]] = {}
        for span in spans:
            if not isinstance(span, dict):
                continue
            slot = span.get("slot")
            text = span.get("text")
            if (
                isinstance(slot, str)
                and isinstance(text, str)
                and slot not in span_by_slot
            ):
                span_by_slot[slot] = span

        qa_metadata = {
            "domain": domain,
            "utterance": utterance,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "spans": spans,
            "source_id": record.get("id"),
            "split": split_name,
            "heldout_fold": fold,
        }

        if not isinstance(domain, str) or domain not in templates_by_domain:
            raise ValueError(f"Missing templates for example domain {domain!r}")

        template_domain = domain if split_name in {"train", "dev"} else fold
        if template_domain not in templates_by_domain:
            raise ValueError(
                f"Missing templates for template domain {template_domain!r}"
            )

        templates = templates_by_domain[template_domain]
        for template in templates:
            slot = template["slot"]
            slot_name = template["slot_name"]
            question = template["question"]
            span = span_by_slot.get(slot)
            if span is None:
                answers = {"text": [], "answer_start": []}
                is_impossible = True
            else:
                text = str(span["text"])
                try:
                    start = answer_start(utterance, text)
                except ValueError:
                    text = span_text_from_tokens(record, span)
                    start = answer_start(utterance, text)
                answers = {
                    "text": [text],
                    "answer_start": [start],
                }
                is_impossible = False

            examples.append(
                {
                    "id": f"{record['id']}__{slot}",
                    "example_id": record["id"],
                    "fold": fold,
                    "split": split_name,
                    "domain": domain,
                    "slot": slot,
                    "slot_name": slot_name,
                    "question": question,
                    "context": utterance,
                    "answers": answers,
                    "is_impossible": is_impossible,
                    "metadata": qa_metadata,
                }
            )
    return examples


def main() -> None:
    args = parse_args()
    templates = load_templates(args.template_path)
    split_names = [
        "train",
        "dev",
        "test_all",
        "test_seen_slots",
        "test_unseen_slots",
        "test_no_slots",
    ]

    for fold_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
        fold = fold_dir.name
        if fold not in templates:
            raise ValueError(f"Missing templates for fold {fold}")

        for split_name in split_names:
            input_path = fold_dir / f"{split_name}.jsonl"
            if not input_path.exists():
                continue
            records = read_jsonl(input_path)
            output_records = build_examples(fold, split_name, records, templates)
            write_jsonl(args.output_root / fold / f"{split_name}.jsonl", output_records)


if __name__ == "__main__":
    main()
