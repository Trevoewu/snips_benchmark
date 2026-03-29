#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leave-one-domain-out SNIPS test subsets from full domain files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/snips"),
        help="Directory containing per-domain SNIPS files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/snips_lodo"),
        help="Directory where fold artifacts will be written.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact duplicate utterance+tag examples within each domain before writing outputs.",
    )
    return parser.parse_args()


def domain_names(input_dir: Path) -> List[str]:
    names = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_dir() or path.name == "original_snips_data":
            continue
        domain_file = path / f"{path.name}.txt"
        if domain_file.exists():
            names.append(path.name)
    return names


def parse_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    start = None
    slot_name = None

    for idx, tag in enumerate(tags):
        if tag == "O":
            if start is not None and slot_name is not None:
                spans.append(make_span(tokens, start, idx - 1, slot_name))
                start = None
                slot_name = None
            continue

        prefix, current_slot = tag.split("-", 1)

        if prefix == "B":
            if start is not None and slot_name is not None:
                spans.append(make_span(tokens, start, idx - 1, slot_name))
            start = idx
            slot_name = current_slot
            continue

        if prefix == "I" and start is not None and slot_name == current_slot:
            continue

        if start is not None and slot_name is not None:
            spans.append(make_span(tokens, start, idx - 1, slot_name))
        start = idx
        slot_name = current_slot

    if start is not None and slot_name is not None:
        spans.append(make_span(tokens, start, len(tags) - 1, slot_name))

    return spans


def make_span(tokens: Sequence[str], start: int, end: int, slot_name: str) -> Dict[str, object]:
    return {
        "slot": slot_name,
        "start": start,
        "end": end,
        "text": " ".join(tokens[start : end + 1]),
    }


def load_examples(input_dir: Path, dedupe: bool) -> Dict[str, List[Dict[str, object]]]:
    all_examples: Dict[str, List[Dict[str, object]]] = {}
    for domain in domain_names(input_dir):
        examples: List[Dict[str, object]] = []
        seen_raw = set()
        path = input_dir / domain / f"{domain}.txt"
        with path.open("r", encoding="utf-8") as handle:
            for line_idx, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Malformed line {line_idx} in {path}: {line!r}")

                utterance, tag_string = parts
                key = (utterance, tag_string)
                if dedupe and key in seen_raw:
                    continue
                seen_raw.add(key)

                tokens = utterance.split()
                tags = tag_string.split()
                if len(tokens) != len(tags):
                    raise ValueError(
                        f"Token/tag length mismatch at {path}:{line_idx}: "
                        f"{len(tokens)} tokens vs {len(tags)} tags"
                    )

                spans = parse_spans(tokens, tags)
                slot_types = sorted({span["slot"] for span in spans})
                examples.append(
                    {
                        "id": f"{domain}-{line_idx}",
                        "domain": domain,
                        "utterance": utterance,
                        "tokens": tokens,
                        "bio_tags": tags,
                        "spans": spans,
                        "present_slot_types": slot_types,
                        "source_line": line_idx,
                    }
                )
        all_examples[domain] = examples
    return all_examples


def collect_slot_types(examples: Iterable[Dict[str, object]]) -> List[str]:
    slot_types = set()
    for example in examples:
        slot_types.update(example["present_slot_types"])
    return sorted(slot_types)


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_slot_counts(records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for span in record["spans"]:
            counter[span["slot"]] += 1
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    examples_by_domain = load_examples(args.input_dir, dedupe=args.dedupe)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "dedupe": args.dedupe,
        "domains": [],
    }

    for heldout_domain in sorted(examples_by_domain):
        train_domains = [d for d in sorted(examples_by_domain) if d != heldout_domain]
        train_examples = [ex for d in train_domains for ex in examples_by_domain[d]]
        heldout_examples = list(examples_by_domain[heldout_domain])

        train_slot_types = collect_slot_types(train_examples)
        heldout_slot_types = collect_slot_types(heldout_examples)
        train_slot_set = set(train_slot_types)
        heldout_slot_set = set(heldout_slot_types)

        seen_slot_types = sorted(heldout_slot_set & train_slot_set)
        unseen_slot_types = sorted(heldout_slot_set - train_slot_set)
        seen_slot_set = set(seen_slot_types)
        unseen_slot_set = set(unseen_slot_types)

        test_all = []
        test_seen = []
        test_unseen = []
        test_no_slots = []

        for example in heldout_examples:
            record = dict(example)
            record["candidate_slot_names"] = heldout_slot_types
            record["seen_slot_names"] = seen_slot_types
            record["unseen_slot_names"] = unseen_slot_types

            present_slots = set(record["present_slot_types"])
            if not present_slots:
                subset = "no_slots"
                test_no_slots.append(record)
            elif present_slots & unseen_slot_set:
                subset = "unseen_slots"
                test_unseen.append(record)
            elif present_slots <= seen_slot_set:
                subset = "seen_slots"
                test_seen.append(record)
            else:
                raise ValueError(
                    f"Could not assign example {record['id']} in {heldout_domain}."
                )

            record["test_subset"] = subset
            test_all.append(record)

        fold_dir = args.output_dir / heldout_domain
        write_jsonl(fold_dir / "test_all.jsonl", test_all)
        write_jsonl(fold_dir / "test_seen_slots.jsonl", test_seen)
        write_jsonl(fold_dir / "test_unseen_slots.jsonl", test_unseen)
        write_jsonl(fold_dir / "test_no_slots.jsonl", test_no_slots)

        metadata = {
            "heldout_domain": heldout_domain,
            "train_domains": train_domains,
            "dedupe": args.dedupe,
            "candidate_slot_names": heldout_slot_types,
            "seen_slot_names": seen_slot_types,
            "unseen_slot_names": unseen_slot_types,
            "counts": {
                "test_all": len(test_all),
                "test_seen_slots": len(test_seen),
                "test_unseen_slots": len(test_unseen),
                "test_no_slots": len(test_no_slots),
            },
            "slot_span_counts": {
                "test_all": summarize_slot_counts(test_all),
                "test_seen_slots": summarize_slot_counts(test_seen),
                "test_unseen_slots": summarize_slot_counts(test_unseen),
                "test_no_slots": summarize_slot_counts(test_no_slots),
            },
        }
        (fold_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["domains"].append(metadata)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
