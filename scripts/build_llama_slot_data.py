#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from build_snips_lodo import collect_slot_types, load_examples


SYSTEM_PROMPT = (
    "You extract slot values from user utterances. "
    "Use only the provided slot names. "
    "Return strict JSON as a single object from slot name to extracted text. "
    "Copy slot text spans exactly from the utterance. "
    "Omit missing slots. Return {} when no slots are present. "
    'Example format: {"slot_name": "exact text"}. '
    "Do not output markdown or extra text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build instruction-tuning data for Llama-style slot extraction."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/snips"),
        help="Directory containing the source SNIPS domain files.",
    )
    parser.add_argument(
        "--lodo-dir",
        type=Path,
        default=Path("data/snips_lodo"),
        help="Directory containing LODO fold metadata and held-out test subsets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/snips_lodo_llama"),
        help="Directory where instruction-formatted data will be written.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Per-source-domain fraction of training examples reserved for dev.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact duplicate utterance+tag examples before conversion.",
    )
    return parser.parse_args()


def stable_bucket(example_id: str) -> float:
    digest = hashlib.md5(example_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_train_dev(
    examples_by_domain: Dict[str, List[Dict[str, object]]],
    train_domains: Sequence[str],
    dev_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_records: List[Dict[str, object]] = []
    dev_records: List[Dict[str, object]] = []

    for domain in train_domains:
        domain_examples = list(examples_by_domain[domain])
        domain_examples.sort(key=lambda example: example["id"])

        target_dev = int(round(len(domain_examples) * dev_ratio))
        if dev_ratio > 0 and len(domain_examples) > 1:
            target_dev = max(1, min(target_dev, len(domain_examples) - 1))
        else:
            target_dev = 0

        ordered = sorted(
            domain_examples,
            key=lambda example: (stable_bucket(str(example["id"])), str(example["id"])),
        )
        dev_ids = {example["id"] for example in ordered[:target_dev]}

        for example in domain_examples:
            if example["id"] in dev_ids:
                dev_records.append(example)
            else:
                train_records.append(example)

    return train_records, dev_records


def render_user_prompt(domain: str, slot_names: Sequence[str], utterance: str) -> str:
    slot_list = ", ".join(slot_names)
    return (
        f"Domain: {domain}\n"
        f"Allowed slot names: {slot_list}\n"
        f"Utterance: {utterance}"
    )


def render_target(spans: Iterable[Dict[str, object]], slot_names: Sequence[str]) -> str:
    span_map: Dict[str, str] = {}
    for span in spans:
        slot = span.get("slot")
        text = span.get("text")
        if not isinstance(slot, str) or not isinstance(text, str):
            continue
        if slot not in span_map:
            span_map[slot] = text

    payload = {slot_name: span_map[slot_name] for slot_name in slot_names if slot_name in span_map}
    return json.dumps(payload, ensure_ascii=False)


def make_record(
    example: Dict[str, object],
    domain: str,
    slot_names: Sequence[str],
    split_name: str,
    include_assistant: bool,
) -> Dict[str, object]:
    target = render_target(example["spans"], slot_names)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_user_prompt(domain, slot_names, str(example["utterance"])),
        },
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": target})

    return {
        "id": example["id"],
        "split": split_name,
        "messages": messages,
        "target": target,
        "metadata": {
            "domain": domain,
            "candidate_slot_names": list(slot_names),
            "utterance": example["utterance"],
            "tokens": example["tokens"],
            "bio_tags": example["bio_tags"],
            "spans": example["spans"],
            "present_slot_types": example["present_slot_types"],
            "source_line": example["source_line"],
            "test_subset": example.get("test_subset"),
        },
    }


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_metadata(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not 0 <= args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be in [0, 1).")

    examples_by_domain = load_examples(args.input_dir, dedupe=args.dedupe)
    slot_names_by_domain = {
        domain: collect_slot_types(examples)
        for domain, examples in examples_by_domain.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_dir": str(args.input_dir),
        "lodo_dir": str(args.lodo_dir),
        "output_dir": str(args.output_dir),
        "dev_ratio": args.dev_ratio,
        "dedupe": args.dedupe,
        "folds": [],
    }

    for heldout_domain in sorted(examples_by_domain):
        fold_lodo_dir = args.lodo_dir / heldout_domain
        fold_meta = load_metadata(fold_lodo_dir / "metadata.json")
        train_domains = list(fold_meta["train_domains"])
        train_examples, dev_examples = split_train_dev(
            examples_by_domain,
            train_domains=train_domains,
            dev_ratio=args.dev_ratio,
        )

        train_records = [
            make_record(
                example,
                domain=str(example["domain"]),
                slot_names=slot_names_by_domain[str(example["domain"])],
                split_name="train",
                include_assistant=True,
            )
            for example in train_examples
        ]
        dev_records = [
            make_record(
                example,
                domain=str(example["domain"]),
                slot_names=slot_names_by_domain[str(example["domain"])],
                split_name="dev",
                include_assistant=True,
            )
            for example in dev_examples
        ]

        fold_output_dir = args.output_dir / heldout_domain
        write_jsonl(fold_output_dir / "train.jsonl", train_records)
        write_jsonl(fold_output_dir / "dev.jsonl", dev_records)

        for split_name in ["test_all", "test_seen_slots", "test_unseen_slots", "test_no_slots"]:
            source_path = fold_lodo_dir / f"{split_name}.jsonl"
            source_records = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            converted = [
                make_record(
                    example,
                    domain=heldout_domain,
                    slot_names=fold_meta["candidate_slot_names"],
                    split_name=split_name,
                    include_assistant=False,
                )
                for example in source_records
            ]
            write_jsonl(fold_output_dir / f"{split_name}.jsonl", converted)

        fold_summary = {
            "heldout_domain": heldout_domain,
            "train_domains": train_domains,
            "candidate_slot_names": fold_meta["candidate_slot_names"],
            "seen_slot_names": fold_meta["seen_slot_names"],
            "unseen_slot_names": fold_meta["unseen_slot_names"],
            "counts": {
                "train": len(train_records),
                "dev": len(dev_records),
                "test_all": fold_meta["counts"]["test_all"],
                "test_seen_slots": fold_meta["counts"]["test_seen_slots"],
                "test_unseen_slots": fold_meta["counts"]["test_unseen_slots"],
                "test_no_slots": fold_meta["counts"]["test_no_slots"],
            },
        }
        (fold_output_dir / "metadata.json").write_text(
            json.dumps(fold_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["folds"].append(fold_summary)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
