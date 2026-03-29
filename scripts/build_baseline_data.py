#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from build_llama_slot_data import render_target, render_user_prompt, split_train_dev
from build_snips_lodo import collect_slot_types, load_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build baseline-ready LODO exports for token classification and T5-style generation."
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
        "--token-output-dir",
        type=Path,
        default=Path("data/snips_lodo_tokencls"),
        help="Directory for token-classification exports.",
    )
    parser.add_argument(
        "--t5-output-dir",
        type=Path,
        default=Path("data/snips_lodo_t5"),
        help="Directory for T5-style text-to-text exports.",
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


def load_metadata(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def bio_vocabulary(examples_by_domain: Dict[str, List[Dict[str, object]]]) -> List[str]:
    slot_types = sorted({slot for examples in examples_by_domain.values() for slot in collect_slot_types(examples)})
    labels = ["O"]
    for slot in slot_types:
        labels.append(f"B-{slot}")
        labels.append(f"I-{slot}")
    return labels


def token_record(example: Dict[str, object], split_name: str, label_to_id: Dict[str, int]) -> Dict[str, object]:
    return {
        "id": example["id"],
        "split": split_name,
        "domain": example["domain"],
        "tokens": example["tokens"],
        "ner_tags": example["bio_tags"],
        "ner_tag_ids": [label_to_id[tag] for tag in example["bio_tags"]],
        "spans": example["spans"],
        "present_slot_types": example["present_slot_types"],
        "source_line": example["source_line"],
        "test_subset": example.get("test_subset"),
    }


def t5_record(example: Dict[str, object], split_name: str, slot_names: Sequence[str]) -> Dict[str, object]:
    return {
        "id": example["id"],
        "split": split_name,
        "domain": example["domain"],
        "input_text": render_user_prompt(str(example["domain"]), slot_names, str(example["utterance"])),
        "target_text": render_target(example["spans"]),
        "utterance": example["utterance"],
        "candidate_slot_names": list(slot_names),
        "spans": example["spans"],
        "present_slot_types": example["present_slot_types"],
        "source_line": example["source_line"],
        "test_subset": example.get("test_subset"),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be in [0, 1).")

    examples_by_domain = load_examples(args.input_dir, dedupe=args.dedupe)
    slot_names_by_domain = {
        domain: collect_slot_types(examples)
        for domain, examples in examples_by_domain.items()
    }
    label_list = bio_vocabulary(examples_by_domain)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    token_summary = {
        "input_dir": str(args.input_dir),
        "lodo_dir": str(args.lodo_dir),
        "output_dir": str(args.token_output_dir),
        "dev_ratio": args.dev_ratio,
        "dedupe": args.dedupe,
        "label_list": label_list,
        "folds": [],
    }
    t5_summary = {
        "input_dir": str(args.input_dir),
        "lodo_dir": str(args.lodo_dir),
        "output_dir": str(args.t5_output_dir),
        "dev_ratio": args.dev_ratio,
        "dedupe": args.dedupe,
        "folds": [],
    }

    args.token_output_dir.mkdir(parents=True, exist_ok=True)
    args.t5_output_dir.mkdir(parents=True, exist_ok=True)
    (args.token_output_dir / "label_list.json").write_text(
        json.dumps(label_list, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for heldout_domain in sorted(examples_by_domain):
        fold_meta = load_metadata(args.lodo_dir / heldout_domain / "metadata.json")
        train_domains = list(fold_meta["train_domains"])
        train_examples, dev_examples = split_train_dev(
            examples_by_domain,
            train_domains=train_domains,
            dev_ratio=args.dev_ratio,
        )

        token_fold_dir = args.token_output_dir / heldout_domain
        t5_fold_dir = args.t5_output_dir / heldout_domain

        token_train = [token_record(example, "train", label_to_id) for example in train_examples]
        token_dev = [token_record(example, "dev", label_to_id) for example in dev_examples]
        t5_train = [
            t5_record(example, "train", slot_names_by_domain[str(example["domain"])])
            for example in train_examples
        ]
        t5_dev = [
            t5_record(example, "dev", slot_names_by_domain[str(example["domain"])])
            for example in dev_examples
        ]

        write_jsonl(token_fold_dir / "train.jsonl", token_train)
        write_jsonl(token_fold_dir / "dev.jsonl", token_dev)
        write_jsonl(t5_fold_dir / "train.jsonl", t5_train)
        write_jsonl(t5_fold_dir / "dev.jsonl", t5_dev)

        for split_name in ["test_all", "test_seen_slots", "test_unseen_slots", "test_no_slots"]:
            source_records = read_jsonl(args.lodo_dir / heldout_domain / f"{split_name}.jsonl")
            token_test = [token_record(example, split_name, label_to_id) for example in source_records]
            t5_test = [t5_record(example, split_name, fold_meta["candidate_slot_names"]) for example in source_records]
            write_jsonl(token_fold_dir / f"{split_name}.jsonl", token_test)
            write_jsonl(t5_fold_dir / f"{split_name}.jsonl", t5_test)

        token_meta = {
            "heldout_domain": heldout_domain,
            "train_domains": train_domains,
            "candidate_slot_names": fold_meta["candidate_slot_names"],
            "seen_slot_names": fold_meta["seen_slot_names"],
            "unseen_slot_names": fold_meta["unseen_slot_names"],
            "label_list_path": str(args.token_output_dir / "label_list.json"),
            "counts": {
                "train": len(token_train),
                "dev": len(token_dev),
                **fold_meta["counts"],
            },
        }
        t5_meta = {
            "heldout_domain": heldout_domain,
            "train_domains": train_domains,
            "candidate_slot_names": fold_meta["candidate_slot_names"],
            "seen_slot_names": fold_meta["seen_slot_names"],
            "unseen_slot_names": fold_meta["unseen_slot_names"],
            "counts": {
                "train": len(t5_train),
                "dev": len(t5_dev),
                **fold_meta["counts"],
            },
        }
        (token_fold_dir / "metadata.json").write_text(
            json.dumps(token_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (t5_fold_dir / "metadata.json").write_text(
            json.dumps(t5_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        token_summary["folds"].append(token_meta)
        t5_summary["folds"].append(t5_meta)

    (args.token_output_dir / "summary.json").write_text(
        json.dumps(token_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.t5_output_dir / "summary.json").write_text(
        json.dumps(t5_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
