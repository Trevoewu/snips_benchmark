#!/usr/bin/env python3
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# This evaluator is PyTorch-only; disabling TF/Flax avoids optional import failures.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from evaluate_slot_json import (
    f1_score,
    get_candidate_slots,
    parse_target_text,
    safe_div,
)


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
        description="Evaluate base Qwen zero-shot slot filling with prompt schemas."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/public_model/qwen3-4b"),
        help="Path to the base causal LM.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/snips_lodo_llama"),
        help="Root directory containing per-fold JSONL data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/qwen3_4b_zero_shot"),
        help="Directory where predictions and reports are saved.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Held-out domains to evaluate. Defaults to every fold in summary.json.",
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["test_all", "test_seen_slots", "test_unseen_slots"],
        help="Held-out splits to evaluate.",
    )
    parser.add_argument(
        "--schema-paths",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional schema JSON files. Each run uses the per-domain function spec from "
            "the schema in the prompt."
        ),
    )
    parser.add_argument(
        "--include-slot-name-baseline",
        action="store_true",
        help="Also evaluate the original slot-name-only prompt.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Tokenizer truncation length for prompt batches.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Maximum generated answer length.",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization if bitsandbytes is installed; otherwise fail.",
    )
    parser.add_argument(
        "--auto-4bit",
        action="store_true",
        help="Use 4-bit quantization only when bitsandbytes is installed.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 autocast on supported CUDA hardware.",
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def device_and_dtype(use_bf16: bool) -> Tuple[torch.device, Optional[torch.dtype]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if use_bf16 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16
        return device, torch.float16
    return torch.device("cpu"), None


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_4bit = args.use_4bit or (args.auto_4bit and bitsandbytes_available())
    if args.use_4bit and not bitsandbytes_available():
        raise RuntimeError("--use-4bit requested but bitsandbytes is not installed.")

    device, dtype = device_and_dtype(args.bf16)
    model_kwargs: Dict[str, object] = {"trust_remote_code": True}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if dtype == torch.bfloat16
            else torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype if dtype is not None else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if not use_4bit:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def load_fold_names(data_root: Path) -> List[str]:
    summary_path = data_root / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        folds = summary.get("folds")
        if isinstance(folds, list):
            names = []
            for fold in folds:
                if isinstance(fold, dict) and isinstance(
                    fold.get("heldout_domain"), str
                ):
                    names.append(str(fold["heldout_domain"]))
            if names:
                return names
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def build_slot_name_prompt(
    domain: str, slot_names: Sequence[str], utterance: str
) -> str:
    slot_list = ", ".join(slot_names)
    return f"Domain: {domain}\nAllowed slot names: {slot_list}\nUtterance: {utterance}"


def load_schema_lookup(path: Path) -> Dict[str, Dict[str, object]]:
    payload = read_json(path)
    functions = payload.get("functions")
    if not isinstance(functions, list):
        raise ValueError(
            f"Schema file {path} must contain a top-level 'functions' list."
        )

    lookup: Dict[str, Dict[str, object]] = {}
    for entry in functions:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name:
            lookup[name] = entry
    if not lookup:
        raise ValueError(f"Schema file {path} does not contain any named functions.")
    return lookup


def build_schema_prompt(
    domain: str,
    schema_lookup: Dict[str, Dict[str, object]],
    slot_names: Sequence[str],
    utterance: str,
) -> str:
    function_spec = schema_lookup.get(domain)
    if function_spec is None:
        raise ValueError(f"Schema is missing a function entry for domain {domain!r}.")

    parameters = function_spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Schema entry for domain {domain!r} is missing parameters.")
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"Schema entry for domain {domain!r} is missing properties.")

    slot_items = []
    for slot_name in slot_names:
        slot_spec = properties.get(slot_name)
        description = ""
        if isinstance(slot_spec, dict) and isinstance(
            slot_spec.get("description"), str
        ):
            description = slot_spec["description"].strip()
        if description:
            slot_items.append(f"{slot_name}: {description}")
        else:
            slot_items.append(slot_name)

    return (
        f"Domain: {domain}\n"
        f"Allowed slot names: {', '.join(slot_items)}\n"
        f"Utterance: {utterance}"
    )


def build_messages(
    record: Dict[str, object],
    schema_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Record {record.get('id')!r} is missing metadata.")

    domain = metadata.get("domain")
    utterance = metadata.get("utterance")
    slot_names = metadata.get("candidate_slot_names")
    if not isinstance(domain, str) or not isinstance(utterance, str):
        raise ValueError(f"Record {record.get('id')!r} is missing domain or utterance.")
    if not isinstance(slot_names, list) or not all(
        isinstance(name, str) for name in slot_names
    ):
        raise ValueError(
            f"Record {record.get('id')!r} is missing candidate slot names."
        )

    if schema_lookup is None:
        user_prompt = build_slot_name_prompt(domain, slot_names, utterance)
    else:
        user_prompt = build_schema_prompt(domain, schema_lookup, slot_names, utterance)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def chat_text(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    template_kwargs = {}
    if add_generation_prompt:
        # Keep Qwen3 generation prompts aligned with the assistant-side training template.
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


class PromptDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        tokenizer: Any,
        schema_lookup: Optional[Dict[str, Dict[str, object]]],
    ):
        self.examples: List[Dict[str, object]] = []
        for record in records:
            messages = build_messages(record, schema_lookup=schema_lookup)
            self.examples.append(
                {
                    "id": record["id"],
                    "prompt_text": chat_text(
                        tokenizer, messages, add_generation_prompt=True
                    ),
                    "target": record["target"],
                    "metadata": record.get("metadata", {}),
                    "split": record.get("split"),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


def prompt_collator(
    tokenizer: Any, batch: Sequence[Dict[str, object]], max_seq_length: int
) -> Dict[str, object]:
    prompts = [example["prompt_text"] for example in batch]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return {
        "id": [example["id"] for example in batch],
        "target": [example["target"] for example in batch],
        "metadata": [example["metadata"] for example in batch],
        "split": [example["split"] for example in batch],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


@torch.no_grad()
def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, object]],
    schema_lookup: Optional[Dict[str, Dict[str, object]]],
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    dataset = PromptDataset(records, tokenizer, schema_lookup=schema_lookup)
    generation_config = deepcopy(getattr(model, "generation_config", None))
    eos_token_id = getattr(generation_config, "eos_token_id", tokenizer.eos_token_id)
    pad_token_id = getattr(generation_config, "pad_token_id", tokenizer.pad_token_id)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.temperature = None
        generation_config.pad_token_id = pad_token_id
        generation_config.eos_token_id = eos_token_id

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: prompt_collator(tokenizer, batch, max_seq_length),
    )

    total_tp = total_fp = total_fn = 0
    exact_match = 0
    per_subset: Dict[str, Dict[str, int]] = {}
    per_slot: Dict[str, Dict[str, int]] = {}
    prediction_rows: List[Dict[str, object]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        for idx, example_id in enumerate(batch["id"]):
            generated_ids = generated[idx][int(prompt_lengths[idx]) :]
            prediction_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            gold_target = batch["target"][idx]
            metadata = batch["metadata"][idx]
            candidate_slots = get_candidate_slots({"metadata": metadata})
            gold_parsed = parse_target_text(gold_target, candidate_slots)
            gold_spans = set(
                gold_parsed["spans"] if isinstance(gold_parsed["spans"], list) else []
            )
            pred_result = parse_target_text(prediction_text, candidate_slots)
            pred_spans = set(
                pred_result["spans"] if isinstance(pred_result["spans"], list) else []
            )

            tp = len(gold_spans & pred_spans)
            fp = len(pred_spans - gold_spans)
            fn = len(gold_spans - pred_spans)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            is_exact = int(gold_spans == pred_spans)
            exact_match += is_exact

            subset = str(
                metadata.get("test_subset") or batch["split"][idx] or "unknown"
            )
            subset_counts = per_subset.setdefault(
                subset,
                {"tp": 0, "fp": 0, "fn": 0, "examples": 0, "exact_match": 0},
            )
            subset_counts["tp"] += tp
            subset_counts["fp"] += fp
            subset_counts["fn"] += fn
            subset_counts["examples"] += 1
            subset_counts["exact_match"] += is_exact

            for slot, _ in gold_spans & pred_spans:
                slot_counts = per_slot.setdefault(slot, {"tp": 0, "fp": 0, "fn": 0})
                slot_counts["tp"] += 1
            for slot, _ in pred_spans - gold_spans:
                slot_counts = per_slot.setdefault(slot, {"tp": 0, "fp": 0, "fn": 0})
                slot_counts["fp"] += 1
            for slot, _ in gold_spans - pred_spans:
                slot_counts = per_slot.setdefault(slot, {"tp": 0, "fp": 0, "fn": 0})
                slot_counts["fn"] += 1

            prediction_rows.append(
                {
                    "id": example_id,
                    "prediction": prediction_text,
                    "target": gold_target,
                    "metadata": metadata,
                    "issues": pred_result["issues"],
                }
            )

    report = {
        "counts": {
            "examples": len(records),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "exact_match": exact_match,
        },
        "metrics": {
            "precision": safe_div(total_tp, total_tp + total_fp),
            "recall": safe_div(total_tp, total_tp + total_fn),
            "micro_f1": f1_score(total_tp, total_fp, total_fn),
            "exact_match": safe_div(exact_match, len(records)),
        },
        "per_subset": {},
        "per_slot": {},
    }
    for subset, counts in sorted(per_subset.items()):
        report["per_subset"][subset] = {
            "examples": counts["examples"],
            "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
            "micro_f1": f1_score(counts["tp"], counts["fp"], counts["fn"]),
            "exact_match": safe_div(counts["exact_match"], counts["examples"]),
        }
    for slot, counts in sorted(per_slot.items()):
        slot_tp = counts["tp"]
        slot_fp = counts["fp"]
        slot_fn = counts["fn"]
        report["per_slot"][slot] = {
            "precision": safe_div(slot_tp, slot_tp + slot_fp),
            "recall": safe_div(slot_tp, slot_tp + slot_fn),
            "f1": f1_score(slot_tp, slot_fp, slot_fn),
            "support": slot_tp + slot_fn,
        }

    if output_path is not None:
        write_jsonl(output_path, prediction_rows)

    return report


def trim_records(
    records: Sequence[Dict[str, object]], max_examples: Optional[int]
) -> List[Dict[str, object]]:
    if max_examples is None:
        return list(records)
    if max_examples < 1:
        raise ValueError("--max-examples-per-split must be at least 1.")
    return list(records[:max_examples])


def aggregate_split_reports(
    reports_by_fold: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    total_examples = total_tp = total_fp = total_fn = total_exact = 0
    micro_f1_values: List[float] = []
    exact_match_values: List[float] = []

    for report in reports_by_fold.values():
        counts = report.get("counts", {})
        metrics = report.get("metrics", {})
        if not isinstance(counts, dict) or not isinstance(metrics, dict):
            continue
        examples = int(counts.get("examples", 0))
        total_examples += examples
        total_tp += int(counts.get("tp", 0))
        total_fp += int(counts.get("fp", 0))
        total_fn += int(counts.get("fn", 0))
        total_exact += int(counts.get("exact_match", 0))
        if examples > 0:
            micro_f1_values.append(float(metrics.get("micro_f1", 0.0)))
            exact_match_values.append(float(metrics.get("exact_match", 0.0)))

    return {
        "counts": {
            "folds": len(reports_by_fold),
            "examples": total_examples,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "exact_match": total_exact,
        },
        "metrics": {
            "micro_precision": safe_div(total_tp, total_tp + total_fp),
            "micro_recall": safe_div(total_tp, total_tp + total_fn),
            "micro_f1": f1_score(total_tp, total_fp, total_fn),
            "exact_match": safe_div(total_exact, total_examples),
            "macro_micro_f1": safe_div(sum(micro_f1_values), len(micro_f1_values)),
            "macro_exact_match": safe_div(
                sum(exact_match_values), len(exact_match_values)
            ),
        },
    }


def run_name_for_schema(schema_path: Optional[Path]) -> str:
    if schema_path is None:
        return "slot_names"
    return schema_path.stem


def validate_run_names(schema_paths: Sequence[Optional[Path]]) -> None:
    seen = set()
    for schema_path in schema_paths:
        name = run_name_for_schema(schema_path)
        if name in seen:
            raise ValueError(
                f"Duplicate run name {name!r}; schema filenames must have unique stems."
            )
        seen.add(name)


def main() -> None:
    args = parse_args()
    folds = args.folds or load_fold_names(args.data_root)
    run_schema_paths: List[Optional[Path]] = []
    if args.include_slot_name_baseline or not args.schema_paths:
        run_schema_paths.append(None)
    run_schema_paths.extend(args.schema_paths)
    validate_run_names(run_schema_paths)

    model, tokenizer, device = load_model_and_tokenizer(args)
    comparison_summary = {
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "folds": folds,
        "test_splits": list(args.test_splits),
        "runs": {},
    }

    for schema_path in run_schema_paths:
        schema_lookup = None
        if schema_path is not None:
            schema_lookup = load_schema_lookup(schema_path)
        run_name = run_name_for_schema(schema_path)
        run_dir = args.output_root / run_name

        run_summary = {
            "run_name": run_name,
            "schema_path": str(schema_path) if schema_path is not None else None,
            "model_path": str(args.model_path),
            "folds": {},
            "aggregate": {},
        }

        for fold in folds:
            fold_dir = args.data_root / fold
            if not fold_dir.exists():
                raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

            fold_summary = {"splits": {}}
            for split_name in args.test_splits:
                split_path = fold_dir / f"{split_name}.jsonl"
                if not split_path.exists():
                    continue
                records = trim_records(
                    read_jsonl(split_path), args.max_examples_per_split
                )
                report = generate_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    schema_lookup=schema_lookup,
                    batch_size=args.eval_batch_size,
                    max_seq_length=args.max_seq_length,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    output_path=run_dir / fold / "predictions" / f"{split_name}.jsonl",
                )
                write_json(run_dir / fold / "reports" / f"{split_name}.json", report)
                fold_summary["splits"][split_name] = report

            run_summary["folds"][fold] = fold_summary

        aggregate = {}
        for split_name in args.test_splits:
            split_reports = {
                fold: summary["splits"][split_name]
                for fold, summary in run_summary["folds"].items()
                if split_name in summary["splits"]
            }
            aggregate[split_name] = aggregate_split_reports(split_reports)
        run_summary["aggregate"] = aggregate
        write_json(run_dir / "summary.json", run_summary)
        comparison_summary["runs"][run_name] = {
            "schema_path": run_summary["schema_path"],
            "aggregate": aggregate,
        }

    write_json(args.output_root / "summary.json", comparison_summary)


if __name__ == "__main__":
    main()
