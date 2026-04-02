#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from evaluate_slot_json import (
    f1_score,
    get_candidate_slots,
    parse_target_text,
    safe_div,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an extractive MRC baseline for SNIPS slot filling."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/public_model/microsoft-deberta-v3-large"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/snips_lodo_mrc"))
    parser.add_argument("--gold-root", type=Path, default=Path("data/snips_lodo_llama"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/deberta_v3_mrc")
    )
    parser.add_argument("--fold", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--scheduler", choices=["cosine", "linear"], default="linear")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-answer-length", type=int, default=12)
    parser.add_argument("--doc-stride", type=int, default=96)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--run-test-after-training", action="store_true")
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["test_all", "test_seen_slots", "test_unseen_slots"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


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


def device_and_dtype(use_bf16: bool) -> Tuple[torch.device, Optional[torch.dtype]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if use_bf16 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16
        return device, torch.float16
    return torch.device("cpu"), None


def build_scheduler(name: str, optimizer: AdamW, warmup_steps: int, total_steps: int):
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def load_fold_records(data_root: Path, fold: str) -> Dict[str, List[Dict[str, object]]]:
    fold_dir = data_root / fold
    names = [
        "train",
        "dev",
        "test_all",
        "test_seen_slots",
        "test_unseen_slots",
        "test_no_slots",
    ]
    return {
        name: read_jsonl(fold_dir / f"{name}.jsonl")
        for name in names
        if (fold_dir / f"{name}.jsonl").exists()
    }


def load_gold_records(gold_root: Path, fold: str) -> Dict[str, List[Dict[str, object]]]:
    fold_dir = gold_root / fold
    names = ["dev", "test_all", "test_seen_slots", "test_unseen_slots"]
    return {
        name: read_jsonl(fold_dir / f"{name}.jsonl")
        for name in names
        if (fold_dir / f"{name}.jsonl").exists()
    }


class TrainQADataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        doc_stride: int,
    ):
        self.features: List[Dict[str, object]] = []
        for record in records:
            question = str(record["question"])
            context = str(record["context"])
            answers = record.get("answers", {})
            answer_texts = answers.get("text") if isinstance(answers, dict) else []
            answer_starts = (
                answers.get("answer_start") if isinstance(answers, dict) else []
            )
            is_impossible = bool(record.get("is_impossible", False))

            encoded = tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )

            for idx in range(len(encoded["input_ids"])):
                input_ids = encoded["input_ids"][idx]
                attention_mask = encoded["attention_mask"][idx]
                offset_mapping = encoded["offset_mapping"][idx]
                sequence_ids = encoded.sequence_ids(idx)
                cls_index = input_ids.index(tokenizer.cls_token_id)

                start_position = cls_index
                end_position = cls_index
                if not is_impossible and answer_texts and answer_starts:
                    answer_text = str(answer_texts[0])
                    answer_start = int(answer_starts[0])
                    answer_end = answer_start + len(answer_text)
                    context_start = next(
                        (i for i, sid in enumerate(sequence_ids) if sid == 1), None
                    )
                    context_end = next(
                        (
                            i
                            for i in range(len(sequence_ids) - 1, -1, -1)
                            if sequence_ids[i] == 1
                        ),
                        None,
                    )
                    if context_start is not None and context_end is not None:
                        if not (
                            offset_mapping[context_start][0] <= answer_start
                            and offset_mapping[context_end][1] >= answer_end
                        ):
                            start_position = cls_index
                            end_position = cls_index
                        else:
                            start_position = context_start
                            while (
                                start_position <= context_end
                                and offset_mapping[start_position][0] <= answer_start
                            ):
                                start_position += 1
                            start_position -= 1
                            end_position = context_end
                            while (
                                end_position >= context_start
                                and offset_mapping[end_position][1] >= answer_end
                            ):
                                end_position -= 1
                            end_position += 1

                self.features.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": start_position,
                        "end_positions": end_position,
                    }
                )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.features[index]


class EvalQADataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        doc_stride: int,
    ):
        self.features: List[Dict[str, object]] = []
        for record in records:
            question = str(record["question"])
            context = str(record["context"])
            encoded = tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            for idx in range(len(encoded["input_ids"])):
                sequence_ids = encoded.sequence_ids(idx)
                offsets = []
                for token_idx, offset in enumerate(encoded["offset_mapping"][idx]):
                    offsets.append(offset if sequence_ids[token_idx] == 1 else None)
                self.features.append(
                    {
                        "feature_id": f"{record['id']}__{idx}",
                        "example_id": str(record["example_id"]),
                        "qa_id": str(record["id"]),
                        "slot": str(record["slot"]),
                        "question": question,
                        "context": context,
                        "target": record.get("answers", {}),
                        "metadata": record.get("metadata", {}),
                        "split": record.get("split"),
                        "input_ids": encoded["input_ids"][idx],
                        "attention_mask": encoded["attention_mask"][idx],
                        "offset_mapping": offsets,
                    }
                )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.features[index]


class TrainCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example["input_ids"]) for example in batch)
        input_ids = []
        attention_mask = []
        start_positions = []
        end_positions = []
        for example in batch:
            pad = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(example["attention_mask"] + [0] * pad)
            start_positions.append(example["start_positions"])
            end_positions.append(example["end_positions"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }


class EvalCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        max_len = max(len(example["input_ids"]) for example in batch)
        input_ids = []
        attention_mask = []
        for example in batch:
            pad = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(example["attention_mask"] + [0] * pad)
        return {
            "feature_id": [example["feature_id"] for example in batch],
            "example_id": [example["example_id"] for example in batch],
            "qa_id": [example["qa_id"] for example in batch],
            "slot": [example["slot"] for example in batch],
            "question": [example["question"] for example in batch],
            "context": [example["context"] for example in batch],
            "target": [example["target"] for example in batch],
            "metadata": [example["metadata"] for example in batch],
            "split": [example["split"] for example in batch],
            "offset_mapping": [example["offset_mapping"] for example in batch],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def compute_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx:
                outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    grad_accum: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else nullcontext()
    )
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(outputs.loss.item())
        total_steps += 1
    if total_steps % grad_accum != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": total_loss / max(total_steps, 1)}


def best_span_from_logits(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Optional[Tuple[int, int]]],
    context: str,
    max_answer_length: int,
) -> Dict[str, object]:
    null_score = float(start_logits[0] + end_logits[0])
    best_score = -float("inf")
    best_text = ""
    best_start = 0
    best_end = 0
    context_indexes = [idx for idx, offset in enumerate(offsets) if offset is not None]
    top_starts = sorted(
        context_indexes, key=lambda idx: float(start_logits[idx]), reverse=True
    )[:20]
    top_ends = sorted(
        context_indexes, key=lambda idx: float(end_logits[idx]), reverse=True
    )[:20]
    for start_idx in top_starts:
        for end_idx in top_ends:
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_length:
                continue
            start_offset = offsets[start_idx]
            end_offset = offsets[end_idx]
            if start_offset is None or end_offset is None:
                continue
            start_char = start_offset[0]
            end_char = end_offset[1]
            if end_char <= start_char:
                continue
            score = float(start_logits[start_idx] + end_logits[end_idx])
            if score > best_score:
                best_score = score
                best_text = context[start_char:end_char]
                best_start = start_char
                best_end = end_char
    if best_score == -float("inf"):
        best_score = null_score
    return {
        "text": best_text,
        "start": best_start,
        "end": best_end,
        "best_score": best_score,
        "null_score": null_score,
        "score_diff": best_score - null_score,
    }


def gather_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_answer_length: int,
) -> Dict[str, Dict[str, object]]:
    model.eval()
    feature_predictions: Dict[str, Dict[str, object]] = {}
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype is not None
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.detach().cpu()
            end_logits = outputs.end_logits.detach().cpu()
            for idx, feature_id in enumerate(batch["feature_id"]):
                span = best_span_from_logits(
                    start_logits[idx].tolist(),
                    end_logits[idx].tolist(),
                    batch["offset_mapping"][idx],
                    batch["context"][idx],
                    max_answer_length,
                )
                feature_predictions[feature_id] = {
                    "feature_id": feature_id,
                    "example_id": batch["example_id"][idx],
                    "qa_id": batch["qa_id"][idx],
                    "slot": batch["slot"][idx],
                    "question": batch["question"][idx],
                    "context": batch["context"][idx],
                    "target": batch["target"][idx],
                    "metadata": batch["metadata"][idx],
                    "split": batch["split"][idx],
                    **span,
                }
    return feature_predictions


def qa_gold_answer(record: Dict[str, object]) -> str:
    target = record.get("target", {})
    if isinstance(target, dict):
        texts = target.get("text", [])
        if isinstance(texts, list) and texts:
            return str(texts[0])
    return ""


def utterance_gold_payload(metadata: Dict[str, object]) -> str:
    spans = metadata.get("spans", []) if isinstance(metadata, dict) else []
    payload: Dict[str, str] = {}
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            slot = span.get("slot")
            text = span.get("text")
            if isinstance(slot, str) and isinstance(text, str) and slot not in payload:
                payload[slot] = text
    return json.dumps(payload, ensure_ascii=False)


def evaluate_grouped_predictions(
    feature_predictions: Dict[str, Dict[str, object]], threshold: float
) -> Dict[str, object]:
    best_by_qa: Dict[str, Dict[str, object]] = {}
    for prediction in feature_predictions.values():
        qa_id = str(prediction["qa_id"])
        current = best_by_qa.get(qa_id)
        if current is None or float(prediction["score_diff"]) > float(
            current["score_diff"]
        ):
            best_by_qa[qa_id] = prediction

    grouped_rows: Dict[str, Dict[str, object]] = {}
    for prediction in best_by_qa.values():
        example_id = str(prediction["example_id"])
        row = grouped_rows.setdefault(
            example_id,
            {
                "id": example_id,
                "prediction_map": {},
                "metadata": prediction["metadata"],
                "target": utterance_gold_payload(prediction["metadata"]),
                "qa_predictions": [],
            },
        )
        predicted_text = ""
        if float(prediction["score_diff"]) > threshold:
            predicted_text = str(prediction["text"]).strip()
            if predicted_text:
                row["prediction_map"][str(prediction["slot"])] = predicted_text
        row["qa_predictions"].append(
            {
                "qa_id": prediction["qa_id"],
                "slot": prediction["slot"],
                "question": prediction["question"],
                "prediction": predicted_text,
                "score_diff": prediction["score_diff"],
                "target": qa_gold_answer(prediction),
            }
        )

    tp = fp = fn = 0
    exact_match = 0
    per_subset: Dict[str, Counter[str]] = {}
    per_slot: Dict[str, Counter[str]] = {}
    rows: List[Dict[str, object]] = []

    for example_id, row in sorted(grouped_rows.items()):
        prediction_text = json.dumps(row["prediction_map"], ensure_ascii=False)
        candidate_slots = get_candidate_slots({"metadata": row["metadata"]})
        gold_spans = set(parse_target_text(row["target"], candidate_slots)["spans"])
        pred_spans = set(parse_target_text(prediction_text, candidate_slots)["spans"])
        example_tp = len(gold_spans & pred_spans)
        example_fp = len(pred_spans - gold_spans)
        example_fn = len(gold_spans - pred_spans)
        tp += example_tp
        fp += example_fp
        fn += example_fn
        is_exact = int(gold_spans == pred_spans)
        exact_match += is_exact

        subset = "unknown"
        if isinstance(row["metadata"], dict):
            subset = str(
                row["metadata"].get("test_subset")
                or row["metadata"].get("split")
                or "unknown"
            )
        subset_counts = per_subset.setdefault(subset, Counter())
        subset_counts.update(
            tp=example_tp,
            fp=example_fp,
            fn=example_fn,
            examples=1,
            exact_match=is_exact,
        )

        for slot, _ in gold_spans & pred_spans:
            per_slot.setdefault(slot, Counter()).update(tp=1)
        for slot, _ in pred_spans - gold_spans:
            per_slot.setdefault(slot, Counter()).update(fp=1)
        for slot, _ in gold_spans - pred_spans:
            per_slot.setdefault(slot, Counter()).update(fn=1)

        rows.append(
            {
                "id": example_id,
                "prediction": prediction_text,
                "target": row["target"],
                "metadata": row["metadata"],
                "qa_predictions": sorted(
                    row["qa_predictions"], key=lambda item: item["slot"]
                ),
            }
        )

    report = {
        "threshold": threshold,
        "counts": {
            "examples": len(grouped_rows),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_match": exact_match,
        },
        "metrics": {
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "micro_f1": f1_score(tp, fp, fn),
            "exact_match": safe_div(exact_match, len(grouped_rows)),
        },
        "per_subset": {},
        "per_slot": {},
        "rows": rows,
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
        report["per_slot"][slot] = {
            "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
            "f1": f1_score(counts["tp"], counts["fp"], counts["fn"]),
            "support": counts["tp"] + counts["fn"],
        }
    return report


def select_threshold(
    dev_feature_predictions: Dict[str, Dict[str, object]],
) -> Tuple[float, Dict[str, object]]:
    candidates = sorted(
        {
            round(float(pred["score_diff"]), 6)
            for pred in dev_feature_predictions.values()
        }
    )
    if not candidates:
        report = evaluate_grouped_predictions(dev_feature_predictions, 0.0)
        return 0.0, report
    search_values = [min(candidates) - 1.0] + candidates + [max(candidates) + 1.0]
    best_threshold = 0.0
    best_report = None
    best_key = None
    for threshold in search_values:
        report = evaluate_grouped_predictions(dev_feature_predictions, threshold)
        key = (
            report["metrics"]["micro_f1"],
            report["metrics"]["exact_match"],
            -threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_report = report
    assert best_report is not None
    return best_threshold, best_report


def save_prediction_rows(path: Path, report: Dict[str, object]) -> None:
    write_jsonl(path, report["rows"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_root / args.fold / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_args.json", vars(args))

    records = load_fold_records(args.data_root, args.fold)
    gold_records = load_gold_records(args.gold_root, args.fold)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    device, amp_dtype = device_and_dtype(args.bf16)
    model.to(device)

    train_dataset = TrainQADataset(
        records["train"], tokenizer, args.max_seq_length, args.doc_stride
    )
    dev_train_dataset = TrainQADataset(
        records["dev"], tokenizer, args.max_seq_length, args.doc_stride
    )
    dev_eval_dataset = EvalQADataset(
        records["dev"], tokenizer, args.max_seq_length, args.doc_stride
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=TrainCollator(tokenizer),
    )
    dev_loader = DataLoader(
        dev_train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=TrainCollator(tokenizer),
    )
    dev_eval_loader = DataLoader(
        dev_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=EvalCollator(tokenizer),
    )

    total_steps = (
        math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(args.scheduler, optimizer, warmup_steps, total_steps)

    history = []
    best_metrics = None
    best_key = None
    best_threshold = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            amp_dtype,
            args.gradient_accumulation_steps,
        )
        dev_loss = compute_loss(model, dev_loader, device, amp_dtype)
        dev_feature_predictions = gather_predictions(
            model, dev_eval_loader, device, amp_dtype, args.max_answer_length
        )
        threshold, dev_report = select_threshold(dev_feature_predictions)
        dev_pred_path = output_dir / "dev_predictions" / f"epoch_{epoch}.jsonl"
        save_prediction_rows(dev_pred_path, dev_report)
        write_json(
            output_dir / "dev_reports" / f"epoch_{epoch}.json",
            {k: v for k, v in dev_report.items() if k != "rows"},
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "dev_loss": dev_loss,
            "dev_precision": dev_report["metrics"]["precision"],
            "dev_recall": dev_report["metrics"]["recall"],
            "dev_micro_f1": dev_report["metrics"]["micro_f1"],
            "dev_exact_match": dev_report["metrics"]["exact_match"],
            "dev_threshold": threshold,
        }
        history.append(epoch_metrics)
        write_json(output_dir / "history.json", {"epochs": history})

        key = (
            epoch_metrics["dev_micro_f1"],
            epoch_metrics["dev_exact_match"],
            -dev_loss,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = dict(epoch_metrics)
            best_threshold = threshold
            write_json(output_dir / "best_metrics.json", best_metrics)
            model.save_pretrained(output_dir / "best_checkpoint")
            tokenizer.save_pretrained(output_dir / "best_checkpoint")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience_epochs:
                break

    if args.run_test_after_training and (output_dir / "best_checkpoint").exists():
        best_model = AutoModelForQuestionAnswering.from_pretrained(
            output_dir / "best_checkpoint"
        )
        best_model.to(device)
        for split_name in args.test_splits:
            if split_name not in records:
                continue
            eval_dataset = EvalQADataset(
                records[split_name], tokenizer, args.max_seq_length, args.doc_stride
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=EvalCollator(tokenizer),
            )
            feature_predictions = gather_predictions(
                best_model, eval_loader, device, amp_dtype, args.max_answer_length
            )
            report = evaluate_grouped_predictions(feature_predictions, best_threshold)
            save_prediction_rows(
                output_dir / "test_predictions" / f"{split_name}.jsonl", report
            )
            write_json(
                output_dir / "test_reports" / f"{split_name}.json",
                {k: v for k, v in report.items() if k != "rows"},
            )


if __name__ == "__main__":
    main()
