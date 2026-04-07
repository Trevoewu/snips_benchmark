#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the active SNIPS artifacts for LLM and MRC experiments."
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="Pass --dedupe to data builders."
    )
    return parser.parse_args()


def run_step(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def remove_split_artifacts(output_root: Path, split_name: str) -> None:
    for artifact_path in output_root.glob(f"*/{split_name}.jsonl"):
        artifact_path.unlink(missing_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    common = [sys.executable]
    maybe_dedupe = ["--dedupe"] if args.dedupe else []
    input_dir = repo_root / "data/snips_raw"
    temp_lodo_dir = repo_root / "data/.snips_lodo_tmp"

    remove_tree(temp_lodo_dir)
    run_step(
        common
        + [
            str(repo_root / "scripts/build_snips_lodo.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(temp_lodo_dir),
        ]
        + maybe_dedupe
    )
    run_step(
        common
        + [
            str(repo_root / "scripts/build_llama_slot_data.py"),
            "--input-dir",
            str(input_dir),
            "--lodo-dir",
            str(temp_lodo_dir),
        ]
        + maybe_dedupe
    )
    remove_split_artifacts(repo_root / "data/snips_lodo_llama", "test_no_slots")
    run_step(common + [str(repo_root / "scripts/build_mrc_slot_data.py")])
    remove_split_artifacts(repo_root / "data/snips_lodo_mrc", "test_no_slots")
    remove_tree(temp_lodo_dir)
    remove_tree(repo_root / "data/snips_lodo")
    remove_tree(repo_root / "data/eval_reports")


if __name__ == "__main__":
    main()
