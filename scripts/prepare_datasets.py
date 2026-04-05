#!/usr/bin/env python3
import argparse
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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    common = [sys.executable]
    maybe_dedupe = ["--dedupe"] if args.dedupe else []

    run_step(common + [str(repo_root / "scripts/build_snips_lodo.py")] + maybe_dedupe)
    run_step(
        common + [str(repo_root / "scripts/build_llama_slot_data.py")] + maybe_dedupe
    )
    run_step(common + [str(repo_root / "scripts/build_mrc_slot_data.py")])
    run_step(
        common
        + [
            str(repo_root / "scripts/evaluate_slot_json.py"),
            "--gold",
            str(repo_root / "data/snips_lodo_llama/AddToPlaylist/test_all.jsonl"),
            "--predictions",
            str(repo_root / "data/snips_lodo_llama/AddToPlaylist/test_all.jsonl"),
            "--output",
            str(repo_root / "data/eval_reports/addtoplaylist_gold_eval.json"),
        ]
    )


if __name__ == "__main__":
    main()
