#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/public_model/Meta-Llama-3.1-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/snips_lodo_llama}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/llama31_8b_sft}"
GPU_LIST="${GPU_LIST:-0}"
SEED="${SEED:-42}"

DEFAULT_FOLDS=(
  AddToPlaylist
  BookRestaurant
  GetWeather
  PlayMusic
  RateBook
  SearchCreativeWork
  SearchScreeningEvent
)

if (($# > 0)); then
  FOLDS=("$@")
else
  FOLDS=("${DEFAULT_FOLDS[@]}")
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if ((${#GPUS[@]} == 0)); then
  echo "No GPUs configured via GPU_LIST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --seed "$SEED"
  --epochs 6
  --patience-epochs 1
  --learning-rate 2e-4
  --warmup-ratio 0.03
  --scheduler cosine
  --train-batch-size 4
  --eval-batch-size 16
  --gradient-accumulation-steps 2
  --max-seq-length 512
  --max-new-tokens 64
  --lora-r 16
  --lora-alpha 32
  --lora-dropout 0.05
  --bf16
  --auto-4bit
  --run-test-after-training
)

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPUS=()

refresh_active() {
  local -a next_pids=()
  local -a next_gpus=()
  local idx
  for idx in "${!ACTIVE_PIDS[@]}"; do
    if kill -0 "${ACTIVE_PIDS[$idx]}" 2>/dev/null; then
      next_pids+=("${ACTIVE_PIDS[$idx]}")
      next_gpus+=("${ACTIVE_GPUS[$idx]}")
    fi
  done
  ACTIVE_PIDS=("${next_pids[@]}")
  ACTIVE_GPUS=("${next_gpus[@]}")
}

find_free_gpu() {
  local gpu active_gpu used
  for gpu in "${GPUS[@]}"; do
    used=0
    for active_gpu in "${ACTIVE_GPUS[@]:-}"; do
      if [[ "$gpu" == "$active_gpu" ]]; then
        used=1
        break
      fi
    done
    if [[ "$used" -eq 0 ]]; then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

launch_fold() {
  local fold="$1"
  local gpu="$2"
  local log_path="$OUTPUT_ROOT/logs/${fold,,}_seed${SEED}.log"

  echo "Launching $fold on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "$ROOT_DIR/scripts/train_llama_sft.py" \
    --fold "$fold" \
    "${COMMON_ARGS[@]}" \
    > "$log_path" 2>&1 &

  ACTIVE_PIDS+=("$!")
  ACTIVE_GPUS+=("$gpu")
}

for fold in "${FOLDS[@]}"; do
  refresh_active
  until gpu="$(find_free_gpu)"; do
    if ! wait -n; then
      echo "A fold run failed; stopping launcher." >&2
      exit 1
    fi
    refresh_active
  done
  launch_fold "$fold" "$gpu"
done

while ((${#ACTIVE_PIDS[@]} > 0)); do
  if ! wait -n; then
    echo "A fold run failed; stopping launcher." >&2
    exit 1
  fi
  refresh_active
done

echo "All requested folds finished. Outputs: $OUTPUT_ROOT"
