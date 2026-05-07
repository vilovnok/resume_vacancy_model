#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_STAGE1="deepvk/RuModernBERT-base"
MODEL_NAME_STAGE3="deepvk/RuModernBERT-base"
MODEL_NAME_STAGE4="deepvk/RuModernBERT-small"
MODEL_NAME_STAGE5="deepvk/RuModernBERT-small"

DATA_PATH="./benchmarks/raw/hr"
INDEX_PATH="./benchmarks/faiss/hr"

RUN_NAME_STAGE1="deepvk_RuModernBERT_base_v1"
RUN_NAME_STAGE3="deepvk_RuModernBERT_base_v1"
RUN_NAME_STAGE4="deepvk_RuModernBERT_base_v1"
RUN_NAME_STAGE5="deepvk_RuModernBERT_small_v1"

mkdir -p "$DATA_PATH" "$INDEX_PATH"

run_stage1() {
  local result_path="./benchmarks/result/stage1"
  local log_dir="./logs/stage1"
  local checkpoint_dir="./checkpoints/stage1"

  mkdir -p "$result_path" "$log_dir" "$checkpoint_dir"

  accelerate launch --num_processes=1 -m stage1.main \
    --model_name "$MODEL_NAME_STAGE1" \
    --run_name "$RUN_NAME_STAGE1" \
    --data_path "$DATA_PATH" \
    --index_path "$INDEX_PATH" \
    --result_path "$result_path" \
    --log_dir "$log_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --extension "flat" \
    --fp16 \
    --hard_negative_strategy "in_batch" \
    --seed 42 \
    --test_only
}

run_stage3() {
  local result_path="./benchmarks/result/stage3"
  local log_dir="./logs/stage3"
  local checkpoint_dir="./checkpoints/stage3"

  mkdir -p "$result_path" "$log_dir" "$checkpoint_dir"

  accelerate launch --num_processes=1 -m stage3.main \
    --model_name "$MODEL_NAME_STAGE3" \
    --run_name "$RUN_NAME_STAGE3" \
    --data_path "$DATA_PATH" \
    --index_path "$INDEX_PATH" \
    --result_path "$result_path" \
    --log_dir "$log_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --extension "flat" \
    --fp16 \
    --hard_negative_strategy "in_batch" \
    --seed 42 \
    --test_only
}

run_stage4() {
  local result_path="./benchmarks/result/stage4"
  local log_dir="./logs/stage4"
  local checkpoint_dir="./checkpoints/stage4"

  mkdir -p "$result_path" "$log_dir" "$checkpoint_dir"

  accelerate launch --num_processes=1 -m stage4.main \
    --model_name "$MODEL_NAME_STAGE4" \
    --run_name "$RUN_NAME_STAGE4" \
    --data_path "$DATA_PATH" \
    --index_path "$INDEX_PATH" \
    --result_path "$result_path" \
    --log_dir "$log_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --extension "flat" \
    --fp16 \
    --hard_negative_strategy "in_batch" \
    --seed 42 \
    --test_only
}

run_stage5() {
  local result_path="./benchmarks/result/stage5"
  local log_dir="./logs/stage5"
  local checkpoint_dir="./checkpoints/stage5"

  mkdir -p "$result_path" "$log_dir" "$checkpoint_dir"

  accelerate launch --num_processes=1 -m stage5.main \
    --model_name "$MODEL_NAME_STAGE5" \
    --run_name "$RUN_NAME_STAGE5" \
    --data_path "$DATA_PATH" \
    --index_path "$INDEX_PATH" \
    --result_path "$result_path" \
    --log_dir "$log_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --extension "flat" \
    --fp16 \
    --hard_negative_strategy "in_batch" \
    --seed 42 \
    --test_only
}

run_all() {
  run_stage1
  run_stage3
  run_stage4
  run_stage5
}

usage() {
  cat <<EOF
Usage:
  $0 stage1
  $0 stage3
  $0 stage4
  $0 stage5
  $0 all

You can pass several stages:
  $0 stage1 stage3 stage5
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for stage in "$@"; do
  case "$stage" in
    all)
      run_all
      ;;
    stage1)
      run_stage1
      ;;
    stage3)
      run_stage3
      ;;
    stage4)
      run_stage4
      ;;
    stage5)
      run_stage5
      ;;
    *)
      echo "Unknown stage: $stage"
      usage
      exit 1
      ;;
  esac
done