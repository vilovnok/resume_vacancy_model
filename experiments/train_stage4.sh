MODEL_NAME="deepvk/RuModernBERT-base"
ST_MODEL_NAME="deepvk/RuModernBERT-small"

DATA_PATH="./benchmarks/raw/hr/"
INDEX_PATH="./benchmarks/faiss/hr/"
RESULT_PATH="./benchmarks/result/stage4/"

LOG_DIR="./logs/stage4"
CHECKPOINT_DIR="./checkpoints/stage4"

mkdir -p "$DATA_PATH"
mkdir -p "$INDEX_PATH"
mkdir -p "$RESULT_PATH"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5

accelerate launch --num_processes=5 -m stage4.main \
  --model_name $MODEL_NAME \
  --student_model_name $ST_MODEL_NAME \
  --run_name "deepvk_RuModernBERT_base_v1" \
  --data_path $DATA_PATH \
  --index_path $INDEX_PATH \
  --result_path $RESULT_PATH \
  --log_dir $LOG_DIR \
  --checkpoint_dir $CHECKPOINT_DIR \
  --extension "flat" \
  --num_epochs 4 \
  --fp16 \
  --hard_negative_strategy "in_batch" \
  --seed 42 \
  --pre_train
