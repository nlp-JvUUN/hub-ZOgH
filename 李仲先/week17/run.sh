#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
if [ -z "$CC" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
fi

echo "=== 1/4 prepare GSM8K data ==="
if [ -f data/train.parquet ]; then
    echo "data already prepared, skip"
else
    python prepare_data.py
fi

echo "=== 2/4 evaluate baseline (before GRPO) ==="
if [ -f output/eval_before.json ]; then
    echo "baseline eval exists, skip"
else
    python eval_math.py --num_samples 200 --save_path output/eval_before.json
fi

echo "=== 3/4 GRPO training ==="
python train_grpo.py \
    --model_path Qwen2.5-0.5B-Instruct \
    --output_dir output/grpo-qwen-gsm8k \
    --max_steps 50 \
    --num_generations 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_completion_length 1024

echo "=== 4/4 evaluate after GRPO ==="
python eval_math.py \
    --adapter_path output/grpo-qwen-gsm8k \
    --num_samples 200 \
    --save_path output/eval_after.json

echo "done. compare output/eval_before.json vs output/eval_after.json"
