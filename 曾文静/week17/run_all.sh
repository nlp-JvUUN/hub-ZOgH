#!/bin/bash
# 一键运行：SFT → 基线摸底 → GRPO → 复测 → 对比（全程 CPU，约 5 分钟）
# 用法：bash run_all.sh   （在 week17 根目录执行）
set -e
cd "$(dirname "$0")/src"
PY=${PY:-/opt/miniconda3/envs/py312/bin/python}

echo "===== [1/5] SFT 基线训练（750 步）====="
$PY train_sft.py --steps 750
echo "===== [2/5] 基线摸底 ====="
$PY evaluate.py
echo "===== [3/5] GRPO 训练（40 步）====="
$PY train_grpo.py
echo "===== [4/5] 训练后复测 ====="
$PY evaluate.py --model ../outputs/grpo_ckpt/grpo.pt --out ../outputs/post_train_probe.json
echo "===== [5/5] 对比分析 ====="
$PY compare.py
echo "完成！结果见 outputs/"
