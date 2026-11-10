#!/usr/bin/env bash
# 训练 ACT。用法：./train_act.sh [步数] [batch]
# 依赖：act_data/lerobot 数据集（先跑 to_lerobot.py）
set -e
STEPS=${1:-20000}
BATCH=${2:-8}
cd /home/po/vision
export -n HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true
/home/po/act-env/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/piper_grasp \
  --dataset.root=/home/po/vision/act_data/lerobot \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=/home/po/vision/act_train/run1 \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=2000 \
  --log_freq=50 \
  --num_workers=4 \
  --wandb.enable=false \
  --policy.push_to_hub=false
