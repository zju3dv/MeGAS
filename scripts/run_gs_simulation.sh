#!/bin/bash

# 定义任务数组
tasks=(
  "bash scripts/1_run_candy_house.sh"
  "bash scripts/3_run_sugar_house.sh"
  "bash scripts/4_run_toy_bulldozer.sh"
  "bash scripts/5_run_chocolate_bunny.sh"
)

# 总GPU数量
num_gpus=2

# 循环任务并分配GPU
for i in "${!tasks[@]}"; do
  gpu_id=$(( i % num_gpus ))
  echo "Running ${tasks[i]} on GPU $gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id ${tasks[i]} &
done

wait  # 等待所有任务完成
