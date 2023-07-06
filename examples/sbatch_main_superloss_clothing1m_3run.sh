#!/bin/bash

#SBATCH -J clssupe
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 70:00:00
#SBATCH -w gpu21
#SBATCH --mem 20240
#SBATCH -o superloss_clothing1m_box_3run.out

source ~/.bashrc

# 设置运行环境

conda activate discrimloss
i=0.75
for seed in 1 2 3
do
python main_superloss_clothing1m.py \
  --lam $i \
  --log_dir CLITHING1M/superloss/lam$i/seed$seed/logs \
  --save_dir CLITHING1M/superloss/lam$i/seed$seed/weights \
  --seed $seed
done

