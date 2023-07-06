#!/bin/bash

#SBATCH -J clodis3
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 30:00:00
#SBATCH -w gpu21
#SBATCH --mem 20240
#SBATCH -o discrimloss_clothing1m_box_3run.out

source ~/.bashrc

# 设置运行环境

conda activate discrimloss

for i in 1 2 3
do
python main_discrimloss_clothing1m_with_hy_params.py \
  --dataset  Clothing1M \
  --seed $i \
  --log_dir CLITHING1M/box_discrimloss_ea_emak_tanh_newq/seed$i/logs \
  --save_dir CLITHING1M/box_discrimloss_ea_emak_tanh_newq/seed$i/weights \
  --learn_inst_parameters \
  --clothing1m_loss_type ea_emak_tanh_newq \
  --clothing1m_with_params_path params/Clothing1m/hy_best_params.json
done

