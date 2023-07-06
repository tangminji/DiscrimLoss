#!/bin/bash

#SBATCH -J 40utkl25r
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 50:00:00
#SBATCH --mem 10240
#SBATCH -o 40utkfacel2_ea_gak_tanh_5run.out

source ~/.bashrc

# 设置运行环境
conda activate discrimloss

for i in 91204 58377 34908 94573 29984
do
python main_discrimloss_utkface_with_hy_params.py \
  --dataset  UTKFACE \
  --reg_loss_type L2 \
  --lr 0.001 \
  --task_type regression \
  --rand_fraction 0.40 \
  --epochs 100 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --log_dir UTKFACE/ea_emak_tanh_L2_results_5run/fract0.40/seed$i/logs \
  --save_dir UTKFACE/ea_emak_tanh_L2_results_5run/fract0.40/seed$i/weights \
  --seed $i \
  --learn_inst_parameters \
  --utkface_loss_type ea_emak_tanh \
  --utkface_with_params_path params/UTKFACE/L2_loss/fract0.40/hy_best_params.json

done



