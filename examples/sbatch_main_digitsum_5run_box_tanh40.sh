#!/bin/bash
        
#SBATCH -J 40digb5
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 30:00:00
#SBATCH --mem 20240
#SBATCH -o 40digitsum5run_box_tanh.out

source ~/.bashrc

# 设置运行环境

conda activate discrimloss

for i in 1 2 3 4 5
do
python main_discrimloss_digitsum_with_hy_params.py \
  --hidden_size 256 \
  --rand_fraction 0.40 \
  --lr 0.1 \
  --epochs 100 \
  --per_gpu_train_batch_size 512 \
  --per_gpu_eval_batch_size 512 \
  --per_gpu_test_batch_size 512 \
  --log_dir DIGITSUM/box_ea_tanh_noscheduler_lr1e-1_epo100_lstm_results/fract0.40/seed$i/logs \
  --save_dir DIGITSUM/box_ea_tanh_noscheduler_lr1e-1_epo100_lstm_results/fract0.40/seed$i/weights \
  --no_scheduler \
  --is_discrimloss \
  --seed $i \
  --digitsum_with_params_path params/DigitSum/fract0.40/hy_best_params.json \
  --digitsum_loss_type ea_tanh_newq
done

