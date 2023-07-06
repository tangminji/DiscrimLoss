#!/bin/bash

#SBATCH -J 4dswigo+
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:2
#SBATCH -w gpu25
#SBATCH -t 40:00:00
#SBATCH --mem 20240
#SBATCH -o wikihow_disc_goal.out

source ~/.bashrc

# 设置运行环境
conda activate discrimloss



python main_discrimloss_wikihow.py \
  --num_train_epochs 5 \
  --sub_epochs_step 1000 \
  --lr 5e-5 \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --per_gpu_test_batch_size 24 \
  --log_dir WIKIHOW/GOAL/newcode_ea_emak_tanh_a0.2_p1.2_newq10_eps3_ses1000_lrp100/logs \
  --save_dir WIKIHOW/GOAL/newcode_ea_emak_tanh_a0.2_p1.2_newq10_eps3_ses1000_lrp100/weights \
  --wikihow_loss_type ea_emak_tanh_newq \
  --a 0.2 \
  --p 1.2 \
  --newq 10 \
  --sup_eps 3 \
  --seed 1 \
  --lr_inst_param 100 \
