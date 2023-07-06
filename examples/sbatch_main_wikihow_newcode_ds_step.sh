#!/bin/bash

#SBATCH -J 2dswist3+
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-sxm2-16gb:2
#SBATCH -t 40:00:00
#SBATCH --mem 20240
#SBATCH -o wikihow_disc_step.out

source ~/.bashrc

# 设置运行环境
conda activate discrimloss



python main_discrimloss_wikihow.py \
  --num_train_epochs 3 \
  --sub_epochs_step 1000 \
  --lr 5e-5 \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --per_gpu_test_batch_size 24 \
  --log_dir WIKIHOW/STEP/newcode_ea_emak_tanh_a0.2_p1.2_newq10_eps3_ses1000_lrp50/seed3/logs \
  --save_dir WIKIHOW/STEP/newcode_ea_emak_tanh_a0.2_p1.2_newq10_eps3_ses1000_lrp50/seed3/weights \
  --wikihow_loss_type ea_emak_tanh_newq \
  --a 0.2 \
  --p 1.2 \
  --newq 11 \
  --sup_eps 3 \
  --seed 3 \
  --lr_inst_param 50 \
  --train_data_path ./dataset/wikihow/step
