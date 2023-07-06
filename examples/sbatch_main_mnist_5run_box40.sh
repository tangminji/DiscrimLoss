#!/bin/bash

#SBATCH -J 40minb5
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 30:00:00
#SBATCH -w gpu17
#SBATCH --mem 20240
#SBATCH -o 40mnist5run_box.out

source ~/.bashrc

# 设置运行环境
# salloc  -p compute -N1 --gres=gpu:geforce_rtx_2080_ti:1 -t 15:00:00 --mem 20240
conda activate discrimloss

for i in 1 2 3 4 5
do
python main_discrimloss_mnist_with_hy_params.py \
  --dataset  MNIST \
  --nr_classes 10 \
  --rand_fraction 0.40 \
  --epochs 20 \
  --lr 0.1 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --log_dir MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.40/seed$i/logs \
  --save_dir MNIST/box_ea_emak_tanh_newq_lr1e-1_epo20/fract0.40/seed$i/weights \
  --seed $i \
  --learn_inst_parameters \
  --mnist_loss_type ea_emak_tanh_newq \
  --mnist_with_params_path params/MNIST/fract0.40/hy_best_params.json
done

