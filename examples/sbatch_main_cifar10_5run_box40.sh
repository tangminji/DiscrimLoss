#!/bin/bash
            
#SBATCH -J 40cf10r5
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 30:00:00
#SBATCH --mem 20240
#SBATCH -o 40cifar10_5run_box.out

source ~/.bashrc

conda activate discrimloss

for i in 1 2 3 4 5
do
python main_discrimloss_cifar_with_hy_params.py \
--dataset  CIFAR10 \
--nr_classes 10 \
--rand_fraction 0.40 \
--epochs 120 \
--lr 0.1 \
--per_gpu_train_batch_size 128 \
--per_gpu_eval_batch_size 128 \
--log_dir CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.40/seed$i/logs \
--save_dir CIFAR10/box_ea_emak_tanh_newq_lr1e-1_results/fract0.40/seed$i/weights \
--seed $i \
--learn_inst_parameters \
--cifar_loss_type ea_emak_tanh_newq \
--cifar_with_params_path params/CIFAR-10/fract0.40/hy_best_params.json
done
    
    