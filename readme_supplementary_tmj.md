# 补充实验
实验环境 discrimloss_torch1.4
## 实验1 a,p,q 超参不敏感 
nr=40% 
CIFAR100

MNIST

## 实验2 非线性ES方法 sin exp
seed=1,2,3 均值方差
CIFAR10 
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar10_3run_sin40.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar10_3run_exp40.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar10_3run_sin60.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar10_3run_exp60.sh
CIFAR100
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar100_3run_sin40.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar100_3run_exp40.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar100_3run_sin60.sh
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_cifar100_3run_exp60.sh

## 实验3 固定K=logC 退化为Reweight
CIFAR10 seed=1(参照wo_es)
/users10/mjtang/wtt/ml-data-parameters-master-20210416/sbatch_main_cifar10_5run_box40_fixk.sh