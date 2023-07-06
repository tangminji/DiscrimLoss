# DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination
This repository accompanies the research paper, 
[DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination](
https://ieeexplore.ieee.org/document/10167857)
(accepted at IEEE Transactions on Multimedia). 


## DiscrimLoss

In this paper, we append a novel loss function DiscrimLoss. Its main effect is to automatically and stably estimate the importance of easy samples and difficult samples (including hard and incorrect samples) at the early stages of training to improve the model performance. Specifically, a model should learn from hard samples to promote generalization rather than overfit to incorrect ones. Then, during the following stages, DiscrimLoss is dedicated to discriminating between hard and incorrect samples to improve the model generalization. Such a training strategy can be formulated dynamically in a self-supervised manner, effectively mimicking the main principle of curriculum learning. Experiments on image classification, image regression, text sequence regression, and event relation reasoning demonstrate the versatility and effectiveness of our method, particularly in the presence of diversified noise levels.

## Environments
You can setup python environment with:
```
conda create -n discrimloss python=3.6.13
conda activate discrimloss
pip install -r requirements.txt
```

## Data
We recommand you put add datasets in the `./dataset` folder. Our experiment was conducted on the following datasets:
+ MNIST
+ CIFAR10
+ CIFAR100
+ UTKFace
+ DIGITSUM
+ Clothing1M
+ WikiHow(zhang 2020): https://github.com/zharry29/wikihow-goal-step


## Running
We provide our params and some shell example in `params` and `examples` folder. You can adjust them according to our paper to reproduct our experiment. Here's a example:
```
cp examples/sbatch_main_cifar10_5run_box40.sh ..
sbatch sbatch_main_cifar10_5run_box40.sh
```
You can also run the shell with `bash`.


## Citation
If you find this code useful in your research then please cite:
```
@ARTICLE{10167857,
  author={Wu, Tingting and Ding, Xiao and Zhang, Hao and Gao, Jinglong and Tang, Minji and Du, Li and Qin, Bing and Liu, Ting},
  journal={IEEE Transactions on Multimedia}, 
  title={DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2023.3290477}}

```
Note: Our implementation uses parts of some public codes [1-2]  
[1] Data Parameters: https://github.com/apple/ml-data-parameters  
[2] SuperLoss: https://github.com/AlanChou/Super-Loss