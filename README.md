# DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination
This repository accompanies the research paper, 
[DiscrimLoss: A Universal Loss for Hard Samples and Incorrect Samples Discrimination](
https://ieeexplore.ieee.org/document/10167857)
(accepted at IEEE Transactions on Multimedia). 


## DiscrimLoss

In this paper, we append a novel loss function DiscrimLoss. Its main effect is to automatically and stably estimate the importance of easy samples and difficult samples (including hard and incorrect samples) at the early stages of training to improve the model performance. Specifically, a model should learn from hard samples to promote generalization rather than overfit to incorrect ones. Then, during the following stages, DiscrimLoss is dedicated to discriminating between hard and incorrect samples to improve the model generalization. Such a training strategy can be formulated dynamically in a self-supervised manner, effectively mimicking the main principle of curriculum learning. Experiments on image classification, image regression, text sequence regression, and event relation reasoning demonstrate the versatility and effectiveness of our method, particularly in the presence of diversified noise levels.

## Environments
We run our code on Nvidia Tesla V100 SXM2 16GB with python 3.6.13. You can setup python environment with:
```
pip install -r requirements.txt
```

## Data

## How to Run


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
