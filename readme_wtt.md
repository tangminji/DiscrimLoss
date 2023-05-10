文件功能:本项目是data-parameters的工程目录（NIPS19）,额外融入了superloss(NIPS20)和我们的代码

+ common
  + cmd_args.py:所有的参数配置位置
  
  + DiscrimLoss.py:我们的所有的loss设计文件
  
    + 包含添加各种部件的loss，具体对照oneone_meeting文件夹中吴婷婷21_1_31.pptx
  
    + k取常数（e.g., CIFAR100,k=ln(100)=4.605）
  
      + 一阶段
  
      ```
      DiscrimLoss, 对应ppt中P10中的our_base
      DiscrimESLoss, 对应ppt中P10中的our_es(with early suppression)
      DiscrimEALoss, 对应ppt中P10中的our_ea(with es+ ema指数滑动平均),实现时带bias correction
      DiscrimEA_WO_BCLoss,在DiscrimEALoss基础上，但未带bias correction【可做ablation study】
      DiscrimEA_WO_ESLoss, 在DiscrimEALoss基础上,但未带early suppression【可做ablation study】
      DiscrimEA_0Loss,其他情况的loss由于l-k，会导致最小值<0,将l-k修正为l，即保证loss的最小值不会为负数
      ```
  
      + 两阶段（k->gamma*k）
  
      ```
      DiscrimEA_2Loss：一阶段DiscrimEALoss，二阶段有两种模式“switch”,直接切换，否则“iteration”慢慢迭代到目标值切换,已设置迭代步数为TOTAL_STEP,切换时机（main_discrimloss.py中的THRESHOLD，表示占总epochs的比例，运行完该比例后切换）
      DiscrimEA_TANHLoss：使用tanh()进行两阶段切换【最后统一使用的第一种loss】
      ```
  
    + k取全局平均
  
      + 一阶段
      
        ```DiscrimEA_GAKLoss, 利用AverageMeter对象计算全局平均```
      
      + 两阶段
      
        ```DiscrimEA_GAK_TANHLoss:使用tanh()进行两阶段切换【最后统一使用的第二种loss】```
      
    + k取指数滑动平均
  
      + 一阶段
      
      ​	```DiscrimEA_EMAKLoss,利用exponential moving average计算滑动平均 ```
      
      + 两阶段
      
      ​    ```DiscrimEA_EMAKLoss:使用tanh()进行两阶段切换【最后统一使用的第三种loss】```
  
  + statistic.py:我们写的基于统计信息构图的文件，包含直方图，线图等:boxing_glove:
  
  + SuperLoss.py:原始文章superloss损失函数
  
  + tune_test.py:调参工具测试（目前未使用）
  
  + utils.py:公共功能调用（原有+新添加）
  
+ dataset

  + cifar_dataset.py : cifar分类数据集:arrow_up_small:
  + imagenet_dataset.py：imagenet分类数据集
  + mnist_dataset.py ：手写数字分类数据集
  + utkface_dataset.py ：图片任务年龄识别回归数据集

+ models

  + resnet.py
  + wide_resnet.py

+ myplot

  + 存储画图结果

+ optimizer

  + sparrse_sgd.py：data_parameter需要用到

+ 其他

  + main_cifar.py：data parameter原始文件，原始采用单卡多线程运行方式（现不用）
  + main_cifar_parallel.py：在maIn_cifar.py上修改为多卡单线程版本（现用）:arrow_up_small:
  + main_discrimloss.py：我们的主函数:arrow_up_small:
  + main_discrimloss_hy.py:我们的主函数+hyperopt调参，后期调参用
  + main_discrimloss_utkface.py:我们的针对UTKFACE的主函数，lr设置略有不同，所以单独开了一个文件
  + main_discrimloss_utkface_hy.py:我们的针对UTKFACE的主函数+hyperopt调参（主要调整a,p,q三个超参）
  + main_discrimloss_tune.py：加入调参工具的主函数（现未用，如有需要还可参考子事件工作jointConstrainedLearning-main中的hyperopt工具相关代码）
  + main_imagenet.py：data parameter中利用imagenet数据集的主函数（现未用）
  + main_superloss.py：从superloss项目移植过来的superloss主函数:arrow_up_small:
  + statistic_graph.py：画图函数:boxing_glove:

+ 生成结果1：CIFAR100

  + baseline_results：main_cifar_parallel.py生成结果路径

    + 未利用课程学习CL，添加不同比例噪声（见README.md）

      ```
      python main_cifar.py \
        --rand_fraction 0.4 \
      ```

  + datapara_results：main_cifar_parallel.py生成结果路径

  + superloss_results：main_superloss.py生成结果路径

  + discrimloss_results：main_discrimloss.py生成结果路径