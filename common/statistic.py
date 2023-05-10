from common.cmd_args import CURRENT_PATH
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import torchvision.transforms as transforms
from dataset.cifar100_dataset import CIFAR100WithIdx
from common.cmd_args import CIFAR_100_PATH, args
import pickle
from PIL import Image
import matplotlib.image as mpimg

seed = 100
#rand_fraction = 0.4
log_pth = 'CIFAR100/baseline_results/weights_n40_bt128/'
log_sl_pth = 'CIFAR100/superloss_results/weights_n40_bt128/'
log_dl_pth = 'CIFAR100/discrimloss_results/weights_n40_bt128_new/'
log_dl_es_pth = 'CIFAR100/discrimloss_results/weights_n40_bt128_es/'
log_dl_ea_pth = 'CIFAR100/discrimloss_results/weights_n40_bt128_ea/'
log_dl_ea_2pth = 'CIFAR100/discrimloss_results/weights_n40_bt128_cn1_ea_tanh_wd6_static_min0.2/'
loss_pth = 'loss_each_sample_one_eps.txt'
sigma_pth = 'sigma_each_sample_one_eps.txt'
results_pth = 'Wide_ResNet.csv'

def method_name(log_pth):
    if 'baseline' in log_pth.split('/')[1]:
        method = 'baseline'
    elif 'superloss' in log_pth.split('/')[1]:
        method = 'superloss'
    else:
        if 'es' in log_pth.split('/')[-2]:#.split('_')[-1]
            method = 'our_es'
        elif 'ea_2' in log_pth.split('/')[-2]:#.split('_')[-1]:
            method = 'our_ea_2'
        elif 'ea' in log_pth.split('/')[-2]:
            method = 'our_ea'
        else:
            method = 'our_base'
    return method

def loss_histogram(log_pth, loss_pth):
    '''
    baseline at the 10th epoch, loss histogram for each sample containing both clean and noisy data
    '''
    with open(os.path.join(CURRENT_PATH, log_pth, loss_pth), 'r') as fl:
        method = method_name(log_pth)

        rand_fraction = int(log_pth.split('/')[-2].split('_')[1][1:]) / 100.0
        for line in fl.readlines():
            epoch, = eval(line)
            if epoch in ['epoch:0','epoch:1','epoch:2','epoch:3','epoch:10','epoch:20','epoch:30','epoch:40','epoch:50','epoch:60','epoch:70','epoch:80','epoch:90','epoch:100','epoch:110', 'epoch:119']:
                loss, = eval(line).values()
                corrupt_nums = int(len(loss)*rand_fraction)
                loss = np.array(loss)
                #loss = np.exp(loss)
                loss = (loss - loss.min()) / (loss.max() - loss.min())
                corrupt_loss = loss[:corrupt_nums]
                clean_loss = loss[corrupt_nums:]
                #color = ['red','green']
                plt.title('loss statistic at {} of {} method'.format(epoch, method))
                #[clean_loss, corrupt_loss]
                plt.hist([clean_loss, corrupt_loss], bins=80, edgecolor='g',density=True, histtype='bar', stacked=True, alpha = 0.8)
                plt.legend(loc='upper right')
                plt.xlabel('loss')
                plt.ylabel('hist density')
                plt.savefig(os.path.join('/users6/ttwu/script/GoalstepRE/ml-data-parameters-master/myplot/', 'ea_2_{}'.format(epoch)))
                plt.show()

def quantile_vars(log_pth, var_pth):
    '''
    clean/noisy samples vars(loss/sigma) quantile: 25%, median, 75%
    '''
    clean_25_lst = []
    clean_50_lst = []
    clean_75_lst = []
    corrupt_25_lst = []
    corrupt_50_lst = []
    corrupt_75_lst = []
    with open(os.path.join(CURRENT_PATH, log_pth, var_pth), 'r') as fl:
        for line in fl.readlines():
            #epoch, = eval(line)
            var, = eval(line).values()
            corrupt_nums = int(len(var)*rand_fraction)
            var = np.array(var)
            var = np.exp(var)
            #loss = (loss - loss.min()) / (loss.max() - loss.min())
            corrupt_var = Series(var[:corrupt_nums])
            clean_var = Series(var[corrupt_nums:])
            clean_25 = clean_var.describe()['25%']
            clean_50 = clean_var.describe()['50%']
            clean_75 = clean_var.describe()['75%']
            corrupt_25 = corrupt_var.describe()['25%']
            corrupt_50 = corrupt_var.describe()['50%']
            corrupt_75 = corrupt_var.describe()['75%']
            clean_25_lst.append(clean_25)
            clean_50_lst.append(clean_50)
            clean_75_lst.append(clean_75)
            corrupt_25_lst.append(corrupt_25)
            corrupt_50_lst.append(corrupt_50)
            corrupt_75_lst.append(corrupt_75)
    x_axis = [i for i in range(120)]
    plt.title('clean/noisy {} distribution of discrimloss_es results(40% noise)'.format(var_pth.split('_')[0]))
    plt.plot(x_axis, corrupt_50_lst, 'y-', label='noisy {}'.format(var_pth.split('_')[0]))
    plt.plot(x_axis, clean_50_lst, 'g-', label='clean {}'.format(var_pth.split('_')[0]))
    plt.fill_between(x_axis, clean_75_lst, clean_25_lst, alpha=0.25, color='g')
    plt.fill_between(x_axis, corrupt_75_lst, corrupt_25_lst, alpha=0.25, color='y')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel(var_pth.split('_')[0])
    plt.show()

def single_sample_vars(log_pth, var_pth):
    '''
    sigma/loss for single sample
    '''
    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']#
    pk_nums = 1#times for random picking
    total_lst = []

    method = method_name(loss_pth)
    rand_fraction = int(log_pth.split('/')[-2].split('_')[1][1:])/100.0
    with open(os.path.join(CURRENT_PATH, log_pth, var_pth), 'r') as fl:
        for line in fl.readlines():
            # epoch, = eval(line)
            var, = eval(line).values()
            corrupt_nums = int(len(var) * rand_fraction)
            sample_lst = [i for i in range(len(var))]
            corrupt_lst = sample_lst[:corrupt_nums]
            clean_lst = sample_lst[corrupt_nums:]
            total_lst.append(var)
        df = DataFrame(total_lst)
        #df = DataFrame(np.exp(df.values))
        for i in range(10):
            corrupt_index = np.random.choice(corrupt_lst, pk_nums)
            clean_index = np.random.choice(clean_lst, pk_nums)
            x_axis = [i for i in range(120)]
            for i, ci in enumerate(corrupt_index):
                plt.plot(x_axis, df.iloc[:, ci].tolist(), color=color[i], label='noisy sample {}'.format(ci))
            for i, ci in enumerate(clean_index):
                plt.plot(x_axis, df.iloc[:, ci].tolist(), color=color[i+pk_nums], label='clean sample {}'.format(ci))

            plt.title('{} results({}% noise)'.format(method, rand_fraction*100))
            plt.legend(loc='upper right')
            plt.xlabel('epoch')
            plt.ylabel(var_pth.split('_')[0])
            plt.show()

def samples_var(pth_lst, var_pth):
    '''
    sigma/loss for single sample in different loss_function
    '''
    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'peru','brown','chartreuse','fuchsia','darkorange','salmon']#
    pk_nums = 1#times for random picking

    for j in range(len(pth_lst)):
        total_lst = []

        method = method_name(pth_lst[j])
        rand_fraction = int(pth_lst[j].split('/')[-2].split('_')[1][1:]) / 100.0
        with open(os.path.join(CURRENT_PATH, pth_lst[j], var_pth), 'r') as fl:
            for line in fl.readlines():
                # epoch, = eval(line)
                var, = eval(line).values()
                corrupt_nums = int(len(var) * rand_fraction)
                sample_lst = [i for i in range(len(var))]
                corrupt_lst = sample_lst[:corrupt_nums]
                clean_lst = sample_lst[corrupt_nums:]
                total_lst.append(var)
            df = DataFrame(total_lst)
            #df = DataFrame(np.exp(df.values))
            np.random.seed(seed)
            corrupt_index = np.random.choice(corrupt_lst, pk_nums)
            clean_index = np.random.choice(clean_lst, pk_nums)
            x_axis = [i for i in range(10)]#120
            for i, ci in enumerate(corrupt_index):
                plt.plot(x_axis, df.iloc[:10, ci].tolist(), color=color[i+2*j*pk_nums], label='noisy sample {}/{}'.format(ci, method))
            for i, ci in enumerate(clean_index):
                plt.plot(x_axis, df.iloc[:10, ci].tolist(), color=color[i+pk_nums+2*j*pk_nums], label='clean sample {}/{}'.format(ci, method))

    plt.title('sample {} of discrimloss results(40% noise)'.format(var_pth.split('_')[0]))
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel(var_pth.split('_')[0])
    plt.show()

def train_val_loss(pth_lst):
    '''
        sigma/loss for single sample in different loss_function
        '''
    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']
    fig, ([[axs1,axs2],[axs3,axs4]]) = plt.subplots(2, 2)
    x_axis = [i for i in range(120)]  # 120
    for j in range(len(pth_lst)):
        method = method_name(pth_lst[j])

        df = pd.read_csv(os.path.join(CURRENT_PATH, pth_lst[j], results_pth))
        axs1 = plt.subplot(221)
        axs1.plot(x_axis, df['train loss'][:120].tolist(), color=colors[j], label='{}'.format(method))
        axs2 = plt.subplot(222)
        axs2.plot(x_axis, df['train acc'][:120].tolist(), color=colors[j], label='{}'.format(method))
        axs3 = plt.subplot(223)
        axs3.plot(x_axis, df['val loss'][:120].tolist(), color=colors[j], label='{}'.format(method))
        axs4 = plt.subplot(224)
        axs4.plot(x_axis, df['val acc'][:120].tolist(), color=colors[j], label='{}'.format(method))

    #plt.title('sample {} of discrimloss results(40% noise)'.format(var_pth.split('_')[0]))
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(lines_labels[0])]

    # finally we invoke the legend (that you probably would like to customize...)

    #plt.legend(lines, labels,bbox_to_anchor=(0.01, 0.01),ncol=5)#
    # axs1.legend_.remove()
    # axs2.legend_.remove()
    # axs3.legend_.remove()
    plt.suptitle('40% noise')
    axs4.legend(loc=3, bbox_to_anchor=(1.0,0.2),borderaxespad = 0.5)
    axs1.set_xlabel('epoch')
    axs1.grid(True)
    axs2.set_xlabel('epoch')
    axs2.grid(True)
    axs3.set_xlabel('epoch')
    axs3.grid(True)
    axs4.set_xlabel('epoch')
    axs4.grid(True)
    axs1.set_ylabel('train loss')
    axs2.set_ylabel('train acc')
    axs3.set_ylabel('val loss')
    axs4.set_ylabel('val acc')
    #plt.ylabel(var_pth.split('_')[0])
    plt.show()

def load_CIFAR100():
    """ load single batch of cifar """
    with open(os.path.join(CIFAR_100_PATH, 'cifar-100-python/train'), 'rb')as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        X = X.reshape(50000, 3, 32, 32)
        Y = np.array(Y)
        print(Y.shape)
        return X, Y

def save_CIFAR100_imgs():
    imgX, imgY = load_CIFAR100()
    with open(os.path.join(CIFAR_100_PATH, 'img_label.txt'), 'w') as f:
        for i in range(imgY.shape[0]):
            f.write('img' + str(i) + ' ' + str(imgY[i]) + '\n')

    for i in range(imgX.shape[0]):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = "img" + str(i) + ".png"
        img.save(CIFAR_100_PATH +"/cifar-100-pics/" + name, "png")
    print("save successfully!")

def show_cifar_img(index):

    lena = mpimg.imread(os.path.join(CIFAR_100_PATH, "cifar-100-pics", 'img{}.png'.format(index)))  # 读取和代码处于同一目录下的 lena.png

    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def clean_hard_noisy_samle(log_pth1, log_pth2):
    '''show confidence changes of sample with different hardness with ea and ea_2 loss'''
    #pick 3 samples with different hardness
    method = method_name(log_pth1)
    with open(os.path.join(CURRENT_PATH, log_pth1, loss_pth), 'r') as fl:
        rand_fraction = int(log_pth.split('/')[-2].split('_')[1][1:]) / 100.0
        lines = fl.readlines()
        line = lines[-1]
        var, = eval(line).values()
        corrupt_nums = int(len(var) * rand_fraction)
        #sample_lst = [i for i in range(len(var))]
        corrupt_lst = var[:corrupt_nums]
        clean_lst = var[corrupt_nums:]
        noisy_ind = sorted(range(len(corrupt_lst)), key=lambda k: corrupt_lst[k], reverse=True)
        clean_ind = sorted(range(len(var)-corrupt_nums), key=lambda k: clean_lst[k], reverse=True)
        clean_ind = [i+corrupt_nums for i in clean_ind]
        # print(noisy_ind[0:3])#bigger first
        # print(clean_ind[0:3])
        # print(clean_ind[-3:])

    total_lst = []
    with open(os.path.join(CURRENT_PATH, log_pth1, sigma_pth), 'r') as fs:
        for line in fs.readlines():
            # epoch, = eval(line)
            var, = eval(line).values()
            total_lst.append(var)
        df = DataFrame(total_lst)
        df = DataFrame(np.exp(df.values))
        x_axis = [i for i in range(120)]
        plt.plot(x_axis, df.iloc[:,noisy_ind[0]], color='#1f77b4', label='noisy sample {}/{}'.format(noisy_ind[3], method))
        plt.plot(x_axis, df.iloc[:,clean_ind[0]], color='#ff7f0e', label='hard sample {}/{}'.format(clean_ind[1], method))
        plt.plot(x_axis, df.iloc[:,clean_ind[-1]], color='#2ca02c', label='easy sample {}/{}'.format(clean_ind[-10], method))

    plt.title('sample results(40% noise)')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('sigma')
    plt.show()

def switch_time():
    #determine the time to switch from phase1 to phase2
    pass




if __name__ == "__main__":
    #pth_lst = [log_sl_pth]
    #train_val_loss(pth_lst)
    #loss_histogram(log_dl_ea_2pth, loss_pth)
    clean_hard_noisy_samle(log_dl_ea_2pth, log_dl_ea_pth)
    #quantile_vars(log_dl_pth, sigma_pth)
    # pth_lst = [log_dl_pth, log_dl_es_pth]#, log_dl_ea_pth]
    #samples_var(pth_lst, sigma_pth)
    #single_sample_vars(log_sl_pth, loss_pth)
    #save_CIFAR100_imgs()
    #show_cifar_img(15110)