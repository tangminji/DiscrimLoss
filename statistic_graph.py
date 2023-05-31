import os
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl

color = ['b','g','r','c','m','y','k','w']
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
          '#1a55FF',
          '#ae3400']

CURRENT_PATH=os.getcwd()
def var_sigma_diff_init():
    with open(os.path.join(CURRENT_PATH, 'var_sigmas.txt'), 'r') as f:
        count = 0
        for line in f.readlines():
            try:
                line = eval(line)
                x_range = [i for i in range(len(line))]
                plt.plot(x_range, line, '{}o-'.format(color[count]), label='lam={}')
                count += 1
            except:
                continue
        plt.show()

#var_sigma_diff_init()




def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

print(0.135*tanh(0.3632*9-7.027))
def paint_tanh():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    x = np.linspace(-10, 10)
    y = tanh(x)


    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xticks([i for i in range(-50,50,5)])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_yticks(np.arange(1,10,0.25).tolist())

    #guarantee min=1, max=x, y=a*tan(kx)+b, we have b=a+1
    A = np.arange(0.50,1.001, 0.50).tolist()
    P = [0.2,1.,5.]
    Q = [0,15]
    N= len(A)*len(P)*len(Q)
    cnt=0
    #cmap = plt.cm.get_cmap("hsv", N+1)#generate any number colors, but colors are hard to tell
    for a in A:
        for p in P:
            for q in Q:
                plt.plot(p*x+q, y*a+a+1, label="{}*x+{}, {}*tanh+{}".format(p,q,a,a+1), color=colors[cnt])
                cnt += 1
    plt.legend(loc=2)
    plt.show()

#paint_tanh()