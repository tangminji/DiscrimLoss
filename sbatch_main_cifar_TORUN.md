1. 运行时间（1 次完整训练，总时间为 MAX_EVALS*per_time ）
    - cifar10: 7h6min
    - cifar100: 7h45min
2. 设定MAX_EVALS
3. 修改脚本的 -t -w -gpu
4. 确认space，最好精确一点
5. run 脚本生成器
6. 执行 sbatch_main_cifar%s_hy_box%d.sh
