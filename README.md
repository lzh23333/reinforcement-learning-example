# 强化学习——猫捉老鼠

## 林仲航	2020210863

## 1. 问题描述与建模

不妨假设环境为$N\times M$的二维网格，其中猫位于网格左上角，即$(0,0)$处，老鼠位于网格右下角，即$(N-1, M-1)$处，将位置$(x,y)$统一编码为$p=N \times x + y$，故而该问题下的状态可以写为二维向量$s=(p_{cat},p_{mouse})$。此外还有一系列障碍物$[p_i,...]$。针对该问题，猫得到的最佳结果为抓到老鼠，即$p_{cat}=p_{mouse}$。可以定义如下的奖励函数：
$$
reward(p_{cat},p_{mouse})=\begin{cases}
10, & p_{cat}=p_{mouse}\\
-10, & p_{cat}=p_i\\
-1 & others
\end{cases}
$$
此外，猫可以采用动作只有上下左右四种，故而可以定义状态-动作值函数$Q(s,a)_{MN\times MN \times4}$，代表在状态$s$下采用动作$a$的未来总收益。当有了最优的$Q$函数后，即可通过贪心策略实现猫寻路的最优解。

## 2. 算法描述

采用Q-Learning算法来对$Q$函数求最优。实验首先将$Q$初始化为0值，随后进行多次循环，每次循环以猫抓到老鼠或者达到障碍物为停止条件。在单次循环中，猫的决策方法为$\epsilon$-greedy方法，具体为：以$1-\epsilon$的概率选取$action=argmax_a Q(s,a)$，以$\epsilon$的概率随机选取其它值$action=random({a|a \neq argmax_a Q(s, a)})$。随后猫采取行动$action$，观察下一个状态$s'$与得到的reward，利用以下公式进行更新：
$$
Q(s,a)\leftarrow Q(s,a)+\alpha[R+\gamma max_aQ(s',a)-Q(s,a)]
$$
多次循环后$Q$趋于收敛。

## 3. 实验细节

### 3.1 简单情况

网格大小为$(4,4)$，老鼠位于$(3,3)$的位置，猫位于$(0,0)$的位置，且老鼠保持静止不动。障碍物坐标为$[(1,2),(2,1)]$。此时其实可以将状态简化为$s=p_{cat}$，但是为了编程上的可复用性，依旧用$s=(p_{cat},p_{mouse})$来进行描述。可以输入命令：

```
python example.py --mouse_pattern stay
```

来运行Q-Learning算法并对结果进行可视化：

![example0-curve](imgs/example0-curve.png)

如图，可以看出算法在约100轮的时候基本收敛，可视化结果如下：

<img src="./imgs/example0.gif" width = "402" height = "443" alt="图片名称" align=center />

可以看出猫可以找出最优路径，证明了强化学习在该问题上的有效性。

### 3.2 老鼠加入移动策略

假设老鼠按照随机策略运动，即老鼠会随机从上下左右四个方向中选取一个移动方向，若移动超过网格范围或者会移动到障碍物上，则保持不动。由于随机性，有时会出现老鼠向猫的位置移动的情况，为了简便，统一认为只有老鼠与猫的位置重合的情况下，才认为猫抓到老鼠。

除此之外，由于随机的移动策略过于简单，不妨给老鼠设定更为实际的运动策略，即在每一轮迭代中，老鼠会以0.5的概率尽可能远离猫的位置，以0.5的概率保持不动。以下给出两种策略的运行命令以及效果：

```
python example.py --mouse_pattern random
python example.py --mouse_pattern away
```

<center class="2 gif">
    <img src="./imgs/example1.gif" width = "402" height = "443" alt="图片名称" align=left />
</center>

