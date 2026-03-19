T2MAC: Targeted and Trusted Multi-Agent Communication through Selective Engagement and Evidence-Driven Integration

[Paper Link](https://arxiv.org/pdf/2505.07207) -- **AAAI2024**
[Code Link](https://github.com/ZangZehua/T2MAC)

### 预备知识-1: 迪利克雷分布

- 普通的筛子 (多项分布)
> 筛子是均匀的，筛到1到6点数的概率都是$\frac{1}{6}$

$$p=[\frac{1}{6}, ..., \frac{1}{6}]$$

- 迪利克雷分布
> 我们不知道筛子的特性，每个面的概率不确定，因此我们需要用迪利克雷分布来确定这个筛子的概率分布。即**分布的分布**。

#### 数学定义与表达式
**1.基本定义**
迪利克雷分布的概率密度函数为:
$$p(\theta|\alpha)=\frac{\Gamma (\textstyle \sum_{i=1}^{K}\alpha_i)}{\textstyle \prod_{i=1}^{K}\Gamma(\alpha_i)}\prod_{i=1}^{K}\theta_i^{\alpha_i-1}$$
其中:
- $\theta=(\theta_1, ..., \theta_K)$是随机变量, 满足$\textstyle \sum_{i=1}^{K}\theta_i=1$, 且$\theta_i \ge 0$ -> **待确定的分布**
- $\alpha=(\alpha_1, ..., \alpha_K)$是分布参数, $\alpha > 0$ -> **控制分布形态的浓度参数, $\alpha$控制着$\theta$的分布形态**
- $\Gamma$是Gamma函数，定义为$\Gamma(s)= \int_{0}^{\infty } x^{(s-1)} e^{-x}dx $

> 一般操作流程为: 通过数据来估计分布参数$\alpha$ -> 通过$\alpha$计算后验分布或通过贝叶斯推断等方法得到$\theta$的分布。

----

### 预备知识2: 主观逻辑 (Subjective Logic, SL)

主观逻辑的核心是**信任三元组**, 表示为:
$$\omega=(b,d,u,a)$$
其中:
- b: 信任度，相信命题为真的程度
- d: 不信任度, 相信命题为假的程度
- u: 不确定度, 对命题真假不确定的程度
- a: 基础率, 先验概率或默认概率

这些分量满足约束条件: $b+d+u=1$。

**概率期望计算**

$$E=b+a·u$$

**SL与迪利克雷分布的关联**

映射关系:
- 主观意见$(b_1, ..., b_k, u)$可转换为迪利克雷参数$\alpha$
- 总证据量$r=\frac{K}{u}-K$
- 各类别证据数$r_i=\frac{b_i·K}{u}$
- 迪利克雷参数$\alpha_i=r_i+1$

----

### 问题背景

- **现有问题:** 传统的多智能体通信采用广播机制，导致信息冗余且缺乏实用性，不仅占用资源，还可能降低通信效率。此外，现有方法通常使用基础的聚合机制或将其视为黑河，难以有效处理决策中的不确定性。

### 方法论

#### 证据理论

- 智能体决策的不确定性
智能体仅能观测到全局的**部分信息**，因此需要量化这种决策的不确定性

- 证据
从智能体的观测数据中提炼出的、能 “支撑决策” 的关键信息。不再传递**原始信息**，而是传递有决策价值的**证据**。

基于上述定义, 主观逻辑体系 (SL)为每个动作分配一个信念质量，且为整个决策场景分配一个全局的不确定性:

$$u_i+\sum_{i=1}^{K}b_i^k=1$$

其中 $u_i \ge 0$，表示智能体$i$对全局的不确定程度，$b_i^k \ge 0$表示智能体$i$对第$k$个动作是最优动作的确定程度。

**构建迪利克雷分布**

根据主观逻辑与迪利克雷分布的关系，每个动作的信念质量依赖于参数$\alpha$:
$$b_i^k=\frac{e_i^k}{S_i}=\frac{\alpha_i^k-1}{S_i} \quad and \quad u_i=\frac{K}{S_i}$$

其中$S_i=\textstyle \sum_{k=1}^{K}(e_i^k+1)=\textstyle \sum_{k=1}^{K}(\alpha_i^k)$为迪利克雷分布参数, $K$为智能体的动作数量, $e_i^k$为智能体$i$选择第$k$个动作的信念。

**具体实现**

- 本地证据 (仅用于智能体$i$的策略生成)
用一个MLP+RNN后，隐藏层经过一个编码器得到本地策略$\pi=f_{\text{local}}(h_i)$
- 定制化证据 (智能体$i$为智能体$j$发送的定制化消息)
智能体$i$在上述得到的隐藏层情况下经过一个编码器得到$e_{ij}=f_{ij}(h_i)$

#### 证据驱动下的消息聚合

假设智能体$i$的证据消息为$M_i=\{ \{b_i^k\}_{k=1}^K, u_i^k \}$, 同理智能体$j$的证据消息为 $M_j$, 那么二者聚合方式为:

$$M=M_i \oplus M_j$$

其中$\oplus$为DST融合算子。计算方式为:

$$b^k=\frac{1}{1-C}(b_i^kb_j^k+b_i^ku_j+b_j^ku_i), \quad u=\frac{1}{1-C}u_iu_j$$

其中，$C$为冲突度，其定义为$C=\textstyle \sum_{k \ne k'}b_i^k b_j^{k'}$。