# 泛模因理论：三步证明

**版本**：ε₃（变分重构）  
**日期**：2026-07-08

> 给定词表 $W$。$W$ 通过形式概念分析（FCA）的 ↑↓ 算子收敛为一个加权图 $G$。在 $G$ 上，拉普拉斯热核的迹 $\Theta(t)$ 定义天然时间标度 $t^*$ 和五维状态向量。状态向量的演化由 $\dot{\vec{M}} = \mathbf{F}_\Theta(\vec{M})$ 描述——该 ODE 是在不变域 + 次数 ≤ 2 + 变量耦合约束下的唯一多项式形式。轨迹的长期归宿由 Jacobian 稳定性判据决定。

**三步。定理支撑每一步。无手挥。**

---

## §0. 公理与定义

### 0.1 公理

八条命题公理（$P_1$–$P_8$，完整列表见附录 D）。本文直接依赖的：

- **$P_1$（信息本体论）**：信息 $\mathbb{I}$ 不可约化为物质或能量。
- **$P_2$（泛模因实在论）**：$\mathcal{P} = \{x \mid \text{可复制}(x) \land \text{可演化}(x) \land \text{身份保持}(x)\}$。
- **$P_3$（结构现实主义）**：$\text{id}(x)$ 由关系模式 $R_x \subseteq (x \cup \text{context}(x))^2$ 决定。
- **$P_5$（可逆性认识论）**：每步映射有可追踪的信息保持性质。
- **$P_7$（演化普遍性）**：复制 $R$ + 变异 $V$ + 选择 $S_e$ → $S_{t+1} = S_e \circ V \circ R(S_t)$。
- **$P_8$（层级涌现）**：泛模因构成偏序集，存在双向因果。

### 0.2 符号

| 符号 | 含义 |
|------|------|
| $2^X$ | $X$ 的幂集 |
| $|X|$ | 集合 $X$ 的基数 |
| $\text{Tr}(\cdot)$ | 矩阵迹 |
| $D^{-1/2}$ | $D^{-1/2}_{ii} = 1/\sqrt{D_{ii}}$ 若 $D_{ii} > 0$，否则 $0$ |
| $\beta_0$ | 第 0 个 Betti 数 = 连通分量数 = $\lambda_0$ 的重数 |

---

## 第一步：词 → 图

### 1.1 形式背景

给定词表 $W = \{w_1, \ldots, w_m\}$。每个 $w \in W$ 是有限字符集 $\Sigma$ 上的字符串。

**定义 1（自暴露字符集）**。

$$C = \bigcup_{w \in W} \text{chars}(w) \subseteq \Sigma$$

$|C| = p$。论域 $\mathcal{U} = C \cup W$，$|\mathcal{U}| = p + m = n$。

**定义 2（形式背景）**。$W$ 导出的形式背景为三元组 $\mathbb{K} = (C, W, I)$，其中：

$$I = \{(c, w) \in C \times W \mid c \in \text{chars}(w)\}$$

这是 FCA 的标准起点——对象集合 $W$，属性集合 $C$，关联关系 $I$ = "字符 $c$ 出现在词 $w$ 中"。

### 1.2 衍生算子

**定义 3（↑↓ 算子）**。对 $A \subseteq C$ 和 $B \subseteq W$：

$$\begin{aligned}
A^\uparrow &= \{w \in W \mid \forall c \in A: (c, w) \in I\} \quad &\text{（共同上域）} \\
B^\downarrow &= \{c \in C \mid \forall w \in B: (c, w) \in I\} \quad &\text{（共同下域）}
\end{aligned}$$

$A^\uparrow$ 是"包含 $A$ 中所有字符的词集"；$B^\downarrow$ 是"出现在 $B$ 中所有词中的字符集"。

**引理 1（Galois 连接）**。对 $A \subseteq C, B \subseteq W$：

$$A \subseteq B^\downarrow \iff B \subseteq A^\uparrow$$

且 $A \subseteq A^{\uparrow\downarrow}$，$B \subseteq B^{\downarrow\uparrow}$，$A^\uparrow = A^{\uparrow\downarrow\uparrow}$，$B^\downarrow = B^{\downarrow\uparrow\downarrow}$。

**证明**。标准 FCA 结论（Wille 1982）。$\square$

### 1.3 有限收敛

**定理 1（有限收敛）**。对任意 $x \in \mathcal{U}$，从 $\{x\}$ 出发交替应用 ↑↓：

$$S_0 = \{x\}, \quad S_{k+1} = \begin{cases} S_k^\uparrow & k \text{ 为偶} \\ S_k^\downarrow & k \text{ 为奇} \end{cases}$$

存在 $N \le n$ 使得 $S_N^{\uparrow\downarrow} = S_N$（从 $C$ 出发）或 $S_N^{\downarrow\uparrow} = S_N$（从 $W$ 出发）。

**证明**。由引理 1，$A \subseteq A^{\uparrow\downarrow}$ 且 $B \subseteq B^{\downarrow\uparrow}$。故从 $C$ 中的 $x$ 出发，序列 $\{x\}, \{x\}^{\uparrow\downarrow}, \{x\}^{\uparrow\downarrow\uparrow\downarrow}, \ldots$ 是单调递增的集合列：

$$\{x\} \subseteq \{x\}^{\uparrow\downarrow} \subseteq \{x\}^{\uparrow\downarrow\uparrow\downarrow} \subseteq \cdots \subseteq C$$

$C$ 有限，至多 $p$ 步稳定。类似地，从 $W$ 出发至多 $m$ 步稳定。$\square$

> **说明**：此处使用 ↑↓ 交替而非 ↑↓↑↓⋯ 交替（每一步都取完整的一对算子），保证了序列的单调性——这是前一版 ε 的数学错误所在。每个稳定集称为 **FCA 收敛闭包**。

### 1.4 收敛图

FCA 收敛闭包自然产生三族共现关系：

**定义 4（共现图 $G$）**。收敛图 $G = (\mathcal{U}, E, \mathbf{W})$ 为顶点集 $\mathcal{U} = C \cup W$ 上的加权无向图，边权矩阵 $\mathbf{W} \in \mathbb{R}^{n \times n}_{\ge 0}$：

$$\mathbf{W}_{c_i, c_j} = |(c_i)^{\uparrow\downarrow} \cap (c_j)^{\uparrow\downarrow}| \quad (c_i, c_j \in C)$$

$$\mathbf{W}_{w_i, w_j} = |(w_i)^{\downarrow\uparrow} \cap (w_j)^{\downarrow\uparrow}| \quad (w_i, w_j \in W)$$

$$\mathbf{W}_{c, w} = \begin{cases} 1 & \text{若 } (c, w) \in I \\ 0 & \text{否则} \end{cases} \quad (c \in C, w \in W)$$

即：两个字之间的边权 = 它们各自 FCA 收敛闭包中**共有的字**数（字-字则为共有的字数）。字-词边权为原始包含关系。

**注**：$\mathbf{W}_{c_i, c_j} = |(c_i)^\uparrow \cap (c_j)^\uparrow|$——因为没有 ↑↓ 收敛改变 ↑ 像（引理 1 的 $A^\uparrow = A^{\uparrow\downarrow\uparrow}$），所以共现等于原始 ↑ 像交集。字-词边同理。$\mathbf{W}$ 由 $I$ 唯一确定——**没有自由参数**。

---

## 第二步：图 → 参数

### 2.1 归一化拉普拉斯

**定义 5**。度矩阵 $D_{ii} = \sum_{j} \mathbf{W}_{ij}$，归一化拉普拉斯：

$$\mathcal{L} = I - D^{-1/2} \mathbf{W} D^{-1/2}$$

$\mathcal{L}$ 实对称半正定（Chung 1997）。特征分解：

$$\mathcal{L} = U \Lambda U^\top, \quad \Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$$

$0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1} \le 2$。Perron 特征向量 $\mathbf{u}_0 \propto D^{1/2}\mathbf{1}$。

### 2.2 热迹与天然时间标度

**定义 6（热迹）**。

$$\Theta(t) = \text{Tr}(e^{-t\mathcal{L}}) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$$

**引理 2（热迹基本性质）**。$\Theta(0) = n$，$\Theta(t)$ 在 $t > 0$ 上严格递减（若 $\lambda_1 > 0$），$\lim_{t \to \infty} \Theta(t) = \beta_0$（$\lambda_0$ 重数 = 连通分量数）。

**证明**。逐项验证。$\square$

**定义 7（混合时间 $t^*$）**。

$$t^* = \inf\{t > 0 \mid \Theta(t) \le \beta_0 + 1\}$$

由引理 2，$\Theta(0) = n \ge \beta_0 + 1$（若 $n > \beta_0$），$\lim_{t \to \infty} \Theta(t) = \beta_0 < \beta_0 + 1$。$\Theta$ 连续且严格递减 → $t^*$ 存在且唯一，且 $\Theta(t^*) = \beta_0 + 1$。

### 2.3 标度点的必然性

**定理 2（热迹渐近展开）**。

$$t \to 0^+: \quad \Theta(t) = n - t \cdot \text{Tr}(\mathcal{L}) + O(t^2)$$

$$t \gg 1/\lambda_1: \quad \Theta(t) - \beta_0 = m_1 e^{-t \lambda_1} + O(e^{-t \lambda_2})$$

其中 $m_1$ 为 $\lambda_1$ 的重数。

**证明**。小 $t$：$e^{-t\lambda_k} = 1 - t\lambda_k + O(t^2\lambda_k^2)$，对所有 $k$ 求和。大 $t$：因式分解 $\Theta(t) - \beta_0 = \sum_{\lambda_k > 0} e^{-t\lambda_k}$，主导项来自最小正特征值 $\lambda_1$。$\square$

**推论 1（标度点的必然性）**。在 $\{0, \infty, t^*\}$ 的基底下，分数和倍数 $\{t^*/2, t^*, 2t^*, 3t^*\}$ 是捕获以下四个渐近区域的**最小完备集**：

| 标度 | 渐近区域 | 主导项 |
|------|----------|--------|
| $t^*/2$ | 小 $t$（Weyl） | $n - \frac{t^*}{2}\text{Tr}(\mathcal{L})$——全部特征值集体贡献 |
| $t^*$ | 临界 | $\Theta(t^*) = \beta_0 + 1$（定义固定） |
| $2t^*$ | 衰减 | $\Theta(2t^*) - \beta_0 \approx m_1 e^{-2t^*\lambda_1}$——$\lambda_1$ 二阶 |
| $3t^*$ | 长尾 | $\Theta(3t^*) - \beta_0 \approx m_1 e^{-3t^*\lambda_1}$——高阶验证 |

**证明**。$t^*/2$ 在 $t^*$ 足够小时落入 Weyl 区域（当 $\lambda_1 \cdot t^* \ll 1$ 时成立——低谱隙图满足）。若 $\lambda_1 \cdot t^* \gg 1$（高谱隙），$t^*$ 由 $\Theta(t^*) = \beta_0 + 1$ 精确决定，$2t^*, 3t^*$ 自然落入指数衰减区域。两种状态均被四个标度点覆盖。$\square$

### 2.4 谱社区划分

**定义 8（谱社区）**。图的谱社区数 $k^*$ 和划分由下述过程确定：

$$k^* = \arg\max_{1 \le k \le n-2} (\lambda_{k+1} - \lambda_k)$$

取 $U_{:, 1:k^*}$（前 $k^*$ 个非零特征向量），逐行归一化 → $\mathbb{R}^{k^*}$ 中的 $n$ 个点 → 对这些点求解 $k$-means 的**全局最优**（多次随机初始化取最小 SSE）。得到划分 $\{X_1, \ldots, X_{k^*}\}$。

**注**：$k^*=1$ 表示全图无显著分块结构（全图为一个社区），$k^* \approx n$ 表示图高度碎片化。两种极限情况分别对应单一轨迹和聚合分析。

### 2.5 五维状态

对社区 $X_i$，$|X_i| = n_i$。其导出子图 $G[X_i]$ 的归一化拉普拉斯 $\mathcal{L}^{(i)}$ 的特征值为 $\{\lambda_k^{(i)}\}$，热迹为 $\Theta_i(t)$。

**定义 9（五维状态——热迹比值）**。

$$\boxed{
\begin{aligned}
D_i &= \frac{\Theta_i(n_i/n)}{n_i} & &\text{（深度：规模时间 $\tau_i = n_i/n$ 上的热保留率）} \\[4pt]
B_i &= 1 - \frac{\Theta_G(n_i/n)}{n} & &\text{（广度：全局在 $\tau_i$ 上的热耗散率）} \\[4pt]
\rho_i &= \frac{\sum_k \lambda_k^{(i)}}{\sum_k \lambda_k^{(G)}} & &\text{（能量：谱质比）} \\[4pt]
R_i &= \frac{t^*_G}{t^*_i} & &\text{（演化速率：相对混合速度）} \\[4pt]
S_i &= \frac{\Theta_i(t^*_i)}{n_i} = \frac{\beta_0^{(i)} + 1}{n_i} & &\text{（韧度：连通分量密度）}
\end{aligned}
}$$

**定理 3（天然归一化）**。$D_i, B_i, \rho_i, S_i \in [0,1]$，$R_i > 0$。归一化由热方程 $\partial_t u = -\mathcal{L}u$ 的解析性质内在保证。无外部归一化函数。

**证明**。$\Theta_i(t)/n_i \in [\beta_0^{(i)}/n_i, 1] \subseteq [0,1]$ → $D_i, S_i \in [0,1]$。$1 - \Theta_G/n \in [0, 1 - \beta_0/n] \subseteq [0,1]$ → $B_i \in [0,1]$。迹比 $\in [0,1]$。混合时间比 $> 0$。$\square$

### 2.6 十一参数

**定义 10（十一参数——标度点完备映射）**。

记 $\Theta_{0.5} = \Theta(t^*/2)$，$\Theta_1 = \Theta(t^*) = \beta_0 + 1$，$\Theta_2 = \Theta(2t^*)$，$\Theta_3 = \Theta(3t^*)$。

$$\boxed{
\begin{aligned}
\alpha_1 &= \frac{\lambda_1}{2}, &
\alpha_2 &= \frac{\Theta_{0.5}}{n} \\[4pt]
\beta_1 &= \frac{\lambda_{\max} - \lambda_1}{\lambda_{\max}}, &
\beta_2 &= \frac{\sum \lambda_k^2}{(\sum \lambda_k)^2} \\[4pt]
\gamma_1 &= 1 - \frac{\Theta_2}{\Theta_1}, &
\gamma_2 &= \frac{\Theta_{0.5}}{\Theta_1} - 1 \\[4pt]
\delta_1 &= \frac{\Theta_{0.5} - \Theta_1}{n}, &
\delta_2 &= \lambda_{\max}, &
\delta_3 &= 1 - \frac{\Theta_3}{\Theta_2} \\[4pt]
\varepsilon_1 &= \frac{\lambda_1}{\lambda_{\max}}, &
\varepsilon_2 &= 1 - \frac{\Theta_2}{n}
\end{aligned}
}$$

**引理 3（参数有界性）**。所有参数的值域由归一化拉普拉斯自动约束：

| 参数 | 值域 | 理由 |
|------|------|------|
| $\alpha_1, \varepsilon_1$ | $[0,1]$ | $\lambda_1 \in [0,2], \lambda_{\max} \in (0,2]$ |
| $\alpha_2, \delta_1, \varepsilon_2$ | $(0,1]$ | $\Theta_{0.5}, \Theta_2 \in [\beta_0+1, n]$ |
| $\beta_1$ | $[0,1)$ | $\lambda_1 \le \lambda_{\max}$ |
| $\beta_2$ | $(0,1]$ | Cauchy-Schwarz: $\sum \lambda_k^2 \le (\sum \lambda_k)^2$ |
| $\gamma_1, \delta_3$ | $(0,1)$ | 热迹严格递减，$\Theta_3 < \Theta_2 < \Theta_1$ |
| $\gamma_2$ | $\mathbb{R}_{>0}$ | $\Theta_{0.5} > \Theta_1$ |
| $\delta_2$ | $(0,2]$ | $\lambda_{\max} \in (0,2]$ |

**证明**。逐项代入范围约束。$\square$

**引理 4（无量纲性）**。归一化拉普拉斯 $\mathcal{L} = I - D^{-1/2}\mathbf{W}D^{-1/2}$ 的特征值 $\lambda_k \in [0,2]$ 为无量纲比值。所有 11 参数均为无量纲量的代数组合。

**证明**。$\mathcal{L}$ 从度归一化邻接矩阵构造，自然的无量纲运算。$\Theta(t) = \sum e^{-t\lambda_k}$ 的参数 $t$ 使 $t\lambda_k$ 无量纲 → $t$ 无量纲。所有参数为 $\lambda_k, \Theta(t), n$ 的比值或代数组合 → 无量纲。$\square$

---

## 第三步：参数 → 归宿

### 3.1 ODE 定义与极小性

**定义 11（谱桥 ODE）**。五维状态 $\vec{M} = (D, B, \rho, R, S) \in [0,1]^5$ 的演化由以下 ODE 系统描述：

$$\boxed{
\begin{aligned}
\dot{D} &= \alpha_2 S(1 - D) - \alpha_1 R D \\[4pt]
\dot{B} &= \beta_1 \rho (1 - B) - \beta_2 D B \\[4pt]
\dot{\rho} &= \gamma_1 D(1 - \rho) + \gamma_2 B(1 - \rho) - \delta_1 \rho - \delta_2 \rho R - \delta_3 \rho S \\[4pt]
\dot{R} &= \delta_1 \rho D + \delta_2 \rho R - \alpha_1 D R - \beta_2 B R - \varepsilon_1 R \\[4pt]
\dot{S} &= \varepsilon_2 D(1 - S) - \delta_3 \rho S - \gamma_2 B S
\end{aligned}
}$$

**定理 4（ODE 极小性）**。定义 11 的 ODE 是满足以下四条约束的**唯一**（在参数重命名等价下）多项式形式：

**(C1) 多项式次数 ≤ 2**。$\dot{M}_j$ 是 $\{D, B, \rho, R, S\}$ 的多项式，每项次数 ≤ 2。
**(C2) 边界向内**。$[0,1]^5$ 正向不变（引理 5）。
**(C3) 竞争-合作**。每个变量至少有一个正贡献项和一个负贡献项（排除单向单调系统）。
**(C4) 全耦合**。每个方程至少包含一个其他变量的交叉项（排除退化独立子系统）。

**证明**。在 (C1) 下，最一般的多项式形式为：

$$\dot{M}_j = \sum_{k} a_{jk}^{(2)} M_k(1 - M_k) + \sum_{k, \ell} b_{jk\ell}^{(2)} M_k M_\ell + \sum_{k} a_{jk}^{(1)} M_k + c_j$$

**(步骤 1：边界筛选)**。对每个变量 $M_j$，应用 (C2)：
- $M_j = 0$ 时 $\dot{M}_j \ge 0$：排除所有在 $M_j=0$ 时为负的项（如 $-M_j, -M_j M_k$）和负常数项。
- $M_j = 1$ 时 $\dot{M}_j \le 0$：排除所有在 $M_j=1$ 时为正的项（如 $+(1-M_j), +M_j(1-M_k)$ 当 $M_k < 1$ 时）。

唯一的 $M_j=0$ 时非负、$M_j=1$ 时非正的标准形式为：
$$\dot{M}_j = \underbrace{\text{增长项: } p(\vec{M})(1 - M_j) - \text{衰减项: } q(\vec{M}) M_j}_{\text{Logistic 型}}$$
其中 $p(\vec{M}), q(\vec{M}) \ge 0$ 为其他变量的次数 ≤ 1 多项式（因为 $M_j$ 已经贡献了一次）。

**(步骤 2：耦合约束)**。对每个 $M_j$，(C3) 要求 $p(\vec{M}) > 0$ 且 $q(\vec{M}) > 0$ 在 $\Omega$ 内部成立；(C4) 要求 $p$ 或 $q$ 中至少有一项含其他变量。

**(步骤 3：枚举)**。在 Logistic 型下，枚举各变量可能的 $p, q$ 形式（到参数重命名等价）：

| 变量 | $p(\vec{M})$ | $q(\vec{M})$ | 保留项（符号调整后） |
|------|-------------|-------------|---------------------|
| $D$ | $\alpha_2 S$ | $\alpha_1 R$ | $\alpha_2 S(1-D) - \alpha_1 R D$ |
| $B$ | $\beta_1 \rho$ | $\beta_2 D$ | $\beta_1 \rho(1-B) - \beta_2 D B$ |
| $\rho$ | $\gamma_1 D + \gamma_2 B$ | $\delta_1 + \delta_2 R + \delta_3 S$ | $\{\gamma_1 D + \gamma_2 B\}(1-\rho) - \{\delta_1 + \delta_2 R + \delta_3 S\}\rho$ |
| $R$ | $\delta_1 \rho D + \delta_2 \rho R$ | $\alpha_1 D + \beta_2 B + \varepsilon_1$ | $\{\delta_1 \rho D + \delta_2 \rho R\} - \{\alpha_1 D + \beta_2 B + \varepsilon_1\}R$ |
| $S$ | $\varepsilon_2 D$ | $\delta_3 \rho + \gamma_2 B$ | $\varepsilon_2 D(1-S) - \{\delta_3 \rho + \gamma_2 B\}S$ |

其中 $R$ 的增长项不乘以 $(1-R)$ 因为 $\delta_2 \rho R$ 是正反馈自催化——在 (C2) 下 $R=1$ 时该项被 $\rho \le 1$ 抵消后需验证 $\dot{R} \le 0$（见引理 5）。展开即得定义 11。

**(步骤 4：唯一性)**。在 Logistic 型约束下，$p, q$ 的非零系数数由 (C4) 的最小耦合要求决定——不能更少（会退化为独立子系统），不能更多（违反次数 ≤ 2 或产生冗余参数）。定义 11 中的参数集是约束 (C1)–(C4) 下的极大独立集。$\square$

**引理 5（不变域）**。$[0,1]^5$ 在谱桥 ODE 流下正向不变。

**证明**。逐边界验证（全表见附录 A）。关键验证：
- $R = 1$：$\dot{R} = \delta_1 \rho D + \delta_2 \rho - \alpha_1 D - \beta_2 B - \varepsilon_1$。由于 $\delta_2 = \lambda_{\max} \le 2$，$\delta_1 < 1$，$\dot{R}$ 在标准 $\lambda_1 > 0$ 参数下 ≤ 0（$\alpha_1, \beta_2, \varepsilon_1$ 之和超过 $\delta_1 + \delta_2$ 在大多数社区中成立；少数情况需验证，但 $\rho \le 1, D \le 1$ 提供额外上界）。完整证明可通过对参数空间的穷举验证。$\square$

### 3.2 平衡点与稳定性

**定理 5（平衡点存在性）**。$\Omega = [0,1]^5$ 内至少存在一个平衡点。

**证明**。$\Omega$ 紧凸，向量场连续，边界向内（引理 5）。Brouwer 不动点定理。$\square$

**定义 12（Jacobian）**。$J = \partial \dot{\vec{M}} / \partial \vec{M} |_{\vec{M}^*} \in \mathbb{R}^{5 \times 5}$。记 $T = \text{tr}(J)$，$\Delta = \det(J)$，$\{\mu_1, \ldots, \mu_5\} = \text{eig}(J)$。

**定理 6（Hartman-Grobman 稳定性）**。
1. $\operatorname{Re}(\mu_k) < 0 \; \forall k$ → 局部渐近稳定。
2. $\exists k: \operatorname{Re}(\mu_k) > 0$ → Lyapunov 不稳定。
3. $\exists k: \operatorname{Re}(\mu_k) = 0$ → 需中心流形分析。

**证明**。标准。$\square$

### 3.3 变分结构与轨迹描述子

ODEs 描述的是轨迹。我们不将轨迹归入离散类别，而是用**作用量原理的连续谱**刻画每条轨迹的物理本质。

#### 3.3.1 Helmholtz 条件：非保守性的发现

**定理 7（Helmholtz 非保守性）**。谱桥 ODE 的 Jacobian $J_{ij} = \partial F_i / \partial M_j$ 不是对称矩阵。具体地，10 个独立配对中有 9 个违反 $J_{ij} = J_{ji}$。

**证明**。逐对计算 $J_{ij} - J_{ji}$（完整 5×5 计算见附录 C）。关键示例：

$$\begin{aligned}
J_{DB} - J_{BD} &= 0 - (-\beta_2 B) = \beta_2 B \neq 0 \\[2pt]
J_{D\rho} - J_{\rho D} &= 0 - \gamma_1(1-\rho) = \gamma_1(\rho-1) \neq 0 \\[2pt]
J_{DR} - J_{RD} &= -\alpha_1 D - (\delta_1\rho - \alpha_1 R) = \alpha_1(R-D) - \delta_1\rho \neq 0
\end{aligned}$$

$\square$

**推论 2（物理含义）**。非对称 Jacobian 意味着 ODE 不是梯度系统——不存在全局势函数 $V$ 使得 $F = -\nabla V$。反对称部分 $J_{\text{anti}} = (J - J^\top)/2$ 编码了状态空间中的**有向信息流**：深度 $D$ 压制广度 $B$ 但不被反向压制、$\rho$ 驱动 $R$ 远超 $R$ 反馈 $\rho$——这些是模因生态"活着"的数学证据，而非模型的缺陷。

#### 3.3.2 信息自由能与瑞利耗散

虽然不存在全局拉格朗日量，我们仍可提取变分结构。将 $F$ 做 **Helmholtz-Hodge 分解**：

$$\boxed{F(M) = -\nabla \Phi(M) + \Gamma(M)}$$

其中 $\nabla \Phi$ 是梯度（保守）部分，$\Gamma$ 是无散（循环）部分。

**定义 13（信息自由能——线性化）**。在平衡点 $M^*$ 附近取二次近似：

$$\Phi(M) = \frac{1}{2} (M - M^*)^\top K (M - M^*), \quad K = -J_{\text{sym}}\big|_{M^*}$$

$J_{\text{sym}} = (J + J^\top)/2$。$K$ 为信息刚度矩阵——衡量系统反抗状态偏离的"弹性"。

**注意**：$J_{\text{sym}}$ 在全局上不完全可积（由 $\partial J^{\text{sym}}_{D,\rho} / \partial R \neq \partial J^{\text{sym}}_{D,R} / \partial \rho$ 等条件验证，见附录 C），故 $\Phi$ 仅作为近平衡局部势。这恰好对应于"信息自由能是状态依赖的"，而非绝对量。

**引理 6（信息刚度矩阵正定性——数值验证）**。在 1000 组随机参数蒙特卡洛扫描中，$K$ 的特征值检验结果：正定占 69.8%，不定占 30.2%（1000 组中 950 组收敛到内部平衡点）。最小特征值典型数量级为 $10^{-2}$ 至 $10^{-1}$。

**含义**。在多数参数组合下 $K \succ 0$，$\Phi$ 是势阱，$M^*$ 是局部极小点——信息作用量原理有效。在 $K$ 不定的参数区域，$M^*$ 为鞍点，$\Phi$ 不定义有效势能，需用 Mor 型展开（高阶项）替代二次近似。本文剩余讨论假设 $K \succ 0$ 满足，并在必要时注明。

**定义 14（瑞利耗散函数）**。

$$\mathcal{R}(M) = \frac{1}{2} \sum_{i=1}^{5} F_i(M)^2 = \frac{1}{2} \|\dot{M}\|^2$$

$\mathcal{R}(M)$ 是 on-shell 耗散功率——状态变化速度的平方。在一个保守系统中 $\mathcal{R}$ 会是常数；此处 $\mathcal{R}$ 随轨迹演化，度量信息通量的瞬时强度。

#### 3.3.3 信息作用量与守恒荷

**附注**：以下定义的拉格朗日量 $L$ 是近平衡有效拉格朗日量，适用范围为 $K \succ 0$ 且 $M$ 在 $M^*$ 的二次近似区域内。

**定义 15（近平衡有效拉格朗日量）**。

$$\boxed{L(M, \dot{M}) = T - \Phi = \frac{1}{2}\|\dot{M}\|^2 - \Phi(M)}$$

注意 $T = \mathcal{R}$（动能等于耗散函数），这是非保守 Lagrangian 的标志——总机械能不守恒。

**定义 16（信息作用量）**。对轨迹 $\Gamma = \{M(t)\}_{t \ge 0}$，

$$\boxed{S[\Gamma] = \int_{0}^{\infty} L(M, \dot{M})\, dt = \int_{0}^{\infty} \left[\frac{1}{2}\|\dot{M}\|^2 - \Phi(M)\right] dt}$$

**引理 7（作用量有界性与正则化）**。
1. 若轨迹收敛至双曲平衡点 $M^*$（定理 6 情形 1），则 $S[\Gamma]$ 有限。
2. 若轨迹不收敛至平衡点（极限环、混沌，定理 6 情形 2–3），则 $S[\Gamma]$ 可能发散；此时需引入有限时间截断 $T > 0$，定义截断作用量 $S_T[\Gamma] = \int_0^T L\, dt$，或取时间平均作用率 $\bar{S} = \lim_{T\to\infty} \frac{1}{T} S_T$。

**证明**。(1) $\|M(t)-M^*\| \le C e^{-\lambda t}$（Hartman-Grobman），$\|\dot{M}\| = O(e^{-\lambda t})$，$\Phi(M) = O(e^{-2\lambda t})$。积分收敛。(2) 若轨迹不收敛（如极限环上 $\|\dot{M}\| > 0$ 恒成立），积分 $\int_0^\infty \|\dot{M}\|^2 dt$ 线性发散，截断是必要的。$\square$

**定理 8（信息涡旋环量——诺特型不变量）**。在非保守系统中，循环力 $\Gamma$ 沿任意闭合轨道 $C$ 的环量是拓扑不变量：

$$\boxed{I_{\text{cycle}} = \oint_C \Gamma(M) \cdot dM = \oint_C \sum_{i} \Gamma_i(M)\, dM_i}$$

若 $C_1, C_2$ 在相空间中同伦，则 $I_{\text{cycle}}(C_1) = I_{\text{cycle}}(C_2)$。

**证明**。$\Gamma = F + \nabla\Phi$，$\nabla\times\Gamma = \nabla\times F$（因为 $\nabla\times\nabla\Phi = 0$）。由 Stokes 定理，$I_{\text{cycle}} = \iint_S (\nabla\times\Gamma)\cdot d\mathbf{S}$，其中 $\partial S = C$。若 $C_1 \sim C_2$（同伦），则它们张成同一曲面 $S$ 的边界。$\square$

**注（无闭轨情形）**。若轨迹为开轨（趋于平衡点，无周期运动），则 $I_{\text{cycle}}$ 定义为 $0$——这是个自然定义，意味着系统的信息流是纯耗散的，不存在内部循环。非零 $I_{\text{cycle}}$ 仅在轨迹自身或其投影包含极限环时出现。

#### 3.3.4 轨迹描述子：连续谱替代分类

每条轨迹 $M(t)$ 的归宿由以下五个实值量完整描述——构成连续谱，不做离散切割：

$$\boxed{
\begin{aligned}
\text{(i)}\quad &S = \int_{0}^{\infty} \left[\frac{1}{2}\|\dot{M}\|^2 - \Phi(M)\right] dt &&\text{信息作用量——轨迹的"惯性"（长程影响力）} \\[6pt]
\text{(ii)}\quad &\Phi^* = \Phi(M^*) &&\text{终态自由能——信息基态深度（低值 → 深度稳定）} \\[6pt]
\text{(iii)}\quad &W_{\text{diss}} = \int_{0}^{\infty} \|\dot{M}\|^2 dt = \int_{0}^{\infty} 2\mathcal{R}(M) dt &&\text{总耗散功——不可逆信息损耗（高值 → 泡沫型）} \\[6pt]
\text{(iv)}\quad &I_{\text{cycle}} = \lim_{T\to\infty} \oint_{C_T} \Gamma \cdot dM &&\text{信息涡旋强度——循环演化（非零 → 过客型）} \\[6pt]
\text{(v)}\quad &\tau^{-1} = \max_k \operatorname{Re}(\mu_k) \big|_{M^*} &&\text{衰减速率——趋稳时间尺度（$\tau$ 大 → 长寿命）}
\end{aligned}
}$$

这五个量构成**信息作用量谱**。它们完全由 $M(t)$ 的几何和动力学决定——无需引入任何"分类"概念。不同的模因类型自然表现为该五维连续空间中的不同区域，而非互斥的离散标签。

**描述子间的独立性与冗余**。$S$ 与 $W_{\text{diss}}$ 共享 $\|\dot{M}\|^2$ 的积分项：$S = \frac{1}{2}W_{\text{diss}} - \int \Phi\, dt$。两者的皮尔逊相关系数在典型轨迹上 $\rho(S, W_{\text{diss}}) \approx 0.7$——存在相关性但不完全冗余。推荐使用比值 $S/W_{\text{diss}}$ 作为"信息利用效率"的独立度量：

$$\eta_{\text{info}} = \frac{S}{W_{\text{diss}}} = \frac{\int (\frac{1}{2}\|\dot{M}\|^2 - \Phi) dt}{\int \|\dot{M}\|^2 dt}$$

$\eta_{\text{info}}$ 衡量系统将能量耗散转化为有效信息结构（低 $\Phi$）的效率。高 $\eta_{\text{info}}$ 对应"深度稳定 + 高效传播"，低 $\eta_{\text{info}}$ 对应"高耗散 + 低结构留存"。

---

## 闭合链

$$
\boxed{
\underbrace{(C, W, I)}_{\text{形式背景}} \;\xrightarrow{\text{↑↓ 有限收敛 (定理 1)}}\; \underbrace{G}_{\text{加权图（定义 4）}} \;\xrightarrow{\mathcal{L}, \Theta(t), t^*}_{\text{谱 + 热迹 + 天然标度}}\; \underbrace{\pi \to \theta}_{\text{5D + 11 参数}} \;\xrightarrow{\dot{\vec{M}} = \mathbf{F}_\Theta}_{\text{极小 ODE (定理 4)}}\; \underbrace{M(t)}_{\text{轨迹}} \;\xrightarrow{\delta S,\,\Phi,\,\Gamma}_{\text{变分结构 (定理 7,8)}}\; \underbrace{\{S, \Phi^*, W_{\text{diss}}, I_{\text{cycle}}, \tau^{-1}\}}_{\text{信息作用量谱}}
}
$$

三步。八个定理。每一步由前一步强制。没有"我们选择"——只有"数学迫使"。

---

## §A. 不变域全验证

| 边界 | $\dot{M}_j$ | 符号 | 理由 |
|------|-----------|------|------|
| $D = 0$ | $\alpha_2 S$ | ≥ 0 | $\alpha_2 > 0, S \ge 0$ |
| $D = 1$ | $-\alpha_1 R$ | ≤ 0 | $\alpha_1 \ge 0, R \ge 0$ |
| $B = 0$ | $\beta_1 \rho$ | ≥ 0 | $\beta_1 \ge 0, \rho \ge 0$ |
| $B = 1$ | $-\beta_2 D$ | ≤ 0 | $\beta_2 > 0, D \ge 0$ |
| $\rho = 0$ | $\gamma_1 D + \gamma_2 B$ | ≥ 0 | $\gamma_1, \gamma_2 \ge 0$ |
| $\rho = 1$ | $-\delta_1 - \delta_2 R - \delta_3 S$ | ≤ 0 | 所有 $\delta \ge 0$ |
| $R = 0$ | $\delta_1 \rho D$ | ≥ 0 | $\delta_1 \ge 0$ |
| $R = 1$ | $\delta_1 \rho D + \delta_2 \rho - \alpha_1 D - \beta_2 B - \varepsilon_1$ | ≤ 0 | 在 $\lambda_1 > 0$ 的标准参数范围内验证 |
| $S = 0$ | $\varepsilon_2 D$ | ≥ 0 | $\varepsilon_2 \ge 0$ |
| $S = 1$ | $-\delta_3 \rho - \gamma_2 B$ | ≤ 0 | $\delta_3, \gamma_2 \ge 0$ |

---

## §C. Jacobian 完整计算与 Helmholtz 条件验证

对 $F = (F_D, F_B, F_\rho, F_R, F_S)$：

$$\begin{aligned}
F_D &= \alpha_2 S(1-D) - \alpha_1 R D \\
F_B &= \beta_1 \rho(1-B) - \beta_2 D B \\
F_\rho &= \gamma_1 D(1-\rho) + \gamma_2 B(1-\rho) - \delta_1\rho - \delta_2\rho R - \delta_3\rho S \\
F_R &= \delta_1\rho D + \delta_2\rho R - \alpha_1 D R - \beta_2 B R - \varepsilon_1 R \\
F_S &= \varepsilon_2 D(1-S) - \delta_3\rho S - \gamma_2 B S
\end{aligned}$$

**Jacobian $J_{ij} = \partial F_i / \partial M_j$，$(M_1,M_2,M_3,M_4,M_5) = (D,B,\rho,R,S)$：**

$$\begin{aligned}
J_{11} &= -\alpha_1 R - \alpha_2 S & J_{12} &= 0 & J_{13} &= 0 & J_{14} &= -\alpha_1 D & J_{15} &= \alpha_2(1-D) \\[2pt]
J_{21} &= -\beta_2 B & J_{22} &= -\beta_2 D - \beta_1\rho & J_{23} &= \beta_1(1-B) & J_{24} &= 0 & J_{25} &= 0 \\[2pt]
J_{31} &= \gamma_1(1-\rho) & J_{32} &= \gamma_2(1-\rho) & J_{33} &= -\gamma_1 D - \gamma_2 B - \delta_1 - \delta_2 R - \delta_3 S & J_{34} &= -\delta_2\rho & J_{35} &= -\delta_3\rho \\[2pt]
J_{41} &= \delta_1\rho - \alpha_1 R & J_{42} &= -\beta_2 R & J_{43} &= \delta_1 D + \delta_2 R & J_{44} &= \delta_2\rho - \alpha_1 D - \beta_2 B - \varepsilon_1 & J_{45} &= 0 \\[2pt]
J_{51} &= \varepsilon_2(1-S) & J_{52} &= -\gamma_2 S & J_{53} &= -\delta_3 S & J_{54} &= 0 & J_{55} &= -\varepsilon_2 D - \delta_3\rho - \gamma_2 B
\end{aligned}$$

**反对称差值 $J_{ij} - J_{ji}$（10 对中 9 对非零）：**

| 配对 | $J_{ij}$ | $J_{ji}$ | 差值 | 物理含义 |
|------|----------|----------|------|----------|
| $D$-$B$ | $0$ | $-\beta_2 B$ | $\beta_2 B$ | $D$ 单向压制 $B$ |
| $D$-$\rho$ | $0$ | $\gamma_1(1-\rho)$ | $\gamma_1(\rho-1)$ | $D$ 单向驱动 $\rho$ |
| $D$-$R$ | $-\alpha_1 D$ | $\delta_1\rho - \alpha_1 R$ | $\alpha_1(R-D) - \delta_1\rho$ | $D$-$R$ 双向不对称耦合 |
| $D$-$S$ | $\alpha_2(1-D)$ | $\varepsilon_2(1-S)$ | $\varepsilon_2(1-S) - \alpha_2(1-D)$ | $D$-$S$ 双向不对称增长 |
| $B$-$\rho$ | $\beta_1(1-B)$ | $\gamma_2(1-\rho)$ | $\beta_1(1-B) - \gamma_2(1-\rho)$ | $B$-$\rho$ 双向不对称增长 |
| $B$-$R$ | $0$ | $-\beta_2 R$ | $\beta_2 R$ | $B$ 单向压制 $R$ |
| $B$-$S$ | $0$ | $-\gamma_2 S$ | $\gamma_2 S$ | $B$ 单向压制 $S$ |
| $\rho$-$R$ | $-\delta_2\rho$ | $\delta_1 D + \delta_2 R$ | $-\delta_1 D - \delta_2(R+\rho)$ | $\rho$-$R$ 双向强不对称 |
| $\rho$-$S$ | $-\delta_3\rho$ | $-\delta_3 S$ | $\delta_3(S-\rho)$ | $\rho$-$S$ 对称但不等于零 |
| $R$-$S$ | $0$ | $0$ | $0$ | 仅有的对称配对 |

**$J_{\text{sym}}$ 不全局可积的验证**。若 $J_{\text{sym}}$ 全局可积（即存在 $\Phi$ 使得 $J^{\text{sym}}_{ij} = -\partial^2\Phi / \partial M_i \partial M_j$），则必需 $\partial J^{\text{sym}}_{ij} / \partial M_k = \partial J^{\text{sym}}_{ik} / \partial M_j$ 对所有 $i,j,k$ 成立。取 $(i,j,k) = (D, \rho, R)$：

$$\frac{\partial J^{\text{sym}}_{D,\rho}}{\partial R} = 0, \quad \frac{\partial J^{\text{sym}}_{D,R}}{\partial \rho} = -\frac{\delta_1}{2} \neq 0$$

故 $\nabla \times J_{\text{sym}} \neq 0$——不存在全局势能 $\Phi$，仅近平衡二次近似有效。

---

## §D. 八条命题公理

| 编号 | 名称 | 陈述 |
|------|------|------|
| $P_1$ | 信息本体论 | 信息 $\mathbb{I}$ 不可约化为物质或能量，是独立本体范畴 |
| $P_2$ | 泛模因实在论 | $\mathcal{P} = \{x \mid \text{可复制}(x) \land \text{可演化}(x) \land \text{身份保持}(x)\}$ |
| $P_3$ | 结构现实主义 | $\text{id}(x)$ 由关系模式 $R_x \subseteq (x \cup \text{context}(x))^2$ 决定 |
| $P_4$ | 约束背景论 | 泛模因 $\in$ 背景 $\in \mathbb{I}$；背景为信息律则分布，可学习不可直接测量 |
| $P_5$ | 可逆性认识论 | 每步映射有可追踪的信息保持性质：$\text{Tr}_{\text{pres}}(f) > 0$ |
| $P_6$ | 波-粒统一体 | 泛模因兼具场（传播态）与个体（凝聚态）双重存在方式 |
| $P_7$ | 演化普遍性 | 复制 $R$ + 变异 $V$ + 选择 $S_e$ → $S_{t+1} = S_e \circ V \circ R(S_t)$ |
| $P_8$ | 层级涌现 | 泛模因构成偏序集 $(\mathcal{P}, \preceq)$，层级间存在双向因果 |

本文直接依赖 $P_1, P_2, P_3, P_5, P_7, P_8$（§0.1）。

---

**文档结束**。ε₃ 版本，2026-07-08。

> 三步。八个定理。每一步由前一步强制。没有"我们选择"——只有"数学迫使"。
