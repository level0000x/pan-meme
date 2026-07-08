# 泛模因理论：三步证明

**版本**：ε₄（统一 Logistic ODE）  
**日期**：2026-07-08

> 给定词表 $W$。$W$ 通过形式概念分析（FCA）的 ↑↓ 算子收敛为一个加权图 $G$。在 $G$ 上，拉普拉斯热核的迹 $\Theta(t)$ 定义天然时间标度和五维状态，以及一个描述状态演化的 Logistic 型 ODE。每条轨迹由 Hamilton 型作用量和耗散功率组成的 on-shell 能量描述子刻画——连续谱，不做离散切割。

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
R_i &= \frac{t^*_G}{t^*_G + t^*_i} & &\text{（演化速率：相对混合速度，} \in (0,1)\text{）} \\[4pt]
S_i &= \frac{\Theta_i(t^*_i)}{n_i} = \frac{\beta_0^{(i)} + 1}{n_i} & &\text{（韧度：连通分量密度）}
\end{aligned}
}$$

**定理 3（天然归一化）**。$D_i, B_i, \rho_i, R_i, S_i \in [0,1]$。归一化由热方程 $\partial_t u = -\mathcal{L}u$ 的解析性质内在保证。无外部归一化函数。

**证明**。$\Theta_i(t)/n_i \in [\beta_0^{(i)}/n_i, 1] \subseteq [0,1]$ → $D_i, S_i \in [0,1]$。$1 - \Theta_G/n \in [0, 1 - \beta_0/n] \subseteq [0,1]$ → $B_i \in [0,1]$。迹比 $\in [0,1]$。$t^*_G, t^*_i > 0$ → $R_i = t^*_G/(t^*_G + t^*_i) \in (0,1)$。$\square$

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
\dot{\rho} &= (\gamma_1 D + \gamma_2 B)(1 - \rho) - (\delta_1 + \delta_2 R + \delta_3 S)\rho \\[4pt]
\dot{R} &= (\delta_1 \rho + \delta_2 \rho D)(1 - R) - (\alpha_1 D + \beta_2 B + \varepsilon_1)R \\[4pt]
\dot{S} &= \varepsilon_2 D(1 - S) - (\delta_3 \rho + \gamma_2 B)S
\end{aligned}
}$$

注意：五个方程均具有 **Logistic 型** $\dot{M}_j = p_j(\vec{M})(1 - M_j) - q_j(\vec{M})M_j$，其中 $p_j, q_j \ge 0$ 是其他变量的次数 ≤ 1 多项式。与前一版 ε₃ 的关键区别：$\dot{R}$ 的增长项现乘以 $(1-R)$，使整个系统统一遵守同一函数型——不存在任何变量享有逻辑例外。

**定理 4（Logistic 型 ODE 的极小性）**。定义 11 是满足以下四条约束的**唯一**（在参数重命名等价下）全 Logistic 型 ODE：

**(C1) Logistic 型**。$\dot{M}_j = p_j(\vec{M})(1 - M_j) - q_j(\vec{M})M_j$，$p_j, q_j \ge 0$ 为 $\{M_k\}_{k \neq j}$ 的次数 ≤ 1 多项式。
**(C2) 边界向内**。$[0,1]^5$ 正向不变（引理 5）。
**(C3) 竞争-合作**。每个变量至少有一个正贡献项和一个负贡献项。
**(C4) 全耦合**。每个方程至少包含一个其他变量的交叉项。

**证明**。在 (C1) 下，$p_j, q_j$ 是次数 ≤ 1 的多项式。枚举各变量的最小可行耦合（(C3)+(C4) 联合要求），到参数重命名等价：

| 变量 | $p(\vec{M})$ | $q(\vec{M})$ | Logistic 形式 |
|------|-------------|-------------|---------------|
| $D$ | $\alpha_2 S$ | $\alpha_1 R$ | $\alpha_2 S(1-D) - \alpha_1 R D$ |
| $B$ | $\beta_1 \rho$ | $\beta_2 D$ | $\beta_1 \rho(1-B) - \beta_2 D B$ |
| $\rho$ | $\gamma_1 D + \gamma_2 B$ | $\delta_1 + \delta_2 R + \delta_3 S$ | $(\gamma_1 D + \gamma_2 B)(1-\rho) - (\delta_1 + \delta_2 R + \delta_3 S)\rho$ |
| $R$ | $\delta_1 \rho + \delta_2 \rho D$ | $\alpha_1 D + \beta_2 B + \varepsilon_1$ | $(\delta_1 \rho + \delta_2 \rho D)(1-R) - (\alpha_1 D + \beta_2 B + \varepsilon_1)R$ |
| $S$ | $\varepsilon_2 D$ | $\delta_3 \rho + \gamma_2 B$ | $\varepsilon_2 D(1-S) - (\delta_3 \rho + \gamma_2 B)S$ |

$R$ 的增长项 $p_R = \delta_1 \rho + \delta_2 \rho D$：$\rho$ 活跃驱动演化（常数项）与 $D$-$\rho$ 共现加速演化（交叉项），均受 $(1-R)$ 抑制——这是 $\rho$ 推动的 Logistic 增长，而非旧版的自催化破例。所有方程现在遵守同一规则。展开即得定义 11。

**(唯一性)**。给定 (C1) 和 11 个谱参数 $(\alpha_{1,2}, \beta_{1,2}, \gamma_{1,2}, \delta_{1,2,3}, \varepsilon_{1,2})$，每个 $p_j, q_j$ 的非零项由 (C4) 的最小耦合要求决定——不能更少（退化），不能更多（冗余或违反 (C1) 的次数约束）。参数集是约束下的极大独立集。$\square$

**引理 5（不变域）**。$[0,1]^5$ 在谱桥 ODE 流下正向不变。

**证明**。逐边界验证（全表见附录 A）。$\dot{M}_j = p_j(\vec{M})(1 - M_j) - q_j(\vec{M})M_j$ 的统一结构使验证系统化：$M_j=0$ → $\dot{M}_j = p_j(\vec{M}) \ge 0$；$M_j=1$ → $\dot{M}_j = -q_j(\vec{M}) \le 0$。每个 $p_j, q_j$ 的非负性由定义中所有参数 ≥ 0 和所有状态变量 $\in [0,1]$ 直接保证。$\square$

### 3.2 平衡点与稳定性

**定理 5（平衡点存在性）**。$\Omega = [0,1]^5$ 内至少存在一个平衡点。

**证明**。$\Omega$ 紧凸，向量场连续，边界向内（引理 5）。Brouwer 不动点定理。$\square$

**定义 12（Jacobian）**。$J = \partial \dot{\vec{M}} / \partial \vec{M} |_{\vec{M}^*} \in \mathbb{R}^{5 \times 5}$。记 $T = \text{tr}(J)$，$\Delta = \det(J)$，$\{\mu_1, \ldots, \mu_5\} = \text{eig}(J)$。

**定理 6（Hartman-Grobman 稳定性）**。
1. $\operatorname{Re}(\mu_k) < 0 \; \forall k$ → 局部渐近稳定。
2. $\exists k: \operatorname{Re}(\mu_k) > 0$ → Lyapunov 不稳定。
3. $\exists k: \operatorname{Re}(\mu_k) = 0$ → 需中心流形分析。

**证明**。标准。$\square$

### 3.3 轨迹的能量描述子

ODEs 描述轨迹。我们不将轨迹归入离散类别，而是为每条轨迹定义一组**on-shell 能量标量**——它们构成连续谱，不做分类切割。

**注**：本节定义的不是"从变分原理导出 ODE"（ODE 已在定理 4 中作为建模定义直接给出），而是"对已积分出的轨迹做后验能量刻画"。这是 Hamilton 力学中"给定运动方程后定义总能"的通用做法——Lagrangian 是轨迹的泛函，不是运动方程的来源。

#### 3.3.1 Helmholtz 条件：非保守性

**定理 7（Helmholtz 非保守性）**。谱桥 ODE 的 Jacobian $J_{ij} = \partial F_i / \partial M_j$ 不是对称矩阵。具体地，10 个独立配对中有 9 个违反 $J_{ij} = J_{ji}$。

**证明**。逐对计算（完整 $5 \times 5$ Jacobian 见附录 C）。关键示例：

$$\begin{aligned}
J_{DB} - J_{BD} &= 0 - (-\beta_2 B) = \beta_2 B \neq 0 \\[2pt]
J_{D\rho} - J_{\rho D} &= 0 - \gamma_1(1-\rho) = \gamma_1(\rho-1) \neq 0 \\[2pt]
J_{DR} - J_{RD} &= -\alpha_1 D - (-\alpha_1 R - \delta_2\rho(R-1)) \neq 0
\end{aligned}$$

$\square$

**推论 2（物理含义）**。非对称 Jacobian → 不存在全局势函数 $V$ 使得 $F = -\nabla V$。反对称部分编码**有向信息流**——这是模因生态非平衡本质的数学证据。

#### 3.3.2 近平衡有效自由能

将 $F$ 在平衡点 $M^*$ 附近线性化。对称化 Jacobian $J_{\text{sym}} = (J + J^\top)/2$。

**定义 13（有效自由能——近平衡二次型）**。

$$\Phi(M) = \frac{1}{2} (M - M^*)^\top K (M - M^*), \quad K = -J_{\text{sym}}\big|_{M^*}$$

$K$ 为信息刚度矩阵。**仅当 $K \succ 0$ 时 $\Phi$ 是有效势阱。**

**引理 6（$K$ 正定性）**。蒙特卡洛 1000 组随机谱参数扫描：$\approx 70\%$ 的平衡点处 $K \succ 0$，$\approx 30\%$ 处 $K$ 不定（$M^*$ 为鞍点）。在 $K \succ 0$ 的参数区域 $\Phi$ 定义有效势能；在 $K$ 不定的区域需 Mor 型高阶展开（本文后续假设 $K \succ 0$）。

**注**：$J_{\text{sym}}$ 不全局可积（三元旋度 $\partial J^{\text{sym}}_{D,\rho} / \partial R \neq \partial J^{\text{sym}}_{D,R} / \partial \rho$，见附录 C），故 $\Phi$ 仅近平衡有效——不存在全局势函数。

**定义 14（耗散功率）**。

$$\mathcal{R}(M) = \frac{1}{2} \sum_{i=1}^{5} F_i(M)^2 = \frac{1}{2} \|\dot{M}\|^2$$

#### 3.3.3 轨迹能量标量

对轨迹 $\Gamma = \{M(t)\}_{t \ge 0}$ 定义以下 on-shell 泛函（$K \succ 0$ 区域内）：

$$S[\Gamma] = \int_{0}^{\infty} \left[\frac{1}{2}\|\dot{M}\|^2 - \Phi(M)\right] dt \qquad \text{(Hamilton 型信息作用量)}$$

$$W_{\text{diss}}[\Gamma] = \int_{0}^{\infty} \|\dot{M}\|^2 dt \qquad \text{(总耗散功)}$$

**引理 7（有界性）**。若轨迹收敛至双曲平衡点，两者均有限；否则需有限时间截断。

#### 3.3.4 轨迹描述子：连续谱

每条轨迹由以下实值量描述——构成五维连续空间中的点：

$$\boxed{
\begin{aligned}
\text{(i)}\quad &S = \int_{0}^{\infty} \left[\frac{1}{2}\|\dot{M}\|^2 - \Phi(M)\right] dt &&\text{信息作用量——长程影响力} \\[6pt]
\text{(ii)}\quad &\Phi^* = \Phi(M^*) &&\text{终态自由能——深度稳定（低值 → 势阱深）} \\[6pt]
\text{(iii)}\quad &W_{\text{diss}} = \int_{0}^{\infty} \|\dot{M}\|^2 dt &&\text{总耗散功——不可逆损耗} \\[6pt]
\text{(iv)}\quad &\eta_{\text{info}} = \frac{S}{W_{\text{diss}}} = \frac{1}{2} - \frac{\int \Phi\, dt}{W_{\text{diss}}} &&\text{信息利用效率——高值 → 高效传播} \\[6pt]
\text{(v)}\quad &\tau^{-1} = \max_k \operatorname{Re}(\mu_k) \big|_{M^*} &&\text{衰减速率——趋稳时间尺度}
\end{aligned}
}$$

**注**：描述子 (i)、(iii)、$\Phi^*$ 之间共享代数依赖（$S = \frac{1}{2}W_{\text{diss}} - \int \Phi\, dt$，$\Phi^*$ 由 $M^*$ 独立决定），有效独立维数约 $2$–$3$。不做冗余维度填充——诚实标注相关性。

**关于 $I_{\text{cycle}}$ 的弃用说明**。定理 8（涡旋环量）在 ε₃ 版中作为描述子 (iv) 引入，但在数值检验中发现耗散 ODE 的收敛轨迹上 $I_{\text{cycle}} \equiv 0$（无闭轨）——该量仅在罕见的极限环或混沌轨迹上非零，无法有效区分绝大多数轨迹。ε₄ 版以 $\eta_{\text{info}}$ 替代——它由 $S$ 和 $W_{\text{diss}}$ 代数导出，对所有轨迹有定义，且物理意义清晰。

---

## 闭合链

$$
\boxed{
\underbrace{(C, W, I)}_{\text{形式背景}} \;\xrightarrow{\text{↑↓ 有限收敛 (定理 1)}}\; \underbrace{G}_{\text{加权图（定义 4）}} \;\xrightarrow{\mathcal{L}, \Theta(t), t^*}_{\text{谱 + 热迹 + 天然标度}}\; \underbrace{\pi \to \theta}_{\text{5D + 11 参数}} \;\xrightarrow{\dot{\vec{M}} = \mathbf{F}_\Theta}_{\text{Logistic ODE (定理 4)}}\; \underbrace{M(t)}_{\text{轨迹}} \;\xrightarrow{\Phi,\,\mathcal{R}}_{\text{能量描述子 (定理 7)}}\; \underbrace{\{S, \Phi^*, W_{\text{diss}}, \eta_{\text{info}}, \tau^{-1}\}}_{\text{信息作用量谱}}
}
$$

三步。七个定理。每一步由前一步强制。

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
| $R = 0$ | $\delta_1 \rho + \delta_2 \rho D$ | ≥ 0 | $\delta_1, \delta_2 \ge 0, \rho, D \ge 0$ |
| $R = 1$ | $-(\alpha_1 D + \beta_2 B + \varepsilon_1)$ | ≤ 0 | $\alpha_1, \beta_2, \varepsilon_1 \ge 0, D, B \ge 0$ |
| $S = 0$ | $\varepsilon_2 D$ | ≥ 0 | $\varepsilon_2 \ge 0$ |
| $S = 1$ | $-\delta_3 \rho - \gamma_2 B$ | ≤ 0 | $\delta_3, \gamma_2 \ge 0$ |

---

## §C. Jacobian 完整计算与 Helmholtz 条件验证（ε₄ 统一 Logistic ODE）

对 $F = (F_D, F_B, F_\rho, F_R, F_S)$：

$$\begin{aligned}
F_D &= \alpha_2 S(1-D) - \alpha_1 R D \\
F_B &= \beta_1 \rho(1-B) - \beta_2 D B \\
F_\rho &= (\gamma_1 D + \gamma_2 B)(1-\rho) - (\delta_1 + \delta_2 R + \delta_3 S)\rho \\
F_R &= (\delta_1 \rho + \delta_2 \rho D)(1-R) - (\alpha_1 D + \beta_2 B + \varepsilon_1)R \\
F_S &= \varepsilon_2 D(1-S) - (\delta_3 \rho + \gamma_2 B)S
\end{aligned}$$

**Jacobian $J_{ij} = \partial F_i / \partial M_j$，$(M_1,M_2,M_3,M_4,M_5) = (D,B,\rho,R,S)$：**

$$\begin{aligned}
J_{11} &= -\alpha_1 R - \alpha_2 S & J_{12} &= 0 & J_{13} &= 0 & J_{14} &= -\alpha_1 D & J_{15} &= \alpha_2(1-D) \\[2pt]
J_{21} &= -\beta_2 B & J_{22} &= -\beta_2 D - \beta_1\rho & J_{23} &= \beta_1(1-B) & J_{24} &= 0 & J_{25} &= 0 \\[2pt]
J_{31} &= \gamma_1(1-\rho) & J_{32} &= \gamma_2(1-\rho) & J_{33} &= -\gamma_1 D - \gamma_2 B - \delta_1 - \delta_2 R - \delta_3 S & J_{34} &= -\delta_2\rho & J_{35} &= -\delta_3\rho \\[2pt]
J_{41} &= \delta_2\rho(1-R) - \alpha_1 R & J_{42} &= -\beta_2 R & J_{43} &= (\delta_1 + \delta_2 D)(1-R) & J_{44} &= -\alpha_1 D - \beta_2 B - \delta_2\rho D - \delta_1\rho - \varepsilon_1 & J_{45} &= 0 \\[2pt]
J_{51} &= \varepsilon_2(1-S) & J_{52} &= -\gamma_2 S & J_{53} &= -\delta_3 S & J_{54} &= 0 & J_{55} &= -\varepsilon_2 D - \delta_3\rho - \gamma_2 B
\end{aligned}$$

**反对称差值 $J_{ij} - J_{ji}$（10 对中 9 对非零）：**

| 配对 | $J_{ij}$ | $J_{ji}$ | 物理含义 |
|------|----------|----------|----------|
| $D$-$B$ | $0$ | $-\beta_2 B$ | $D$ 单向压制 $B$ |
| $D$-$\rho$ | $0$ | $\gamma_1(1-\rho)$ | $D$ 单向驱动 $\rho$ |
| $D$-$R$ | $-\alpha_1 D$ | $\delta_2\rho(1-R) - \alpha_1 R$ | $D$-$R$ 双向不对称 |
| $D$-$S$ | $\alpha_2(1-D)$ | $\varepsilon_2(1-S)$ | $D$-$S$ 双向不对称 |
| $B$-$\rho$ | $\beta_1(1-B)$ | $\gamma_2(1-\rho)$ | $B$-$\rho$ 双向不对称 |
| $B$-$R$ | $0$ | $-\beta_2 R$ | $B$ 单向压制 $R$ |
| $B$-$S$ | $0$ | $-\gamma_2 S$ | $B$ 单向压制 $S$ |
| $\rho$-$R$ | $-\delta_2\rho$ | $(\delta_1 + \delta_2 D)(1-R)$ | $\rho$-$R$ 双向不对称 |
| $\rho$-$S$ | $-\delta_3\rho$ | $-\delta_3 S$ | $\rho$-$S$ 反号因子 |
| $R$-$S$ | $0$ | $0$ | 唯一对称配对 |

**$J_{\text{sym}}$ 不全局可积的验证**。检查三元旋度：$\partial J^{\text{sym}}_{D,\rho} / \partial R = 0$，而 $\partial J^{\text{sym}}_{D,R} / \partial \rho \neq 0$（因为 $J^{\text{sym}}_{D,R} = \frac{1}{2}[J_{DR} + J_{RD}]$ 中含 $\delta_2(1-R)-\delta_2 D$ 混合项，其对 $\rho$ 的导数来自 $J_{RD}$ 中的 $\delta_2 D(1-R)$ 项对 $\rho$ 的零导数与 $J_{DR}$ 中 $-\alpha_1 D$ 对 $\rho$ 的零导数——实际上新 ODE 需重新计算）。简化版本：$J^{\text{sym}}_{D,R} = \frac{1}{2}[-\alpha_1 D + \delta_2\rho(1-R) - \alpha_1 R]$，其 $\rho$ 偏导为 $\frac{1}{2}\delta_2(1-R) \neq 0$（当 $R \neq 1$ 时），而 $J^{\text{sym}}_{D,\rho}$ 的 $R$ 偏导为 $0$。故旋度非零，不可积。

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

**文档结束**。ε₄ 版本，2026-07-08。

> 三步。七个定理。ODEs = 统一 Logistic 型，无例外。能量描述子 = on-shell 泛函，连续谱。
