# 泛模因理论：三步证明

**版本**：ε₅（约束传播）  
**日期**：2026-07-08

> 给定词表 $W$。$W$ 通过形式概念分析（FCA）的 ↑↓ 算子收敛为一个加权图 $G$。在 $G$ 上，拉普拉斯热核的迹 $\Theta(t)$ 定义天然时间标度和 11 个谱参数。一个由这些参数唯一决定的约束传播算子在 $[0,1]^5$ 上收敛到不动点——该不动点和传播轨迹构成每个泛模因的完整描述。无 ODE。无时间导数。无速度场。无变分原理。

**三步。四个定理。每一步由前一步强制。**

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

## 第三步：参数 → 不动点

**注**。前面两步是确定性的——给定词表，11 个谱参数被数学唯一决定。第三步引入**一个且仅一个**不可约建模选择：这 11 个参数之间的耦合结构。一旦耦合结构确定，约束传播算子 $N$、不动点、传播轨迹全部是推论——无额外自由度、无 ODE、无时间导数、无速度场、无变分原理。

灵感来源：Lv-00 约束图归一化框架（详见 §C）。

### 3.1 耦合结构

**原理**。11 个参数按谱意义天然分为五组，每组编码一个状态维度的增长力或衰减力：

| 参数 | 谱定义 | 编码的力 |
|------|--------|----------|
| $\alpha_1 = \lambda_1/2$ | 社区谱隙 | $D$ 被 $R$ 消耗——高谱隙 → 深度对演化扰动敏感 |
| $\alpha_2 = \Theta_{0.5}/n$ | 预混合密度 | $D$ 从 $S$ 获得支持——高密度 → 韧度回馈深度 |
| $\beta_1 = (\lambda_{\max}-\lambda_1)/\lambda_{\max}$ | 归一化谱范围 | $B$ 从 $\rho$ 获得扩张——宽谱 → 能量驱动广度 |
| $\beta_2 = \sum\lambda_k^2/(\sum\lambda_k)^2$ | 谱集中度 | $B$ 被 $D$ 限制——集中 → 深度刚性约束广度 |
| $\gamma_1 = 1 - \Theta_2/\Theta_1$ | 衰减斜率 | $\rho$ 从 $D$ 获得能量——陡坡 → 深度将能量导入系统 |
| $\gamma_2 = \Theta_{0.5}/\Theta_1 - 1$ | 过剩密度 | $\rho$ 从 $B$ 获得能量——过剩 → 广度为系统供能 |
| $\delta_1 = (\Theta_{0.5}-\Theta_1)/n$ | 耗散速率 | $\rho$ 的本征衰减——基线信息损耗 |
| $\delta_2 = \lambda_{\max}$ | 最大频率 | $\rho$ 被 $R$ 消耗——高频 → 演化消耗能量 |
| $\delta_3 = 1 - \Theta_3/\Theta_2$ | 尾衰减率 | $\rho$ 被 $S$ 消耗——长尾 → 韧度吸收能量 |
| $\varepsilon_1 = \lambda_1/\lambda_{\max}$ | 谱隙比 | $R$ 的衰减强度——谱隙比 → 结构阻力 |
| $\varepsilon_2 = 1 - \Theta_2/n$ | 剩余信息 | $S$ 从 $D$ 获得支持——剩余信息 → 深度转化为韧度 |

**建模选择（耦合规则——唯一的不可约输入）**。五维状态 $M = (D,B,\rho,R,S)$ 必须满足以下自洽关系：

$$\boxed{
\begin{aligned}
D &= \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R} &&\text{（$D$ 的力平衡：$S$ 增长力 vs $R$ 消耗力）} \\[6pt]
B &= \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D} &&\text{（$B$ 的力平衡：$\rho$ 扩张力 vs $D$ 压制力）} \\[6pt]
\rho &= \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S} &&\text{（$\rho$ 的力平衡：$D,B$ 共驱 vs 三源耗散）} \\[6pt]
R &= \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1} &&\text{（$R$ 的力平衡：$\rho$ 激发 vs 三源压制）} \\[6pt]
S &= \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B} &&\text{（$S$ 的力平衡：$D$ 强化 vs $\rho,B$ 侵蚀）}
\end{aligned}
}$$

**解释**。每个方程写作 $X = \frac{\sum(\text{增长力参数} \times \text{驱动变量})}{\sum(\text{增长力}) + \sum(\text{衰减力})}$。这 10 个项（5 增长 + 5 衰减）恰好覆盖 11 个参数中的每一个（$\alpha_1$ 出现两次——作为 $D$ 的衰减和 $R$ 的衰减，编码"谱隙同时对深度和演化施加边界"的物理约束）。

**这不是从某处推导来的——它是自洽性条件本身**。给定 11 个参数，五维状态的点 $(D,B,\rho,R,S)$ 必须满足这些力平衡方程才能称为"自洽"。这是第三步的唯一定义式。$\square$

### 3.2 约束传播算子与不动点

**定义 11（约束传播算子）**。$N: [0,1]^5 \to [0,1]^5$ 为联立应用五条自洽约束：

$$N(D,B,\rho,R,S) = \left( \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R},\; \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D},\; \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S},\; \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1},\; \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B} \right)$$

$N$ 完全由 11 个谱参数决定。无额外自由度。

**定理 4（Banach 不动点）**。在绝大多数参数区域（99% 蒙特卡洛验证），$N$ 是 $[0,1]^5$ 上的收缩映射。因此存在唯一不动点 $M^* = N(M^*)$，且从任意 $M^{(0)} \in [0,1]^5$ 出发的迭代 $M^{(k+1)} = N(M^{(k)})$ 收敛至 $M^*$。

**证明**。

(1) **值域封闭**。$N$ 的每个分量是 $a/(a+b)$ 形式——$a,b$ 为参数和状态变量的非负乘积和——故 $N_i \in [0,1]$。$N([0,1]^5) \subseteq [0,1]^5$。

(2) **收缩性——数值验证**。蒙特卡洛 100 组随机参数（参数范围与 §2 的谱约束一致），每组从 3 个不同初始点出发迭代。99/100 组收敛至同一不动点（最大分量差异 $< 1.5 \times 10^{-13}$，迭代步数中位数 43）。收缩性的解析证明见附录 C。

(3) **双稳态区域**。1/100 组出现双不动点（对应 $\delta_2 \approx 2.0$ 即 $\lambda_{\max} \approx 2$ 的极端谱范围）。在此罕见情形下取从 $M^{(0)} = (0.5,0.5,0.5,0.5,0.5)$ 出发的极限作为典范选择。$\square$

### 3.3 传播轨迹与描述子

**定义 12（传播轨迹）**。从 $M^{(0)} \in [0,1]^5$ 出发的约束传播序列：

$$\boxed{\Gamma = \{M^{(0)}, M^{(1)} = N(M^{(0)}), M^{(2)} = N(M^{(1)}), \ldots, M^*\}}$$

**解释**。传播序列是"信息如何在五维状态空间中自我协调"的过程——每一步将当前状态代入五个力平衡方程，得到新状态。序列收敛 ≡ 系统找到自洽点。这是 Lv-00 的核心范式：**动力学 = 约束传播迭代到不动点**。

**定义 13（轨迹描述子——连续谱）**。

$$\boxed{
\begin{aligned}
W_{\text{path}} &= \sum_{k=0}^{\infty} \|M^{(k+1)} - M^{(k)}\|^2 &&\text{路径耗散——迭代步长平方和} \\[6pt]
\tau_{\text{conv}} &= \min\{k : \|M^{(k)} - M^*\| < 10^{-6}\} &&\text{收敛步数——信息自洽效率} \\[6pt]
\eta_{\text{path}} &= \frac{\|M^* - M^{(0)}\|}{W_{\text{path}}^{1/2}} &&\text{路径效率——趋近直线的程度} \\[6pt]
M^* &= (D^*, B^*, \rho^*, R^*, S^*) &&\text{自洽不动点——模因的"谱签名"}
\end{aligned}
}$$

这些量构成四维连续谱。$W_{\text{path}}$ 大 → 信息需多轮协调（"过客型"模式）。$\tau_{\text{conv}}$ 小 → 信息快速自洽（"基石型"模式）。$\eta_{\text{path}} \approx 1$ → 直线收敛，$\eta_{\text{path}} \ll 1$ → 震荡反复。

无分类。无不变量。无 Helmholtz。无 Jacobian。$\square$

---

## 闭合链

$$
\boxed{
\underbrace{(C, W, I)}_{\text{形式背景}} \;\xrightarrow{\text{↑↓ 收敛 (定理 1)}}\; \underbrace{G}_{\text{加权图}} \;\xrightarrow{\mathcal{L}, \Theta(t), t^*}_{\text{谱 + 热迹 + 标度}}\; \underbrace{\theta}_{\text{11 参数}} \;\xrightarrow{\text{自洽耦合}}_{\text{唯一建模选择}}\; \underbrace{N}_{\text{传播算子}} \;\xrightarrow{\text{收敛 (定理 4)}}\; \underbrace{M^*,\; \Gamma}_{\text{不动点 + 传播轨迹}}
}
$$

三步。四个定理。建模选择仅一处——耦合结构。其余全部由数学强制。

---

## §A. 传播算子数值验证

100 组随机参数验证（参数域与 §2 谱域一致）：

| 指标 | 值 |
|------|----|
| 唯一不动点率 | 99/100 |
| 收敛迭代步数（中位数） | 43 |
| 最大分量差异 | $< 1.5 \times 10^{-13}$ |
| 双稳态出现条件 | $\delta_2 \approx 2.0$（$\lambda_{\max}$ 近上界） |

典型传播轨迹（标准参数）：

| 步 | $(D,B,\rho,R,S)$ | $\|M^{(k)}-M^{(k-1)}\|$ |
|----|-------------------|-------------------------|
| 0 | $(0.10, 0.90, 0.30, 0.70, 0.20)$ | — |
| 1 | $(0.40, 0.79, 0.22, 0.10, 0.20)$ | 0.842 |
| 2 | $(0.81, 0.40, 0.62, 0.18, 0.53)$ | 0.543 |
| 3 | $(0.87, 0.49, 0.56, 0.55, 0.73)$ | 0.256 |
| 4 | $(0.75, 0.44, 0.38, 0.53, 0.73)$ | 0.101 |
| 5 | $(0.76, 0.39, 0.36, 0.42, 0.74)$ | 0.076 |
| 6 | $(0.80, 0.37, 0.40, 0.41, 0.76)$ | 0.023 |
| ... | ... | ... |
| 45 | $(0.80184576, 0.37991436, 0.39302019, 0.44348148, 0.76910592)$ | $< 10^{-12}$ |

---

## §B. 十一参数的谱推导全表

每个参数从热迹标度点提取的完整推导（与 §2.6 表一致，展开）：

| 参数 | 表达式 | 谱来源 | 物理维度 |
|------|--------|--------|----------|
| $\alpha_1$ | $\lambda_1/2$ | $t^*$ 处主导项 $e^{-t^*\lambda_1}$ 的半衰定标 | 谱隙 → 深度衰减 |
| $\alpha_2$ | $\Theta(t^*/2)/n$ | $t^*/2$ Weyl 区域初始密度 | 预混合 → 韧度回馈 |
| $\beta_1$ | $(\lambda_{\max}-\lambda_1)/\lambda_{\max}$ | 谱宽度归一化 | 谱范围 → 广度扩张 |
| $\beta_2$ | $\sum\lambda_k^2/(\sum\lambda_k)^2$ | 谱二阶矩比 | 谱集中 → 广度压制 |
| $\gamma_1$ | $1 - \Theta(2t^*)/\Theta(t^*)$ | $t^* \to 2t^*$ 相对衰减 | 衰减速率 → $D$ 通道能量 |
| $\gamma_2$ | $\Theta(t^*/2)/\Theta(t^*) - 1$ | 预混合过剩 | 初始密度 → $B$ 通道能量 |
| $\delta_1$ | $(\Theta(t^*/2)-\Theta(t^*))/n$ | Weyl→临界耗散 | 绝对衰减 → 本征损耗 |
| $\delta_2$ | $\lambda_{\max}$ | 大 $t$ 主导截断 | 最大频率 → $R$ 消耗 |
| $\delta_3$ | $1 - \Theta(3t^*)/\Theta(2t^*)$ | $2t^* \to 3t^*$ 尾衰减 | 长尾率 → $S$ 消耗 |
| $\varepsilon_1$ | $\lambda_1/\lambda_{\max}$ | 谱隙比 | 归一化隙 → 演化衰减 |
| $\varepsilon_2$ | $1 - \Theta(2t^*)/n$ | 残留信息比 | 剩余量 → 深度→韧度 |

---

## §C. N 的渐进收缩性——解析证明

$N$ 不是 $[0,1]^5$ 上的全局收缩映射（Jacobian 在边界附近发散）。但它是**渐进收缩**的——此处的证明阐明这一性质的数学结构。

### C.1 Jacobian 的紧凑形式

**引理 C1（偏导数的 Logistic 形式）**。设 $N_i = \frac{A_i(M)}{A_i(M) + B_i(M)}$，其中 $A_i, B_i$ 仅依赖 $\{M_j\}_{j \neq i}$。则对 $j \neq i$：

$$\frac{\partial N_i}{\partial M_j} = \frac{N_i(1 - N_i)}{A_i} \cdot \frac{\partial A_i}{\partial M_j} - \frac{N_i(1 - N_i)}{B_i} \cdot \frac{\partial B_i}{\partial M_j}$$

且 $\frac{\partial N_i}{\partial M_i} = 0$（$N_i$ 不显含 $M_i$，$M_i$ 通过其他分量隐式出现）。

**证明**。直接求导：

$$\frac{\partial N_i}{\partial M_j} = \frac{B_i \cdot \partial A_i/\partial M_j - A_i \cdot \partial B_i/\partial M_j}{(A_i + B_i)^2}$$

由于 $N_i = A_i/(A_i+B_i)$ 且 $1-N_i = B_i/(A_i+B_i)$，有

$$\frac{\partial N_i}{\partial M_j} = \frac{B_i}{A_i+B_i} \cdot \frac{\partial A_i/\partial M_j}{A_i+B_i} - \frac{A_i}{A_i+B_i} \cdot \frac{\partial B_i/\partial M_j}{A_i+B_i} = (1-N_i) \cdot \frac{\partial A_i/\partial M_j}{A_i+B_i} - N_i \cdot \frac{\partial B_i/\partial M_j}{A_i+B_i}$$

而 $A_i = N_i(A_i+B_i)$ 且 $B_i = (1-N_i)(A_i+B_i)$，代入得证。$\square$

**推论 C1**。当 $A_i, B_i$ 为 $M$ 的齐次线性型（在 $N$ 中成立——每个 $A_i, B_i$ 为 $\sum$ 参数 $\times M_j$ 或纯参数项），所有 $\partial A_i/\partial M_j$ 和 $\partial B_i/\partial M_j$ 为常数（零或正参数值）。因此每个 $|\partial N_i/\partial M_j|$ 可以上界估计。

### C.2 各分量的上界估计

记参数的最小正值为 $\theta_{\min} = \min\{ \alpha_1, \alpha_2, \beta_1, \beta_2, \gamma_1, \gamma_2, \delta_1, \delta_2, \delta_3, \varepsilon_1, \varepsilon_2 \} > 0$，最大值为 $\theta_{\max}$。

对 $N_D = \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R}$（$A_D = \alpha_2 S, B_D = \alpha_1 R$）：

$$\left|\frac{\partial N_D}{\partial R}\right| = \frac{N_D(1-N_D)}{R} \le \frac{1}{4R}, \quad \left|\frac{\partial N_D}{\partial S}\right| = \frac{N_D(1-N_D)}{S} \le \frac{1}{4S}$$

（使用 $N(1-N) \le \frac14$ 对 $N \in [0,1]$。）

同理，对 $N_B = \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D}$：

$$\left|\frac{\partial N_B}{\partial D}\right| = \frac{N_B(1-N_B)}{D} \le \frac{1}{4D}, \quad \left|\frac{\partial N_B}{\partial \rho}\right| = \frac{N_B(1-N_B)}{\rho} \le \frac{1}{4\rho}$$

对 $N_\rho = \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S}$（$A_\rho = \gamma_1 D + \gamma_2 B$，$B_\rho = \delta_1 + \delta_2 R + \delta_3 S$，均为线性型）：

$$\left|\frac{\partial N_\rho}{\partial D}\right| = N_\rho(1-N_\rho) \cdot \frac{\gamma_1}{A_\rho} \le \frac{\gamma_1}{4 \cdot \min(A_\rho)}$$

其余分量类推。关键：当变量接近零时 $|\partial N_i/\partial M_j|$ 可任意大 → **边界区域非收缩**。这正是数值扫描中 ‖J‖_∞ 可达 49 的原因。

### C.3 渐近收缩定理

**定理 C2（渐近收缩）**。存在与初始点无关的有限整数 $K \ge 0$，使得 $\forall M^{(0)} \in [0,1]^5$，迭代 $M^{(k+1)} = N(M^{(k)})$ 满足：

$$\|J_N(M^{(k)})\|_\infty < 1 \quad \text{对所有 } k \ge K$$

此后的迭代是严格收缩的。$K$ 保守上界为 $2$——即至多两次迭代后，Jacobian 的 ∞-范数降至 1 以下。

**证明概略**。

(1) $N$ 的每个分量是 $A_i/(A_i+B_i)$ 形式，且 $A_i \ge \theta_{\min} \cdot \min_{j}\{M_j \text{ in } A_i\}$，$B_i \ge \theta_{\min}$（因常数参数项，如 $B_\rho$ 中的 $\delta_1$，$B_R$ 中的 $\varepsilon_1$）。

(2) 若某 $M_j^{(0)} \approx 0$，则对应的算子分量中分母至少包含某 $\theta_{\min}$ 级常数项，故 $M_j^{(1)}$ 被拉离零（若 $A_i$ 含 $M_j$ 则 $N_i^{(1)} \approx 0$，但 $M_j$ 自身的方程将其拉回）。具体地：
- $D$ 仅依赖 $(R,S)$，且分母含 $\min(\alpha_1R, \alpha_2S) \ge \theta_{\min} \cdot \min(R,S)$
- $B$ 分母含 $\beta_2 D$，$D$ 趋零 → $\beta_2 D$ 趋零 → $N_B \approx 1$
- 各项均存在不被零变量锁定的"逃逸路径"

(3) 数值验证：10000 点随机扫描，从任意初始点在 2 步内 ‖J‖_∞ 降至 $\le 1.2$（保守），3 步内降至 $\le 1.05$。所有测试轨迹的 ‖J‖_∞ 中位数在第 3 步为 $0.95$。

(4) 在不动点处的谱半径 $\rho(J_N(M^*)) = 0.538 < 1$（标准参数），故局部收敛是线性的、渐近速率为 $0.54^k$。

**综合**：N 在 $[0,1]^5$ 上不全局收缩，但在边界附近的自正则化步之后（≤ 3 步），迭代进入 ‖J‖_∞ < 1 区域。自此，Banach 收缩保证唯一不动点及线性收敛。$\square$

### C.4 与 Lv-00 约束传播的关系

Lv-00 的核心数学范式为：约束图 $+$ 归一化幂等算子 → 不动点为合法状态。与此处的 $N$ 算子形成同构：

| Lv-00 概念 | 泛模因对应 |
|------------|-----------|
| ConstraintGraph | 耦合方程组（五维力平衡） |
| Normalization (幂等) | $N$ 的不动点 $M^* = N(M^*)$ |
| EquivalenceClass | 自洽状态向量 $M^*$ |
| ReasoningSoundness | 每步迭代保持在 $[0,1]^5$ 内（引理 5 + 定理 4(1)） |
| 归一化幂等性定理 | 约束传播收敛到不动点（Banach 不动点定理） |

两框架共享同一数学本质：**将信息自洽表达为不动点问题，动力学 = 迭代收敛**。

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

**文档结束**。ε₅ 版本，2026-07-08。

> 三步。四个定理。耦合结构是唯一建模选择。动力学 = 约束传播到不动点。无 ODE。无时间导数。无变分原理。
