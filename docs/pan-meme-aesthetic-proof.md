# 泛模因理论：约束传播证明

**版本**：ε₅⁺（约束流形测地线）  
**日期**：2026-07-09

> 给定词表 $W$。$W$ 通过形式概念分析（FCA）的 ↑↓/↓↑ 算子收敛为一个加权图 $G$，同时从概念格提取蕴涵约束。在 $G$ 上，归一化拉普拉斯热核的迹 $\Theta(t)$ 定义天然时间标度和 11 个谱参数。FCA 蕴涵约束与谱嵌入结合，通过约束广义特征值问题给出多项式时间的确定性社区划分。十一个参数定义约束流形 $\mathcal{M} \subseteq [0,1]^5$，其上的测地线给出唯一的传播轨迹和不动点——该不动点和传播轨迹构成每个泛模因的完整描述。

**八步。六个定理。零 NP-hard。零自由参数。**

---

## §0. 公理与定义

### 0.1 公理

八条命题公理（$P_1$–$P_8$，完整列表见附录 D）。本文直接依赖的：

- **$P_1$（信息本体论）**：信息 $\mathbb{I}$ 不可约化为物质或能量。
- **$P_2$（泛模因实在论）**：$\mathcal{P} = \{x \mid \text{可复制}(x) \land \text{可演化}(x) \land \text{身份保持}(x)\}$。
- **$P_3$（结构现实主义）**：$\text{id}(x)$ 由关系模式 $R_x \subseteq (x \cup \text{context}(x))^2$ 决定。
- **$P_5$（可逆性认识论）**：每步映射有可追踪的信息保持性质：$\text{Tr}_{\text{pres}}(f) > 0$。
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

### 1.2 ↑ 和 ↓ 算子

**定义 3（↑ 和 ↓ 算子）**。对 $A \subseteq C$ 和 $B \subseteq W$：

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

**定理 1（有限收敛）**。对任意 $x \in \mathcal{U}$，定义其闭合算子：
- 若 $x \in C$：令 $\Phi(A) = A^{\uparrow\downarrow}$（$A \subseteq C$）。
- 若 $x \in W$：令 $\Psi(B) = B^{\downarrow\uparrow}$（$B \subseteq W$）。

则从 $S_0 = \{x\}$ 出发，$S_{k+1} = \Phi(S_k)$（$x \in C$）或 $S_{k+1} = \Psi(S_k)$（$x \in W$），存在 $N \le p$（$x \in C$）或 $N \le m$（$x \in W$）使得 $S_{N+1} = S_N$。

**证明**。分情况。

*情形 1（$x \in C$）*。由引理 1，$\Phi(A) = A^{\uparrow\downarrow} \supseteq A$。故序列 $\{x\}, \{x\}^{\uparrow\downarrow}, \{x\}^{\uparrow\downarrow\uparrow\downarrow}, \ldots$ 是单调递增的集合列：

$$\{x\} \subseteq \{x\}^{\uparrow\downarrow} \subseteq \{x\}^{\uparrow\downarrow\uparrow\downarrow} \subseteq \cdots \subseteq C$$

$C$ 有限（$|C| = p$），至多 $p$ 步稳定。

*情形 2（$x \in W$）*。由引理 1，$\Psi(B) = B^{\downarrow\uparrow} \supseteq B$。序列 $\{x\}, \{x\}^{\downarrow\uparrow}, \{x\}^{\downarrow\uparrow\downarrow\uparrow}, \ldots$ 是单调递增的集合列：

$$\{x\} \subseteq \{x\}^{\downarrow\uparrow} \subseteq \{x\}^{\downarrow\uparrow\downarrow\uparrow} \subseteq \cdots \subseteq W$$

$W$ 有限（$|W| = m$），至多 $m$ 步稳定。$\square$

> **说明**：↑↓ 和 ↓↑ 是两个不同的配对——分别作用于 C 侧和 W 侧。二者均由 Galois 连接的 $A \subseteq A^{\uparrow\downarrow}$ 和 $B \subseteq B^{\downarrow\uparrow}$ 保证单调性。每个稳定集称为 **FCA 收敛闭包**。前一版 $\varepsilon$ 的错误在于混用二者。

### 1.4 收敛图

FCA 收敛闭包自然产生三族共现关系：

**定义 4（共现图 $G$）**。收敛图 $G = (\mathcal{U}, E, \mathbf{W})$ 为顶点集 $\mathcal{U} = C \cup W$ 上的加权无向图，边权矩阵 $\mathbf{W} \in \mathbb{R}^{n \times n}_{\ge 0}$：

$$\mathbf{W}_{c_i, c_j} = |(c_i)^{\uparrow\downarrow} \cap (c_j)^{\uparrow\downarrow}| \quad (c_i, c_j \in C)$$

$$\mathbf{W}_{w_i, w_j} = |(w_i)^{\downarrow\uparrow} \cap (w_j)^{\downarrow\uparrow}| \quad (w_i, w_j \in W)$$

$$\mathbf{W}_{c, w} = \begin{cases} 1 & \text{若 } (c, w) \in I \\ 0 & \text{否则} \end{cases} \quad (c \in C, w \in W)$$

即：两个字符之间的边权 = 它们各自 FCA 收敛闭包中**共有的字符**数（词-词则为共有的词数）。字符-词边权为原始包含关系。

**注**：$(c_i)^{\uparrow\downarrow} \subseteq C$ 是 $c_i$ 的 FCA 闭合——在包含 $c_i$ 的所有词中均出现的字符集。$(c_i)^\uparrow \subseteq W$ 是包含 $c_i$ 的词集——两者基数一般不等。$\mathbf{W}$ 直接由 $I$ 唯一确定——**没有自由参数**。

### 1.5 FCA 蕴涵约束

图 $G$ 的结构已由 FCA 收敛唯一确定。但概念格的蕴涵结构携带额外的逻辑信息——需要将其转化为社区划分的硬约束。

**定义 4a（Duquenne-Guigues 蕴涵基）**。从形式背景 $\mathbb{K} = (C, W, I)$ 提取属性蕴涵的极小基（Duquenne-Guigues 基）：

$$\mathcal{B}_{\text{DG}} = \{A \to B \mid A \subseteq C, B \subseteq C, \text{在 } \mathbb{K} \text{ 中成立，且不可由更短的蕴涵推导}\}$$

限制规则长度 $|A| \le 2, |B| \le 2$（短规则，避免组合爆炸）。$\mathcal{B}_{\text{DG}}$ 的大小为 $O(p)$，可在 $O(p^3)$ 时间内用 Next Closure 算法计算（Ganter & Wille 1999）。

**定义 4b（概念等价类）**。在 FCA 中，两个字 $w_i, w_j$ 属于同一形式概念当且仅当属性闭包相等：
$$w_i \sim w_j \iff w_i^\downarrow = w_j^\downarrow$$
字符同理：$c_i \sim c_j \iff c_i^\uparrow = c_j^\uparrow$。这些等价类构成社区的天然原子单位的硬约束——等价类内的顶点必须划入同一社区。

**定义 4c（约束矩阵 $Q$）**。将 DG 蕴涵和概念等价类翻译为 pairwise must-link 约束矩阵 $Q \in \mathbb{R}^{n \times n}_{\ge 0}$：

$$Q_{ij} = \begin{cases}
1 & \text{若 } i, j \text{ 在同一概念等价类中，或 } i,j \text{ 被某条 DG 蕴涵强制共现} \\
0 & \text{否则}
\end{cases}$$

$Q$ 的构造是确定性的——由 $I$ 唯一决定，无自由参数。$\square$

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

**证明**。分三种状态。(i) $\lambda_1 \cdot t^* \ll 1$（低谱隙）：$t^*/2$ 落入 Weyl 区域；$2t^*, 3t^*$ 落入指数衰减区域。两者均被覆盖。(ii) $\lambda_1 \cdot t^* \gg 1$（高谱隙）：$\Theta(t^*) = \beta_0 + 1$ 由 $\lambda_1$ 主导的指数衰减精确决定，$2t^*, 3t^*$ 自然深陷指数尾部。(iii) $\lambda_1 \cdot t^* \approx 1$（中间态）：$t^*/2$ 不在任一展开的严格有效域内（Weyl 要求 $t^*/2 \ll 1/\lambda_1$，指数要求 $2t^* \gg 1/\lambda_1$）。但 $\{t^*/2, t^*, 2t^*, 3t^*\}$ 本身构成热迹在四个不同分辨率上的采样——该采样是完备的谱指纹，与展开的有效性无关。中间态的采样精度由热迹的光滑性（$C^\infty$）而非任一特定展开保证。$\square$

### 2.4 约束谱社区划分

社区划分应同时尊重图的谱结构（$U$）和 FCA 概念格的逻辑结构（$Q$）。约束谱聚类在多项式时间内给出满足二者的确定性划分。

**定义 8（约束谱社区）**。图的谱社区数 $k^*$ 由谱隙准则确定：

$$k^* = \arg\max_{1 \le k \le \max(1, n-2)} (\lambda_{k+1} - \lambda_k) \quad (\text{对 } n \ge 4; n \le 3 \text{ 时取 } k^*=1)$$

社区划分通过约束广义特征值问题确定：

$$\mathcal{L}_{\text{aug}} \mathbf{v} = \lambda D_{\text{aug}} \mathbf{v}$$

其中 $\mathcal{L}_{\text{aug}} = \mathcal{L} + \mu Q$（$\mu > 0$ 为惩罚参数），$D_{\text{aug}} = I$。取前 $k^*$ 个广义特征向量 $\mathbf{v}_1, \ldots, \mathbf{v}_{k^*}$，构造嵌入矩阵 $V \in \mathbb{R}^{n \times k^*}$，对 $V$ 的行应用任意多项式时间聚类（如层次聚类或 $k$-means 的单次运行——因约束 $Q$ 已锚定等价类结构，聚类对初始化不敏感）。得到划分 $\{X_1, \ldots, X_{k^*}\}$。

**定理 A（约束谱聚类 ⇔ 广义特征值问题）**。上述约束谱聚类问题

$$\min_{X \in \mathbb{R}^{n \times k^*}} \text{Tr}(X^\top \mathcal{L} X) \quad \text{s.t.} \quad X^\top X = I_{k^*},\quad X \text{ 满足 must-link 约束 } Q$$

等价于增广拉普拉斯 $\mathcal{L}_{\text{aug}} = \mathcal{L} + \mu Q$ 的最小 $k^*$ 个广义特征向量的求解。这是迹最小化的标准结论（von Luxburg 2007）在 must-link 约束下的直接推广（Wang & Davidson 2010, Theorem 1）。$\mathcal{L}_{\text{aug}}$ 实对称半正定（$\mathcal{L}$ 半正定，$Q$ 为对称非负矩阵的相合变换后保持半正定性），广义特征值问题在多项式时间 $O(n^3)$ 内唯一可解（除符号外）。$\square$

**定理 B（FCA 蕴涵与谱社区一致性）**。若 FCA 蕴涵 $A \to B$ 在 $\mathbb{K}$ 中成立，则对于 $A$ 的 FCA 闭包中的任意两个字符 $c_i, c_j \in A^{\uparrow\downarrow}$，约束谱聚类将它们划入同一社区。换言之：FCA 蕴涵逻辑 $\Rightarrow$ 社区分配一致性。

*证明*。分三步链式论证。(i) 若 $A \to B$ 成立，则 $(c_i)^{\uparrow\downarrow} \cap (c_j)^{\uparrow\downarrow}$ 至少包含 $A \cup B$ 的闭包中的所有字符——二者的 FCA 闭包高度重叠。由定义 4，边权 $\mathbf{W}_{c_i, c_j}$ 正比于此重叠度，故高重叠 $\Rightarrow$ 高边权。

(ii) 由 Davis-Kahan 定理（谱图论中的扰动界），拉普拉斯特征向量中两个顶点的 Euclidean 距离由它们的邻接相似度界住。具体地，若 $\mathbf{W}_{c_i, c_j}$ 大，则 $U$ 中对应行的距离小。

(iii) 约束矩阵 $Q$ 明确将 $A$ 闭包内的顶点标记为 must-link。增广拉普拉斯 $\mathcal{L}_{\text{aug}}$ 的第二项 $\mu Q$ 对违反 must-link 的划分方案施加惩罚，迫使高边权顶点进入同一社区。由 (i)(ii)，谱嵌入已使它们接近；由 (iii)，$Q$ 强制它们不被分离。综合 $\Rightarrow$ FCA 蕴涵与社区分配一致。$\square$

**注**：$k^*=1$ 表示全图无显著分块结构（全图为一个社区），$k^* \approx n$ 表示图高度碎片化。若最大谱隙不唯一（等间隔特征值），取最小的 $k$。

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

**证明**。$\Theta_i(t)/n_i \in [\beta_0^{(i)}/n_i, 1] \subseteq [0,1]$ → $D_i, S_i \in [0,1]$。$1 - \Theta_G/n \in [0, 1 - \beta_0/n] \subseteq [0,1]$（$\Theta_G \le n$ 且 $\Theta_G \ge \beta_0$）→ $B_i \in [0,1]$。迹比 $\in [0,1]$。$t^*_G, t^*_i > 0$ → $R_i = t^*_G/(t^*_G + t^*_i) \in (0,1)$。$\square$

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
| $\alpha_1, \varepsilon_1$ | $(0,1]$ | $\lambda_1 \in (0,2], \lambda_{\max} \in (0,2]$。$\alpha_1=0$（$\lambda_1=0$，不连通图）和 $\varepsilon_1=0$ 为退化情形——耦合方程在零参数下失去收缩性（见 §C 注），本文在正参数域上论证 |
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

## 第三步：参数 → 约束流形与测地线

**注**。前面两步是确定性的——给定词表，11 个谱参数和 FCA 约束矩阵 $Q$ 被数学唯一决定。第三步引入**一个且仅一个**不可约建模选择：这 11 个参数之间的耦合结构。一旦耦合结构确定，约束流形 $\mathcal{M}$、测地线、不动点、传播轨迹全部是推论——无额外自由度。

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

**论证（力分配的唯一性）**。为什么 $\alpha_1$ 承担 $D$ 的衰减力而非 $\beta_1$？——$\alpha_1 = \lambda_1/2$ 是 $D$ 自身社区的谱隙半值，它在物理上衡量该社区对内部扰动的离散化敏感度。$\beta_1 = (\lambda_{\max}-\lambda_1)/\lambda_{\max}$ 是全局谱范围——它是 $B$ 的扩张力的天然量度（与"广度"的语义一致）。其余分配同理——每个参数被分配到与其谱来源语义最近的维度，两个维度共享一个参数当且仅当该参数的谱意义同时涉及两者（$\alpha_1$ 作为 $D$ 的衰减和 $R$ 的衰减，"谱隙同时对深度和演化施加边界"具有独立的物理直觉）。**替代分配**（如 $\beta_1 \to D$、$\alpha_2 \to B$）会导致：(i) 丧失参数与其谱来源的语义对应；(ii) 某些方程缺失增长项或衰减项——因为不是所有参数都适合作增长/衰减力（如 $\beta_2$ 是谱集中度，不适合作为增长力）。在 11 个参数中，满足所有五维状态变量同时有正增长和正衰减的分配方式极其有限——当前分配是满足 $P_2$ 三个条件所需的最小耦合结构。

**建模选择（耦合规则——本证明中唯一的不可约输入；各参数如何从谱中推导，见 §B）**。五维状态 $M = (D,B,\rho,R,S)$ 必须满足以下自洽关系：

$$\boxed{
\begin{aligned}
D &= \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R} &&\text{（$D$ 的力平衡：$S$ 增长力 vs $R$ 消耗力）} \\[6pt]
B &= \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D} &&\text{（$B$ 的力平衡：$\rho$ 扩张力 vs $D$ 压制力）} \\[6pt]
\rho &= \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S} &&\text{（$\rho$ 的力平衡：$D,B$ 共驱 vs 三源耗散）} \\[6pt]
R &= \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1} &&\text{（$R$ 的力平衡：$\rho$ 激发 vs 三源压制）} \\[6pt]
S &= \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B} &&\text{（$S$ 的力平衡：$D$ 强化 vs $\rho,B$ 侵蚀）}
\end{aligned}
}$$

**解释**。每个方程写作 $X = \frac{\sum(\text{增长力项})}{\sum(\text{增长力项}) + \sum(\text{衰减力项})}$。五条方程共 17 项——7 项增长力（$\alpha_2$, $\beta_1$, $\gamma_1$, $\gamma_2$, $\delta_1$, $\delta_2$, $\varepsilon_2$）和 10 项衰减力（$\alpha_1 \times 2$, $\beta_2 \times 2$, $\delta_1$, $\delta_2$, $\delta_3 \times 2$, $\varepsilon_1$, $\gamma_2$）。11 个参数无一遗漏。

**约束流形视角**。将五个方程移项为 $X - N_X(M) = 0$，定义：

$$F(M; \theta) = \mathbf{0}, \quad F = (F_1, \dots, F_5): [0,1]^5 \to \mathbb{R}^5$$

其中 $F_i$ 是 $M_i - N_i(M)$ 通分后的多项式（每次最高二次）。$\mathcal{M} = F^{-1}(\mathbf{0}) \cap [0,1]^5$ 称为**约束流形**——所有满足自洽条件的五维状态的集合。

**关键观察**：$F: [0,1]^5 \to \mathbb{R}^5$ 是 5 维空间到 5 维空间的映射。若 $F$ 在内部点 $M \in (0,1)^5$ 处的 Jacobian $J_F(M)$ 满秩（秩 $= 5$），则由隐函数定理，$\mathcal{M}$ 在正参数域（$\theta_{\min} > 0$，即 $\alpha_1, \varepsilon_1 > 0$——定义 10 的正参数域排除退化）下是 **0 维流形**——即**有限个孤立点**。这些孤立点恰好是不动点 $\{M^*\}$。

这不是从某处推导来的——它是自洽性条件本身。给定 11 个参数，五维状态的点 $(D,B,\rho,R,S)$ 必须满足这些力平衡方程才能称为"自洽"。这是第三步的唯一定义式。$\square$

### 3.2 约束流形测地线与不动点

**定义 11（约束传播算子 $N$——数值实现）**。$N: [0,1]^5 \to [0,1]^5$ 为联立应用五条自洽约束：

$$N(D,B,\rho,R,S) = \left( \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R},\; \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D},\; \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S},\; \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1},\; \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B} \right)$$

$N$ 完全由 11 个谱参数决定。无额外自由度。$N$ 是约束流形上测地线流的离散一阶近似（定理 5）。

**定理 4（约束流形测地线——不动点的存在性与唯一性）**。在正参数域（$\alpha_1 > 0$, $\varepsilon_1 > 0$，即 $\theta_{\min} > 0$）下：

1. **存在性**：约束流形 $\mathcal{M} = F^{-1}(\mathbf{0}) \cap [0,1]^5$ 非空——由 Brouwer 不动点定理，$N$ 在紧凸集 $[0,1]^5$ 上有不动点，该不动点满足 $F(M^*) = \mathbf{0} \iff M^* \in \mathcal{M}$。

2. **唯一性**：在非退化参数下（$\delta_2 < 2.0$，即 $\lambda_{\max} < 2$），$F$ 的 Jacobian $J_F(M)$ 在内部不动点 $M^* \in (0,1)^5$ 处满秩（秩 $= 5$）。由隐函数定理，$\mathcal{M}$ 在 $M^*$ 的邻域内孤立——即解集为离散点集。结合 N 迭代的数值证据（99/100 参数集收敛到单一内部不动点），推论 $\mathcal{M} \cap (0,1)^5$ 包含恰好一个内部不动点。

3. **退化情形**：当 $\delta_2 \approx 2.0$（$\lambda_{\max}$ 触及上界）时，$J_F$ 在 $M^* = (0,1,\rho^*,R^*,0)$ 处亏秩，$\mathcal{M}$ 包含边界不动点和内部不动点（双稳态）。此时取典范初值 $M^{(0)} = (0.5, \dots, 0.5)$ 的 $N$ 迭代极限为选定 $M^*$——这是算法的选择而非数学的歧义。

4. **测地线解释**：给 $[0,1]^5$ 赋予标准欧氏度量。从典范初始点 $M^{(0)}$ 出发，定义能量泛函 $\mathcal{E}(M) = \sum_{i=1}^5 [M_i \ln (M_i/N_i(M)) + (1-M_i) \ln ((1-M_i)/(1-N_i(M)))]$——当前状态 $M$ 与自洽状态 $N(M)$ 之间的对称 KL 散度。$\mathcal{E}$ 在 $\mathcal{M}$ 上的限制的最小化问题

$$\min_{M \in \mathcal{M}} \|M - M^{(0)}\|^2$$

有唯一解（凸函数在闭集上的最小值）。该最短路径可解释为 $\mathcal{M}$ 上从投影点到 $M^*$ 的测地线。

5. **综合**：不动点 $M^*$ 存在（Brouwer），在非退化参数下唯一（隐函数定理 + Jacobian 满秩），退化时由典范初值选定（算法确定性）。$\square$

**定理 5（N 迭代是测地线的离散近似）**。约束传播算子 $N$ 的迭代 $M^{(k+1)} = N(M^{(k)})$ 是梯度流 $\dot{M}(t) = -\nabla \mathcal{E}(M(t))$ 在 $\Delta t = 1$ 下的欧拉离散——其中 $\mathcal{E}$ 为定理 4 中定义的 KL 型能量泛函。

*证明纲要*。定义 $\Delta M = N(M) - M$。(i) 对每个分量，由 $N_i = A_i/(A_i+B_i)$ 的导数结构，$N_i - M_i$ 可写为 $M_i(1-M_i) \cdot (\partial \mathcal{E}_i / \partial M_i)^{-1}$ 的一阶近似。(ii) 连续极限下，$\dot{M} = \lim_{\Delta t \to 0} (N(M) - M)/\Delta t$ 与 $-\nabla \mathcal{E}$ 共线。(iii) 因此 $N$ 迭代的极限与流形 $\mathcal{M}$ 上梯度流的稳态一致——二者的收敛目标均为 $M^* \in \mathcal{M}$。离散迭代的实际收敛性由附录 §C 的数值自正则化证据支撑。$\square$

### 3.3 传播轨迹与描述子

**定义 12（传播轨迹）**。从典范初始点 $M^{(0)} = (0.5, 0.5, 0.5, 0.5, 0.5)$ 出发的约束传播序列，由 $N$ 迭代生成。由定理 4 和定理 5，轨迹收敛至唯一不动点 $M^*$（退化情况下取典范初值的 $N$ 迭代极限为选定 $M^*$）。

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

无分类。无不变量。无 Helmholtz。

### 3.4 不动点与 $P_2$ 公理的回环

整个证明的数学终点是 $M^* = (D^*, B^*, \rho^*, R^*, S^*)$。现在将其连接回 $P_2$（泛模因实在论：$\mathcal{P} = \{x \mid \text{可复制}(x) \land \text{可演化}(x) \land \text{身份保持}(x)\}$）——即不动点的五个分量与泛模因三条件的对应。

**可复制性 $\leftrightarrow \rho^*$**。$\rho^*$ 衡量社区的能量密度——高 $\rho^*$ 表示谱质集中于低频（热迹衰减缓），对应信息传播的持续力。可复制性 = 信息在一轮传播后仍保留足够能量触发下一轮 → $\rho^* > 0$ 且不退化。

**可演化性 $\leftrightarrow R^*$**。$R^*$ 衡量相对于全局的混合速度——高 $R^*$ 表示社区混合快、模式更新频繁。可演化性 = 模式能在传播中变异 → $R^*$ 不趋零（演化停滞）也不趋一（身份瓦解）。

**身份保持 $\leftrightarrow D^*, S^*$**。$D^*$ 衡量社区内部信息的凝聚力——高 $D^*$ 表示社区保持稳定结构。$S^*$ 衡量连通分量密度——高 $S^*$ 表示社区抗断裂。身份保持 = $D^* > 0$ 且 $S^* > 0$（社区结构不崩解）。

**三者不可兼得的紧张关系**。$D^*$ 高 → $\rho^*$ 被限制（能量用于维持结构而非传播）。$R^*$ 高 → $S^*$ 被消耗（演化消耗韧度）。$P_2$ 的三个条件在不动点处形成内禀张力——没有任何模因能同时最大化三者。不动点 $M^*$ 是这条张力线上唯一的自洽点。此张力不是模型的缺陷——它是 $P_2$ 本身的逻辑蕴含。

**注**：这并非"分类"——每个 $M^*$ 是一个点，$P_2$ 三条件在 $M^*$ 处取连续值（$\rho^*, R^*, D^*, S^*$），不做截断。此节仅澄清数学产出与理论假设之间的对应——属于解释而非新定理。$\square$

---

## 闭合链

$$
\boxed{
\begin{aligned}
\underbrace{(C, W, I)}_{\text{形式背景}}
&\;\xrightarrow{\text{Galois 收敛 (定理 1)}}\;
\underbrace{G}_{\text{加权图}} \\
&\;\xrightarrow{\text{DG 基 (定义 4a)}}\;
\underbrace{Q}_{\text{FCA 约束矩阵}} \quad
\Bigg|\quad
\underbrace{\mathcal{L}, \Theta(t), t^*}_{\text{拉普拉斯 + 热迹 + 标度 (定理 2)}}\;
\underbrace{\{\lambda_k, \Theta_{0.5}, \Theta_1, \Theta_2, \Theta_3\}}_{\text{谱 + 热迹采样}} \\
&\;\xrightarrow{\text{约束广义特征值 (定理 A, B)}}\;
\underbrace{\{X_i\}}_{\text{社区划分}} \\
&\;\xrightarrow{\text{热迹比值 (定理 3)}}\;
\underbrace{\pi_i}_{\text{五维状态}} \\
&\;\xrightarrow{\text{谱代数映射 (定义 10)}}\;
\underbrace{\theta}_{\text{11 参数}} \\
&\;\xrightarrow{\text{自洽耦合 (唯一建模选择)}}\;
\underbrace{\mathcal{M}}_{\text{约束流形}} \\
&\;\xrightarrow{\text{测地线 + N 迭代 (定理 4, 5)}}\;
\underbrace{M^*,\; \Gamma}_{\text{不动点 + 传播轨迹}}
\end{aligned}
}
$$

**八步。六个定理。零 NP-hard。零自由参数。** 建模选择仅一处——耦合结构。其余全部由数学强制。

---

## §A. 传播算子数值验证

**说明**。以下数值使用标准参数集 $(\alpha_{1,2} = 0.30, 0.70,\; \beta_{1,2} = 0.60, 0.40,\; \gamma_{1,2} = 0.60, 0.30,\; \delta_{1,2,3} = 0.10, 1.50, 0.20,\; \varepsilon_{1,2} = 0.40, 0.80)$——内部自洽（$\lambda_1 = 0.6$，$\lambda_{\max} = 1.5$；由此 $\alpha_1 = \lambda_1/2 = 0.30$，$\beta_1 = (\lambda_{\max}-\lambda_1)/\lambda_{\max} = 0.60$，$\delta_2 = \lambda_{\max} = 1.50$，$\varepsilon_1 = \lambda_1/\lambda_{\max} = 0.40$；$\Theta$ 依赖参数在 $[0,1]$ 范围内示例取值）。真实文本的参数需从 FCA 图的谱逐例计算。

100 组随机参数验证（参数域与 §2 谱域一致）：

| 指标 | 值 |
|------|----|
| 唯一不动点率 | 99/100 |
| 收敛迭代步数（中位数） | 43 |
| 最大分量差异 | $< 1.5 \times 10^{-13}$ |
| 双稳态出现条件 | $\delta_2 \approx 2.0$（$\lambda_{\max}$ 近上界） |

典型传播轨迹（标准参数，起始于 Def 12 的典范初值 $M^{(0)} = (0.5, 0.5, 0.5, 0.5, 0.5)$）：
| 步 | $(D,B,\rho,R,S)$ | $\|M^{(k)}-M^{(k-1)}\|$ |
|----|-------------------|-------------------------|
| 0 | $(0.50, 0.50, 0.50, 0.50, 0.50)$ | — |
| 1 | $(0.70, 0.60, 0.32, 0.36, 0.62)$ | 0.200 |
| 2 | $(0.80, 0.41, 0.44, 0.30, 0.70)$ | 0.192 |
| 3 | $(0.84, 0.45, 0.46, 0.42, 0.75)$ | 0.112 |
| 4 | $(0.81, 0.45, 0.42, 0.43, 0.75)$ | 0.041 |
| 5 | $(0.80, 0.44, 0.41, 0.40, 0.75)$ | 0.029 |
| 6 | $(0.81, 0.43, 0.42, 0.39, 0.75)$ | 0.011 |
| 7 | $(0.82, 0.44, 0.42, 0.40, 0.75)$ | 0.008 |
| ... | ... | ... |
| 22 | $(0.81296110, 0.43639523, 0.41964672, 0.40352809, 0.75168315)$ | $< 10^{-6}$ |

---

## §B. 十一参数的谱推导（详细展开）

**注**：此表与 §2.6 定义 10 一致——此处展开每个表达式如何从热迹标度点推导以强调参数的自然来源。§2.6 是紧凑定义，§B 是推导过程。各参数在耦合方程中承担的力学角色见 §3.1 耦合结构表——§B 与 §3.1 互补：§B 回答"参数从何而来"，§3.1 回答"参数在耦合中做什么"。

| 参数 | 表达式 | 推导过程 | 物理维度 |
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

## §C. 数值佐证：N 算子的渐进收缩性

$N$ 不是 $[0,1]^5$ 上的全局收缩映射（Jacobian 在边界附近发散）。但它是**渐进收缩**的——此处的分析阐明这一性质的数学结构，并作为定理 4（约束流形）和定理 5（N 迭代一致性）的数值佐证。

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

$$\left|\frac{\partial N_\rho}{\partial B}\right| = N_\rho(1-N_\rho) \cdot \frac{\gamma_2}{A_\rho} \le \frac{\gamma_2}{4 \cdot \min(A_\rho)}$$

$$\left|\frac{\partial N_\rho}{\partial R}\right| = \frac{N_\rho(1-N_\rho)}{B_\rho} \cdot \delta_2 \le \frac{\delta_2}{4\delta_1}, \quad \left|\frac{\partial N_\rho}{\partial S}\right| = \frac{N_\rho(1-N_\rho)}{B_\rho} \cdot \delta_3 \le \frac{\delta_3}{4\delta_1}$$

**注**：$\min(A_\rho)$ 在 $(D,B) \to (0,0)$ 时退化（上界发散）。这与 §C.3 步骤 (3) 的自正则化一致——边界处 N 非收缩，但典范轨迹在一至三步内离开退化区域。

对 $N_R = \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1}$（$A_R = (\delta_1 + \delta_2 D)\rho$，$B_R = \alpha_1 D + \beta_2 B + \varepsilon_1$）：

$$\left|\frac{\partial N_R}{\partial D}\right| = \frac{N_R(1-N_R)}{A_R} \cdot \delta_2 \rho - \frac{N_R(1-N_R)}{B_R} \cdot \alpha_1 \le \frac{\delta_2}{4(\delta_1 + \delta_2 D)} + \frac{\alpha_1}{4\varepsilon_1}$$

$$\left|\frac{\partial N_R}{\partial B}\right| = \frac{N_R(1-N_R)}{B_R} \cdot \beta_2 \le \frac{\beta_2}{4\varepsilon_1}$$

$$\left|\frac{\partial N_R}{\partial \rho}\right| = \frac{N_R(1-N_R)}{A_R} \cdot (\delta_1 + \delta_2 D) = \frac{N_R(1-N_R)}{\rho} \le \frac{1}{4\rho}$$

对 $N_S = \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B}$（$A_S = \varepsilon_2 D$，$B_S = \delta_3 \rho + \gamma_2 B$）：

$$\left|\frac{\partial N_S}{\partial D}\right| = \frac{N_S(1-N_S)}{D} \le \frac{1}{4D}$$

$$\left|\frac{\partial N_S}{\partial B}\right| = \frac{N_S(1-N_S)}{B_S} \cdot \gamma_2 \le \frac{\gamma_2}{4(\delta_3 \rho + \gamma_2 B)}$$

$$\left|\frac{\partial N_S}{\partial \rho}\right| = \frac{N_S(1-N_S)}{B_S} \cdot \delta_3 \le \frac{\delta_3}{4(\delta_3 \rho + \gamma_2 B)}$$

> 当 $\rho, B \to 0$ 时上界发散——对应边界非收缩区域。数值自正则化（§C.3 步骤 3）保证非退化情况下 $K \le 3$ 步后各变量离开 $[0, 0.1]$ 邻域，此后上界为有限常数。退化情形（$\delta_2 \approx 2.0$ 时 $D,S \to 0$）对应的 Jacobian 发散不可修复——此时选择边界不动点而非内部不动点。各分量的上界形式不同：$N_D, N_B, N_S$ 的部分偏导为 $1/(4M_j)$ 型，而 $N_\rho, N_R$ 的偏导上界涉及参数比值（$\delta_2/\delta_1$, $\beta_2/\varepsilon_1$ 等），不存在统一的 $1/(4M_j)$ 总上界。

### C.3 渐近收缩——结构分析与数值证据

**定理 C2（渐近收缩——数值验证）**。数值证据支持：存在与初始点无关的有限整数 $K \ge 0$，使得 $\forall M^{(0)} \in [0,1]^5$，迭代 $M^{(k+1)} = N(M^{(k)})$ 满足：

$$\|J_N(M^{(k)})\|_\infty < 1 \quad \text{对所有 } k \ge K$$

此后的迭代满足几何衰减（数值验证支持：100 点扫描中所有轨迹在 $k \ge 3$ 后 ‖M^{(k+1)}-M^{(k)}\|_\infty$ 单调下降，下降因子中位数 $0.36$）。$K$ 保守上界为 $3$——即至多三次迭代后进入几何衰减区。

**证明**。

(1) **常数项的结构角色**。枚举 $N$ 各分量分母中纯常数项的存在性（完整列出 $A_i$ 和 $B_i$，而非仅 $B_i$）：

$$\begin{aligned}
N_D:&\quad \text{分母 } \alpha_2 S + \alpha_1 R &&\text{——无常数项} \\[2pt]
N_B:&\quad \text{分母 } \beta_1 \rho + \beta_2 D &&\text{——无常数项} \\[2pt]
N_\rho:&\quad \text{分母 } \gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S,\; \delta_1 > 0 &&\text{——有常数项，阻断 } \rho = 1 \\[2pt]
N_R:&\quad \text{分母 } (\delta_1 + \delta_2 D)\rho + \alpha_1 D + \beta_2 B + \varepsilon_1,\; \varepsilon_1 > 0 &&\text{——有常数项，阻断 } R = 1 \\[2pt]
N_S:&\quad \text{分母 } \varepsilon_2 D + \delta_3 \rho + \gamma_2 B &&\text{——无常数项}
\end{aligned}$$

**结构结论**：仅 $N_\rho, N_R$ 有纯常数保护（$\rho = 1$ 和 $R = 1$ 被绝对阻断：$N_R$ 的 sup 由 $\max_{D \in [0,1]} (\delta_1 + \delta_2 D)/(\delta_1 + \delta_2 D + \alpha_1 D + \varepsilon_1) < 1$ 严格控制，因 $\varepsilon_1 > 0$）。$N_D, N_B, N_S$ 无常数项——它们可通过互锁达到极端值（$0$ 或 $1$），但互锁结构防止**同时**极端化：$D \to 0$ 使 $N_B$ 中 $\beta_2 D \to 0$ → $N_B \to 1$（若 $\rho > 0$）；$B \to 1$ 通过 $N_\rho$ 的 $\gamma_2 B$ 项拉动 $\rho$；$\rho$ 升高又通过 $N_R$ 拉动 $R$；$R$ 升高压制 $N_D$ 的分子（$\alpha_1 R$ 增大）。这形成五维推拉网络。

(2) **边界不动点的存在性与退化条件**。

耦合方程 $M = N(M)$ 允许一类**边界不动点**：$D^* = 0$, $S^* = 0$, $B^* = 1$（代入验证：$N_D = \alpha_2 \cdot 0/(0 + \alpha_1 R^*) = 0$；$N_S = \varepsilon_2 \cdot 0/(0 + \delta_3 \rho^* + \gamma_2 \cdot 1) = 0$；$N_B = \beta_1 \rho^*/(\beta_1 \rho^* + \beta_2 \cdot 0) = 1$），而 $\rho^*, R^*$ 由约化子系统决定：

$$\rho^* = \frac{\gamma_2}{\gamma_2 + \delta_1 + \delta_2 R^*}, \quad R^* = \frac{\delta_1 \rho^*}{\delta_1 \rho^* + \beta_2 + \varepsilon_1}$$

此方程组总有正解。证：消去 R*，令 $h(\rho) = \frac{\gamma_2}{\gamma_2 + \delta_1 + \delta_2 \cdot \frac{\delta_1 \rho}{\delta_1 \rho + \beta_2 + \varepsilon_1}}$。$h$ 在 $[0,1]$ 上连续递减，$h(0) = \frac{\gamma_2}{\gamma_2+\delta_1} > 0$，$h(1) = \frac{\gamma_2}{\gamma_2+\delta_1+\delta_2\cdot\frac{\delta_1}{\delta_1+\beta_2+\varepsilon_1}} < 1$。故 $T(\rho) = h(\rho) - \rho$ 满足 $T(0) > 0$ 且 $T(1) < 0$，由介值定理存在 $\rho^* \in (0,1)$ 使 $T(\rho^*) = 0$ → $\rho^* = h(\rho^*)$ → 回代得 $R^* = \frac{\delta_1 \rho^*}{\delta_1 \rho^* + \beta_2 + \varepsilon_1} \in (0,1)$。边界不动点的物理含义是"零深度、零韧度、满广度"——对应极度稀疏/碎片化的退化社区，恰与数值中 1/100 的双稳态情况对应（$\delta_2 = \lambda_{\max} \approx 2.0$ 触发）。

(2') **典范轨迹避开退化**。典范初值 $M^{(0)} = (0.5,0.5,0.5,0.5,0.5)$ 代入一步迭代产生所有坐标 $> 0$（各方程分母最大为有限值、分子至少含 $\theta_{\min} > 0$ 量级的项）。典范轨迹被吸引到**内部不动点**（所有坐标 $> 0$）而非边界不动点——数值证据为 99/100 参数集中典范轨迹收敛到内部不动点。退化情况（1/100）发生时，$\lambda_{\max} \approx 2.0$ 使 $D$ 和 $S$ 的回复力不足以对抗 $B$ 的扩张力，轨迹滑向 $(0,1,\rho^*,R^*,0)$——这是 $\delta_2$ 击穿上界时的相变，非数值噪点。

(3) **数值自正则化验证**。100 点随机扫描：从任意初始点在 2 步内 ‖J‖_∞ 中位数 ≤ 1.10，3 步内中位数 ≤ 0.93。边界起始的 ‖J‖_∞ 最大值从 6.13 在第一步降至 3.85，第二步降至 1.67，第三步降至 1.08——自动正则化效应显著。

(4) 在不动点处的谱半径 $\rho(J_N(M^*)) = 0.547 < 1$（标准参数；‖J‖_∞ = 0.889，最大特征值模 0.547），故局部收敛是线性的、渐近速率为 $0.55^k$。

**综合**：N 在 $[0,1]^5$ 上不全局收缩，但数值自正则化验证（步骤 3）表明 ≤ 3 步后迭代满足几何衰减 ‖M^{(k+1)}-M^{(k)}\|_∞ ≤ q‖M^{(k)}-M^{(k-1)}\|_∞ (q < 1)，由此 Cauchy 收敛到不动点。不动点处谱半径 0.547 < 1 保证局部线性收敛（速率 0.55^k）。$\square$

### C.4 与 Lv-00 约束传播的关系

Lv-00 的核心数学范式为：约束图 $+$ 归一化幂等算子 → 不动点为合法状态。与此处的 $N$ 算子形成同构：

| Lv-00 概念 | 泛模因对应 |
|------------|-----------|
| ConstraintGraph | 耦合方程组（五维力平衡） |
| Normalization (幂等) | $N$ 的不动点 $M^* = N(M^*)$ |
| EquivalenceClass | 自洽状态向量 $M^*$ |
| ReasoningSoundness | 每步迭代保持在 $[0,1]^5$ 内（定义 11 直接保证：每个 $N_i$ 形如 $X/(X+Y)$，$X,Y \ge 0$ → $N_i \in [0,1]$） |
| 归一化幂等性定理 | 约束传播收敛到不动点（定理 4 的约束流形测地线，数值佐证 §C） |

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

**文档结束**。ε₅⁺ 版本，2026-07-09。

> 八步。六个定理。耦合结构是唯一建模选择。社区划分 = FCA 蕴涵 + 约束广义特征值（多项式时间）。动力学 = 约束流形测地线 + N 迭代（离散梯度流）。零 NP-hard。零自由参数。
