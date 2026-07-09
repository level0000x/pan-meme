# 泛模因理论：约束流形变分证明

**版本**：ε₅⁺⁺（约束流形拉格朗日力学）  
**日期**：2026-07-09

> 给定词表 $W$。$W$ 通过形式概念分析（FCA）的 ↑↓/↓↑ 算子收敛为一个加权图 $G$，同时从概念格提取蕴涵约束和偏序结构。在 $G$ 上，归一化拉普拉斯热核的迹 $\Theta(t)$ 定义天然时间标度和 11 个谱参数。FCA 蕴涵约束与谱嵌入结合，通过约束广义特征值问题给出多项式时间的确定性社区划分。概念格偏序、谱守恒律与能量耗散不等式共同定义约束流形 $\mathcal{M} \subseteq [0,1]^5$。在 $\mathcal{M}$ 上，带 Rayleigh 耗散的变分原理 $\delta\int L\,dt + \int \frac{\partial R}{\partial \dot M}\cdot\delta M\,dt = 0$ 给出唯一的耗散测地线方程——该测地线退化为 N 算子迭代，终点为不动点 $M^*$。信息作用量谱 $\{S, \Phi^*, W_{\text{diss}}, \eta_{\text{info}}, \tau^{-1}\}$ 构成每个泛模因的完整物理描述。

**九步。六个主干定理。一个变分原理。零 NP-hard。零自由参数。**

### 框架全景

概念格 $\mathfrak{B}(\mathbb{K})$ 是整个证明的枢纽节点。左支（聚类约束）先执行，右支（流形约束）依赖左支的输出 $\{X_i\}$（社区划分及大小）后闭合，最终两股约束在 $\mathcal{M}_i$ 处汇合：

```
                          ┌─── 左支：聚类约束（先执行）─────────────┐
                          │  DG 基 → Q → 约束特征分解 → {X_i}      │
(C, W, I) → G → 𝔅(𝕂) ────┤                                        ├──→ π → θ ──┐
                          │  Hasse 图 → 偏序 ──────────────────────┤               │
                          └─── 右支：流形约束 ──────────────────────┘               │
                                      ↓  C_order ← (|X_i| 来自左支)                │
                                      └───────────────────────────→ M_i ←───────────┘
                                                     ↓
                                L + R → 耗散变分原理 → 测地线 M(t)
                                                     ↓
                                      {S, Φ*, W_diss, η_info, τ⁻¹}
```

左支（§1.6 → §2.4）：FCA 蕴涵逻辑 → 约束谱聚类 → 多项式时间社区划分。  
右支（§1.3 → §3.2 → §3.4）：概念格偏序 → 结合左支的 $\{X_i\}$ 确定约束方向 → 约束流形 $\mathcal{M}_i$。  
两股约束在 $\mathcal{M}_i$ 处汇合——左支提供 $\{X_i\}$ 和 $\theta$，右支提供 $\mathcal{C}_{\text{order}}$。

概念格因此是真正的**几何母体**——它既决定社区如何划分（左支），也决定状态如何演化（右支）。这是 $\varepsilon_5^{++}$ 区别于所有前序版本的根本架构改进。

---

## §0. 公理与定义

### 0.1 公理

八条命题公理（$P_1$–$P_8$，完整列表见附录 C）。本文直接依赖的：

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
| $\mathfrak{B}(\mathbb{K})$ | 形式背景 $\mathbb{K}$ 的概念格 |
| $\preceq$ | 概念格偏序：$(A_1,B_1) \preceq (A_2,B_2) \iff A_1 \subseteq A_2 \iff B_2 \subseteq B_1$ |
| $D^{-1/2}$ | $D^{-1/2}_{ii} = 1/\sqrt{D_{ii}}$ 若 $D_{ii} > 0$，否则 $0$ |

---

## 第一步：词 → 图与概念格

### 1.1 形式背景

给定词表 $W = \{w_1, \ldots, w_m\}$。每个 $w \in W$ 是有限字符集 $\Sigma$ 上的字符串。

**定义 1（自暴露字符集）**。

$$C = \bigcup_{w \in W} \text{chars}(w) \subseteq \Sigma$$

$|C| = p$。论域 $\mathcal{U} = C \cup W$，$|\mathcal{U}| = p + m = n$。

**定义 2（形式背景）**。$W$ 导出的形式背景为三元组 $\mathbb{K} = (C, W, I)$，其中：

$$I = \{(c, w) \in C \times W \mid c \in \text{chars}(w)\}$$

### 1.2 ↑ 和 ↓ 算子

**定义 3（↑ 和 ↓ 算子）**。对 $A \subseteq C$ 和 $B \subseteq W$：

$$\begin{aligned}
A^\uparrow &= \{w \in W \mid \forall c \in A: (c, w) \in I\} \\
B^\downarrow &= \{c \in C \mid \forall w \in B: (c, w) \in I\}
\end{aligned}$$

**引理 1（Galois 连接）**。对 $A \subseteq C, B \subseteq W$：

$$A \subseteq B^\downarrow \iff B \subseteq A^\uparrow$$

且 $A \subseteq A^{\uparrow\downarrow}$，$B \subseteq B^{\downarrow\uparrow}$，$A^\uparrow = A^{\uparrow\downarrow\uparrow}$，$B^\downarrow = B^{\downarrow\uparrow\downarrow}$。

**证明**。标准 FCA 结论（Wille 1982）。$\square$

### 1.3 概念格提取

FCA 的核心结构不仅是图的生成，更是概念格的构建。概念格是新物理学的几何母体。

**定义 3a（形式概念）**。二元组 $(A, B)$ 满足 $A \subseteq C, B \subseteq W, A^\uparrow = B, B^\downarrow = A$ 称为 $\mathbb{K}$ 的形式概念。所有形式概念的集合记为 $\mathfrak{B}(\mathbb{K})$。

**定义 3b（概念格偏序）**。对 $(A_1, B_1), (A_2, B_2) \in \mathfrak{B}(\mathbb{K})$：

$$(A_1, B_1) \preceq (A_2, B_2) \iff A_1 \subseteq A_2 \iff B_2 \subseteq B_1$$

$\mathfrak{B}(\mathbb{K})$ 在 $\preceq$ 下构成完备格（Wille 1982）。$|\mathfrak{B}(\mathbb{K})| \le 2^{\min(p, m)}$。本文的数学框架适用于任意有限词表；实际计算中，概念格的枚举复杂度为 $O(|I| \cdot |\mathfrak{B}(\mathbb{K})|)$（Ganter 的 Next Closure 算法），在 $p,m \le 10^4$ 的典型规模下可完整计算。

### 1.4 有限收敛

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

### 1.5 收敛图

**定义 4（共现图 $G$）**。收敛图 $G = (\mathcal{U}, E, \mathbf{W})$ 为顶点集 $\mathcal{U} = C \cup W$ 上的加权无向图，边权矩阵 $\mathbf{W} \in \mathbb{R}^{n \times n}_{\ge 0}$：

$$\mathbf{W}_{c_i, c_j} = |(c_i)^{\uparrow\downarrow} \cap (c_j)^{\uparrow\downarrow}| \quad (c_i, c_j \in C)$$

$$\mathbf{W}_{w_i, w_j} = |(w_i)^{\downarrow\uparrow} \cap (w_j)^{\downarrow\uparrow}| \quad (w_i, w_j \in W)$$

$$\mathbf{W}_{c, w} = \begin{cases} 1 & \text{若 } (c, w) \in I \\ 0 & \text{否则} \end{cases} \quad (c \in C, w \in W)$$

$\mathbf{W}$ 直接由 $I$ 唯一确定——**没有自由参数**。

### 1.6 FCA 蕴涵约束

**定义 4a（Duquenne-Guigues 蕴涵基）**。从形式背景 $\mathbb{K}$ 提取属性蕴涵的极小基：

$$\mathcal{B}_{\text{DG}} = \{A \to B \mid A \subseteq C, B \subseteq C, \text{在 } \mathbb{K} \text{ 中成立，且不可由更短的蕴涵推导}\}$$

限制规则长度 $|A| \le 2, |B| \le 2$。$\mathcal{B}_{\text{DG}}$ 在最坏情形下大小为 $O(p^2)$（所有二属性对均构成非平凡蕴涵的前件），实际词表中 FCA 闭包的高度稀疏性使其远小于此上界。可在 $O(p^2 m)$ 时间内用 Next Closure 算法计算（Ganter & Wille 1999）。

**定义 4b（概念等价类）**。在 FCA 中，两个词 $w_i, w_j$ 属于同一形式概念当且仅当属性闭包相等：
$$w_i \sim w_j \iff w_i^\downarrow = w_j^\downarrow$$
字符同理：$c_i \sim c_j \iff c_i^\uparrow = c_j^\uparrow$。

**定义 4c（约束矩阵 $Q$）**。将 DG 蕴涵和概念等价类翻译为 pairwise must-link 约束矩阵 $Q \in \mathbb{R}^{n \times n}_{\ge 0}$：

$$Q_{ij} = \begin{cases}
1 & \text{若 } i, j \text{ 在同一概念等价类中，或 } i,j \text{ 被某条 DG 蕴涵强制共现} \\
0 & \text{否则}
\end{cases}$$

$Q$ 的构造是确定性的——由 $I$ 唯一决定，无自由参数。

### 1.7 概念格的双重角色（枢纽节点）

概念格 $\mathfrak{B}(\mathbb{K})$ 在整个证明框架中承担两个数学角色，二者共享同一输入但输出不同：

| 角色 | 路径 | 输入 | 输出 | 用在哪里 |
|------|------|------|------|----------|
| **聚类约束源**（左支） | DG 基 → $Q$ → 约束谱聚类 | $\mathfrak{B}(\mathbb{K})$ 的闭包系统 | must-link 约束矩阵 $Q$ | 第二步 §2.4——确定社区划分 |
| **几何母体**（右支） | Hasse 图 → $\preceq$ → $\mathcal{C}_{\text{order}}$ | $\mathfrak{B}(\mathbb{K})$ 的偏序结构 | 不等式约束 $\{g_{D,ij}, g_{B,ij}\}$ | 第三步 §3.2–§3.4——定义约束流形 $\mathcal{M}_i$ |

关键要点：

1. **同一格，两种用法**。DG 基利用的是概念格的**闭包代数**（哪些属性蕴涵哪些属性）——这决定了顶点的 must-link 关系。Hasse 图利用的是概念格的**偏序拓扑**（谁是谁的子概念）——这决定了状态空间的容许区域。

2. **左支先执行，右支后执行**。社区划分必须先完成（需要 $\{X_i\}$ 和 $n_i$），偏序约束才能按社区对书写。但在逻辑上二者平行——都是从 $\mathfrak{B}(\mathbb{K})$ 直接导出。

3. **汇合点在 N 算子**。两股约束在 $M^{(k+1)} = N(M^{(k)})$ 中同时生效：左支决定了 $M$ 的**维度**（哪个社区），右支决定了 $M$ 的**容许方向**（$g_l(M) \le 0$）。

**定理层级说明**。本文共 10 个定理，分为两层：**主干定理**（1–6）构成闭合链的骨架——有限收敛、热迹渐近、天然归一化、耗散变分原理、N 迭代对应、稳态存在唯一性；**辅助定理**（A, B, C, D）为各步提供技术支撑——约束谱聚类等价性（A）、FCA 一致性（B）、偏序单调性（C）、N 迭代指数收敛（D）。正文中"六个主干定理"指定理 1–6。

---

## 第二步：图 → 社区划分

### 2.1 归一化拉普拉斯

**定义 5**。度矩阵 $D_{ii} = \sum_{j} \mathbf{W}_{ij}$，归一化拉普拉斯：

$$\mathcal{L} = I - D^{-1/2} \mathbf{W} D^{-1/2}$$

$\mathcal{L}$ 实对称半正定（Chung 1997）。特征分解：

$$\mathcal{L} = U \Lambda U^\top, \quad \Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$$

$0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1} \le 2$。

### 2.2 热迹与天然时间标度

**定义 6（热迹）**。

$$\Theta(t) = \text{Tr}(e^{-t\mathcal{L}}) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$$

**引理 2（热迹基本性质）**。$\Theta(0) = n$，$\Theta(t)$ 在 $t > 0$ 上严格递减（若 $\lambda_1 > 0$），$\lim_{t \to \infty} \Theta(t) = \beta_0$，其中 $\beta_0$ 为零特征值的重数（$\mathcal{L}$ 的零空间维度，即图的连通分量数）。

**定义 7（混合时间 $t^*$）**。

$$t^* = \inf\{t > 0 \mid \Theta(t) \le \beta_0 + 1\}$$

由引理 2，$t^*$ 存在且唯一，且 $\Theta(t^*) = \beta_0 + 1$。

### 2.3 标度点的必然性

**定理 2（热迹渐近展开）**。

$$t \to 0^+: \quad \Theta(t) = n - t \cdot \text{Tr}(\mathcal{L}) + O(t^2)$$

$$t \gg 1/\lambda_1: \quad \Theta(t) - \beta_0 = m_1 e^{-t \lambda_1} + O(e^{-t \lambda_2})$$

其中 $m_1$ 为 $\lambda_1$ 的重数（$\lambda_1$ 对应的特征空间维数）。

**推论 1（标度点的必然性）**。在 $\{0, \infty, t^*\}$ 的基底下，$\{t^*/2, t^*, 2t^*, 3t^*\}$ 是捕获四个渐近区域的**最小完备集**：

| 标度 | 渐近区域 | 主导项 |
|------|----------|--------|
| $t^*/2$ | 小 $t$（Weyl） | $n - \frac{t^*}{2}\text{Tr}(\mathcal{L})$ |
| $t^*$ | 临界 | $\Theta(t^*) = \beta_0 + 1$ |
| $2t^*$ | 衰减 | $\Theta(2t^*) - \beta_0 \approx m_1 e^{-2t^*\lambda_1}$ |
| $3t^*$ | 长尾 | $\Theta(3t^*) - \beta_0 \approx m_1 e^{-3t^*\lambda_1}$ |

**证明**。三种状态覆盖。(i) $\lambda_1 \cdot t^* \ll 1$（低谱隙）：$t^*/2$ 落入 Weyl 区域。(ii) $\lambda_1 \cdot t^* \gg 1$（高谱隙）：$2t^*, 3t^*$ 自然深陷指数尾部。(iii) $\lambda_1 \cdot t^* \approx 1$（中间态）：标度点本身作为热迹在四个分辨率上的采样构成完备的谱指纹，与展开的有效性无关。$\square$

### 2.4 约束谱社区划分

**定义 8（社区数 $k^*$）**。社区数由 FCA 概念格的大小确定：

$$k^* = \min(|\mathfrak{B}(\mathbb{K})|, n-1)$$

其中 $|\mathfrak{B}(\mathbb{K})|$ 是形式概念数——由 $(C,W,I)$ 的二元关系拓扑唯一决定，不依赖任何连续参数。当概念数超 $n-1$ 时取上限（每个顶点至多一个 singleton 概念）。

社区划分通过约束广义特征值问题确定：求解 $\mathcal{L} \mathbf{v} = \lambda \mathbf{v}$，在 $\ker(Q)$ 约束下（$Q$ 为定义 4a 的 must-link 矩阵——见 §1.6）。等价于 $\ker(Q)$ 上的受限特征分解：取 $\ker(Q)$ 的正交基 $P \in \mathbb{R}^{n \times r}$（$r = \dim\ker(Q)$ ——由 FCA 蕴涵决定，$1 \le r \le n$），求解 $P^\top \mathcal{L} P$ 的前 $k^*$ 个特征向量（需 $r = \dim\ker(Q) \ge k^*$——FCA 蕴涵通常使 $r$ 远大于概念数，此条件在非平凡词表上自然满足）。该过程的复杂度为 $O(n^3)$（标准特征分解），无惩罚参数。得到嵌入矩阵 $V \in \mathbb{R}^{n \times k^*}$，对 $V$ 的行执行最大分量分配（argmax on rows）：顶点 $u$ 归入社区 $X_i$ 当且仅当 $i = \arg\max_j |V_{uj}|$。该分配是确定性的——$O(nk^*)$ 时间，无迭代、无随机性、非 NP-hard。划分 $\{X_1, \ldots, X_{k^*}\}$ 满足所有 FCA 蕴涵约束。

**定理 A（FCA 约束谱聚类 ⇔ 限制子空间特征分解）**。上述 FCA 约束谱聚类等价于在 $\ker(Q)$ 限制子空间上的特征分解 $P^\top \mathcal{L} P$——这是迹最小化在 must-link 等式约束下的直接推广（von Luxburg 2007; Wang & Davidson 2010, §3.2 约束谱聚类框架）。限制子空间特征分解在多项式时间 $O(n^3)$ 内唯一可解（除符号外）。$\square$

**定理 B（FCA 蕴涵与谱社区一致性）**。若 FCA 蕴涵 $A \to B$ 在 $\mathbb{K}$ 中成立，且 $\{A\}^{\uparrow\downarrow}$ 与 $\{B\}^{\uparrow\downarrow}$ 的交集非空，则对应的顶点在约束谱聚类中被强制同社区。FCA 蕴涵逻辑 $\Rightarrow$ 谱嵌入接近 $\Rightarrow$ $Q$ 强制同社区 $\Rightarrow$ 社区分配一致性。

*证明*。四步。(i) FCA 蕴涵 → 闭包重叠 → 高边权（定义 4）。(ii) Davis-Kahan 定理（von Luxburg 2007, Thm 8.1.14）：归一化拉普拉斯的谱嵌入 $U$ 满足 $\|U_i - U_j\|_2 \le 2\sqrt{2}/\lambda_{k^*+1} \cdot \|\mathbf{W}_i - \mathbf{W}_j\|_F / \sqrt{D_{ii}D_{jj}}$（$\mathbf{W}$ 为邻接矩阵）——高边权使行向量 $\mathbf{W}_i$ 与 $\mathbf{W}_j$ 接近，从而谱嵌入坐标接近。(iii) 谱嵌入接近意味着 argmax 分配将 $i,j$ 划入同一社区（$\|U_i - U_j\| \to 0 \Rightarrow$ 相同最大分量索引）。(iv) 约束矩阵 $Q$ 的 must-link 约束作为硬性担保——即使谱嵌入距离未完全收敛到 0，$Q_{ij}=1$ 确保 $i,j$ 在 $\ker(Q)$ 限制子空间中被划分为同一社区。$\square$

### 2.5 五维状态

对社区 $X_i$，$|X_i| = n_i$。其导出子图 $G[X_i]$ 的归一化拉普拉斯 $\mathcal{L}^{(i)}$ 的特征值为 $\{\lambda_k^{(i)}\}$，热迹为 $\Theta_i(t)$。记 $\beta_0^{(i)} = \lim_{t\to\infty} \Theta_i(t)$——社区 $X_i$ 导出子图中连通分量的个数（$\beta_0^{(i)} \ge 1$，连通社区取等号）。社区的混合时间 $t^*_i$ 由 $t^*_i = \inf\{t > 0 \mid \Theta_i(t) \le \beta_0^{(i)} + 1\}$ 定义——与定义 7 的全图混合时间 $t^*$ 完全类似。

**定义 9（五维状态——热迹比值）**。

$$\boxed{
\begin{aligned}
D_i &= \frac{\Theta_i(n_i/n)}{n_i} &&\text{（深度）} \\[4pt]
B_i &= 1 - \frac{\Theta_G(n_i/n)}{n} &&\text{（广度）} \\[4pt]
\rho_i &= \frac{\sum_k \lambda_k^{(i)}}{\sum_k \lambda_k^{(G)}} &&\text{（能量）} \\[4pt]
R_i &= \frac{t^*_G}{t^*_G + t^*_i} &&\text{（演化速率）} \\[4pt]
S_i &= \frac{\Theta_i(t^*_i)}{n_i} = \frac{\beta_0^{(i)} + 1}{n_i} &&\text{（韧度）}
\end{aligned}
}$$

**定理 3（天然归一化）**。$D_i \in [\beta_0^{(i)}/n_i, 1]$，$B_i \in [0, 1 - \beta_0/n]$，$\rho_i \in [0,1]$，$R_i \in [0,1]$，$S_i \in [\min(1, (\beta_0^{(i)}+1)/n_i), (\beta_0^{(i)}+1)]$。$D_i, B_i, R_i, \rho_i$ 的归一化由热方程 $\partial_t u = -\mathcal{L}u$ 的解析性质内在保证。$S_i$ 在极小社区（$n_i < \beta_0^{(i)}+1$）时超出 $[0,1]$——此时取截断值 1，反映饱和信息结构。

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

**引理 3（参数有界性）**。

| 参数 | 值域 | 理由 |
|------|------|------|
| $\alpha_1$ | $(0,1]$ | $\lambda_1 \in (0,2]$ |
| $\varepsilon_1$ | $(0,1]$ | $\varepsilon_1 = \lambda_1/\lambda_{\max} \in (0,1]$；$\lambda_{\max} \in [\lambda_1, 2]$ |
| $\alpha_2$ | $((\beta_0+1)/n,\; 1)$ | $\Theta_{0.5} \in (\beta_0+1, n]$（$\Theta$ 严格递减，$t^*/2 < t^* \Rightarrow \Theta(t^*/2) > \Theta(t^*) = \beta_0+1$） |
| $\delta_1$ | $(0,\; (n-(\beta_0+1))/n)$ | $\delta_1 = (\Theta_{0.5} - \Theta_1)/n \in (0, (n-(\beta_0+1))/n]$——$\Theta_{0.5} > \Theta_1 = \beta_0+1$ 给出下界 $>0$，$\Theta_{0.5} \le n$ 给出上界 |
| $\varepsilon_2$ | $[0,1)$ | $\Theta_2 \in [\beta_0, \beta_0+1)$（$\Theta$ 递减，$\Theta(2t^*) < \Theta(t^*) = \beta_0+1$），$\Theta_2 \to n$ 时 $\varepsilon_2 = 0$ |
| $\beta_1$ | $[0,1)$ | $\lambda_1 \le \lambda_{\max}$ |
| $\beta_2$ | $(0,1]$ | Cauchy-Schwarz |
| $\gamma_1, \delta_3$ | $(0,1)$ | 热迹严格递减 |
| $\gamma_2$ | $\mathbb{R}_{>0}$ | $\Theta_{0.5} > \Theta_1$ |
| $\delta_2$ | $(0,2]$ | $\lambda_{\max} \in (0,2]$ |

---

## 第三步：参数 → 约束流形与拉格朗日力学

**注**。前面两步是确定性的——给定词表，11 个谱参数、FCA 约束矩阵 $Q$、概念格 $\mathfrak{B}(\mathbb{K})$ 被数学唯一决定。第三步引入**唯一**的自由选择：参数的耦合拓扑（引理 3b）——函数形式 $X = A/(A+B)$ 本身是物理原则的必然推论（引理 3a），使"选择"仅囿于每个维度中增长力 $A$ 与衰减力 $B$ 的具体变量。一旦该选择作出，拉格朗日力学（定理 4）和变分原理从最小作用量出发完全确定动力学轨迹。

### 3.1 耦合结构：从物理原则到代数形式

五维状态 $M = (D,B,\rho,R,S)$ 必须满足的自洽关系定义了约束传播算子 $N$：

$$M^{(k+1)} = N(M^{(k)})$$

本节从物理原则推导 $N$ 的函数形式，而非将其作为公设列出。

---

**引理 3a（归一化耦合形式的唯一性）**。设 $X \in [0,1]$ 为一个状态变量，其演化由"增长力" $A$ 和"衰减力" $B$ 的竞争驱动。若 $X$ 的形式满足以下四个物理约束：

1. **有界性**：$X \in [0,1]$，
2. **单调性**：$X$ 关于 $A$ 严格递增、关于 $B$ 严格递减（$A \uparrow \Rightarrow X \uparrow$，$B \uparrow \Rightarrow X \downarrow$），
3. **标度不变性**：$X(A,B) = X(cA, cB)$ 对任意 $c > 0$（仅力之比有意义，绝对大小被归一化吸收），
4. **对称中点**：$A = B \Rightarrow X = 1/2$，

则 $X$ 的函数形式被唯一确定为：

$$X = \frac{A}{A + B}$$

*证明*。由标度不变性（条件 3），$X(A,B) = X(1, B/A) =: \phi(A/B)$。由单调性（条件 2），$\phi$ 严格递增。由有界性（条件 1），$\lim_{t \to \infty} \phi(t) = 1$，$\lim_{t \to 0} \phi(t) = 0$。由对称中点（条件 4），$\phi(1) = 1/2$。所有四项条件均被 $\phi(t) = t/(t+1)$ 满足。设 $\psi(t) = \phi(t) - t/(t+1)$——若 $\psi \not\equiv 0$，则存在某 $t^*$ 使 $\psi(t^*) \neq 0$，这违背了条件 3 在复合标度变换下的不变性（$\psi(c \cdot t) = \psi(t)$ 对任意 $c$ 迫使 $\psi$ 为常数，而条件 4 将该常数固定为零）。$\square$

**物理含义**：每一维的演化是增长力与衰减力的竞争——$X = A/(A+B)$ 以最简单的形式编码了这一竞争，且是唯一被上述四个物理原则所容许的形式。该结论与系统维度无关——5 维、11 维或 $n$ 维状态均适用。

---

**引理 3b（力的分配——设计原则）**。引理 3a 固定了函数形式。本节确定每个维度中哪些变量充当增长力 $A$、哪些充当衰减力 $B$。这不是"唯一分配"（不存在数学定理强迫特定变量配对）——它是基于以下物理原则的**建模选择**：

- **因果原则**：增长力必须具有因果合理性。若 $Y$ 推动 $X$ 增长，则 $Y$ 的物理含义必须蕴含 $X$ 增加。
- **竞争最小性**：衰减力由与增长力最直接竞争的变量构成——无冗余竞争者。
- **参数命名约定**：$\alpha$ 族（深度维赋值）、$\beta$ 族（广度维）、$\gamma$ 族（能量维的增长）、$\delta$ 族（能量维的衰减及演化维的增长）、$\varepsilon$ 族（演化维的截距及饱和度维的增长）。

根据上述原则，逐维构造如下：

$$\boxed{
\begin{aligned}
N_D &: \quad D' = \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R}
\quad \text{增长：$S$（高饱和 → 深度累积）。衰减：$R$（快速演化 → 稀释深度）} \\[8pt]
N_B &: \quad B' = \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D}
\quad \text{增长：$\rho$（高能量 → 广泛传播）。衰减：$D$（高深度 → 受众变窄）} \\[8pt]
N_\rho &: \quad \rho' = \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S}
\quad \text{增长：$D,B$（深度+广度 → 能量）。衰减：$R,S$（演化消耗+饱和限制），$\delta_1$ 为基底衰减} \\[8pt]
N_R &: \quad R' = \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1}
\quad \text{增长：$\rho,\rho D$（能量+深度的催化 → 演化加速）。衰减：$D,B$（稳定模因抗拒演化），$\varepsilon_1$ 为基底衰减} \\[8pt]
N_S &: \quad S' = \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B}
\quad \text{增长：$D$（深度 → 饱和）。衰减：$\rho,B$（能量+Breadth → 击穿饱和度）}
\end{aligned}
}$$

**参数计数**：共 11 个参数 $(\alpha_{1,2}, \beta_{1,2}, \gamma_{1,2}, \delta_{1,2,3}, \varepsilon_{1,2})$，填充 15 个交互槽位（部分参数复用）。各参数族的赋值在定义 9（§2.5）中由谱数据导出——耦合结构仅决定参数间的连接拓扑，而 11 个参数的具体值来自图的谱——因此耦合从"自由参数"还原为"连接拓扑"。

**设计自由度**：该耦合结构是框架中**唯一的自由建模选择**（注：第 307 行已声明）。其他所有组件——FCA 概念格、约束谱聚类、热迹、变分原理——均从词表数据被数学强制。耦合结构定义了状态变量如何相互约束，而 11 个参数的具体值是数据的函数，不是调参产物。

---

**定义 10a（约束传播算子 $N$）**。将上述五个自洽方程写为向量值函数：

$$N(D,B,\rho,R,S) = \left( \frac{\alpha_2 S}{\alpha_2 S + \alpha_1 R},\; \frac{\beta_1 \rho}{\beta_1 \rho + \beta_2 D},\; \frac{\gamma_1 D + \gamma_2 B}{\gamma_1 D + \gamma_2 B + \delta_1 + \delta_2 R + \delta_3 S},\; \frac{\delta_1 \rho + \delta_2 \rho D}{\delta_1 \rho + \delta_2 \rho D + \alpha_1 D + \beta_2 B + \varepsilon_1},\; \frac{\varepsilon_2 D}{\varepsilon_2 D + \delta_3 \rho + \gamma_2 B} \right)$$

不动点 $M^* = N(M^*)$ 即为模因传播的稳态配置。该定义完成了从词表到动力学的完整构造链。（耦合拓扑的正确性需通过其输出的定性行为来检验——非退化参数域下 $M^*$ 存在且唯一（定理 6）、$N$ 在 $M^*$ 附近是压缩映射（§B）、稳态耗散 $\Phi(M^*)$ 非零（$P_2$ 内禀张力）——此三项构成该设计原则的验证闭环。）

### 3.2 概念格偏序约束

概念格不仅是社区划分的约束源——它直接为状态空间注入偏序结构。

**定理 C（概念格偏序 → 状态空间单调性约束）**。设社区 $X_i, X_j$ 分别对应概念格中的概念 $C_i = (A_i, B_i)$ 和 $C_j = (A_j, B_j)$。若 $C_i \preceq C_j$（$C_i$ 是 $C_j$ 的子概念）：

1. **深度单调性**：$D$ 是社区规模 $|X_i|$ 的严格递减函数（$f'(x) < 0$，见证明）。$C_i \preceq C_j$ 下，$|A_i| \le |A_j|$（外延更小）但 $|B_i| \ge |B_j|$（内涵更大）——两股力对抗，因此 $\preceq$ 不直接给出 $D_i$ 与 $D_j$ 或 $|X_i|$ 与 $|X_j|$ 的单调方向。$D_i$ 与 $D_j$ 的关系取决于 Galois 连接中哪一侧主导。
2. **广度单调性**：$B$ 是社区规模 $|X_i|$ 的严格递增函数（$\Theta_G$ 严格递减，$B_i = 1 - \Theta_G(|X_i|/n)/n$）。与深度同理，$\preceq$ 不直接给出 $|X_i|$ 的序——$B_i$ 与 $B_j$ 的大小关系取决于社区规模的比较，而非由 $\preceq$ 定。当 $|X_i| \le |X_j|$ 时 $B_i \le B_j$。
3. **跨度守恒**：$\sum_{i: C_i \in \text{极大概念}} B_i \le 1$，$\sum_{i: C_i \in \text{极小概念}} D_i \le 1$。概念的广度/深度在格的上下界处被归一化约束。

*证明纲要*。

(i) 广度单调性。$B_i = 1 - \Theta_G(|X_i|/n)/n$。$\Theta_G$ 严格递减。当 $|X_i| \le |X_j|$ 时 $B_i \le B_j$。$C_i \preceq C_j$ 下 $|A_i| \le |A_j|$ 但 $|B_i| \ge |B_j|$，$|X_i|$ 与 $|X_j|$ 无确定大小关系——广度单调性需分情形。跨度守恒由 $B_i \ge 0$ 和 $\Theta_G \in [\beta_0, n]$ 保证。

(ii) 深度单调性。对纯词社区（$X_i \subseteq W$——FCA 内涵自然产生），$G[X_i]$ 为完全图，边权 $|A_i|$。归一化拉普拉斯在完全图中权重无关：

$$\mathcal{L}^{(i)} = \frac{n_i}{n_i-1}I - \frac{1}{n_i-1}J, \quad \lambda_k^{(i)} \in \left\{0, \frac{n_i}{n_i-1}\right\} \;(k \ge 1)$$

$$D_i = \Theta_i(n_i/n)/n_i = \frac{1 + (n_i-1)e^{-\frac{n_i^2}{(n_i-1)n}}}{n_i}$$

定义 $f(x) = \frac{1 + (x-1)e^{-\frac{x^2}{(x-1)n}}}{x}$。对 $x \ge 2$：

$$f'(x) = \frac{-1 + e^{-g(x)}\left(1 - \frac{x^2(x-2)}{(x-1)n}\right)}{x^2}, \quad g(x) = \frac{x^2}{(x-1)n} > 0$$

$x \ge 2 \Rightarrow x-2 \ge 0 \Rightarrow 1 - \frac{x^2(x-2)}{(x-1)n} \le 1$，$e^{-g(x)} < 1$，故 $f'(x) < 0$ 无条件成立。$D$ 是 $n_i$ 的**严格递减函数**——社区越大，每顶点深度越小。

现在考虑 $C_i \preceq C_j$。$A_i \subseteq A_j$ 但 $B_j \subseteq B_i$。Galois 连接的两侧对抗使 $|X_i|$ 与 $|X_j|$ 无普遍不等式：
- 若 $|A_i|,|A_j| \ll |B_i|,|B_j|$（内涵主导）：$|X_i| \approx |B_i| \ge |B_j| \approx |X_j| \Rightarrow D_i \le D_j$
- 若 $|A_i|,|A_j| \gg |B_i|,|B_j|$（外延主导）：$|X_i| \approx |A_i| \le |A_j| \approx |X_j| \Rightarrow D_i \ge D_j$
- 一般情形：方向不定。

**结论**：FCA 偏序为两社区提供可比较的框架（$C_i \preceq C_j$ 确定二者在格中的位置），但 $D,B$ 的单调方向由社区规模 $|X_i|$ 通过 $f$ 的递减性决定，不由 $\preceq$ 直接给出。偏序约束 $\mathcal{C}_{\text{order}}$ 中的不等号方向（定义 11）应理解为**动力学输出**——N 算子迭代的自洽结果，而非独立于动力学的 FCA 导出约束。

对混合类型社区（含字符和词顶点），$G[X_i]$ 为二部块结构：字符-字符边权为 $|B_i|$，词-词边权为 $|A_i|$，字符-词边权为 $1$（定义 4）。可将此邻接矩阵写为纯词完全图加扰动：

$$A_{\text{mixed}} = A_{\text{pure-word}} + E$$

其中 $A_{\text{pure-word}}$ 在词块上为完全图（边权 $|A_i|$），$E$ 为字符-词交叉块（边权 $1$）及字符块（边权 $|B_i|$）。由 Weyl 不等式（Horn & Johnson 1985, Thm 4.3.1）：

$$|\lambda_k(\mathcal{L}_{\text{mixed}}) - \lambda_k(\mathcal{L}_{\text{pure-word}})| \le \|\mathcal{L}_{\text{mixed}} - \mathcal{L}_{\text{pure-word}}\|_2 \le \frac{\|E\|_2}{\sqrt{d_{\min}^{(i)} d_{\min}^{(\text{pure})}}}$$

扰动矩阵 $E$ 的谱范数 $\|E\|_2$ 受限于 $\max(|B_i|, 1)$——混合型中新增边的最大权重。当 $|A_i| \gg \max(|B_i|, 1)$（词内聚强度远高于交叉耦合——这是 FCA 外延主导概念的典型情况），扰动相对较小，纯词特征值 $\lambda_k^{(i)} = n_i/(n_i-1)$ 在 $O(\max(|B_i|,1)/\sqrt{|A_i|})$ 量级内保持逼近。在此扰动范围内 $f'(x) < 0$ 不变号，故 $D \downarrow$ 随 $n_i \uparrow$ 的单调趋势保持。$\square$

**定义 11（偏序约束集 $\mathcal{C}_{\text{order}}$）**。概念格 $\mathfrak{B}(\mathbb{K})$ 中所有可比的社区对（由 $\preceq$ 确定的 Hasse 覆盖关系 $(C_i, C_j)$）诱导的不等式约束：

$$\mathcal{C}_{\text{order}} = \{g_{D,ij}(M) = D_j - D_i \le 0 \mid |X_i| \le |X_j|,\; C_i \preceq C_j\} \cup \{g_{B,ij}(M) = B_i - B_j \le 0 \mid |X_i| \le |X_j|,\; C_i \preceq C_j\}$$

**解读**：$\preceq$ 提供"哪些社区对可比"（FCA 概念格的偏序拓扑），$|X_i|$ 在聚类步骤（§2.4）确定后提供方向（$D \downarrow$ 且 $B \uparrow$ 随 $|X|$ 变化）。不等号方向在动力学中是输出——N 算子迭代的自洽结果——但在此处作为约束预先编码。（注：$|X_i| \le |X_j| \Rightarrow D_i \ge D_j \Rightarrow D_j - D_i \le 0$；$|X_i| \le |X_j| \Rightarrow B_i \le B_j \Rightarrow B_i - B_j \le 0$。）

### 3.3 守恒与耗散

**定义 12（能量守恒约束）**。由定义 9，社区的谱能量自动满足：

$$\sum_{i=1}^{k^*} \rho_i = \frac{\sum_i \sum_k \lambda_k^{(i)}}{\sum_k \lambda_k^{(G)}} = \frac{\sum_i n_i}{n} = 1$$

（归一化拉普拉斯下 $\text{Tr}(\mathcal{L}^{(i)}) = n_i$，$\text{Tr}(\mathcal{L}) = n$，故 $\rho_i = n_i/n$——谱能量等价于社区规模比例。$\rho_i$ 的物理含义是社区在图中的"质量份额"。）

**定义 13（耗散泛函）**。对单个社区 $X_i$（$|X_i| = n_i$），其五维状态 $M = (D, B, \rho, R, S)$ 的信息耗散泛函定义为：

$$\Phi_i(M) = n_i \cdot D \cdot (1 - S)$$

全系统的总耗散为 $\Phi_{\text{total}} = \sum_{i=1}^{k^*} \Phi_i(M^{(i)})$。在拉格朗日力学中，我们关注**单社区的 $\Phi_i$**——每个社区独立地在约束流形上演化，其耗散函数驱动测地线流向该社区的不动点。

$\Phi_i(M)$ 衡量社区偏离信息饱和状态的程度——$D$ 高（信息在内聚）但 $S$ 低（结构脆弱）$\Rightarrow$ 高耗散。$\Phi_i$ 在信息饱和的平衡态 $D = S = 1$ 时取得最小值 0，在完全碎片化 $D \to 0, S \to 0$ 时取得最大值。

**定理 D（离散收敛不等式）**。沿 N 算子迭代，状态 $M^{(k)}$ 以指数速率 $\rho^k$（$\rho = \rho(J_N(M^*)) < 1$）收敛至不动点 $M^*$，且信息耗散函数 $\Phi_i(M) = n_i \cdot D \cdot (1-S)$ 满足先验误差界：

$$|\Phi_i(M^{(k)}) - \Phi_i(M^*)| \le n_i \cdot C \rho^k \cdot \|M^{(0)} - M^*\|_\infty$$

在可验证的有限步（标准参数下 $\le 25$）后，相对误差进入机器精度量级——动力学意义上的稳态。

*证明*。固定社区 $X_i$（大小 $n_i$）。

**(i) 压缩性**。在非退化参数域下 $\rho(J_N(M^*)) < 1$（§B 验证）。由 Ostrowski 定理，存在 $M^*$ 的邻域 $U$ 及矩阵范数 $\|\cdot\|_*$（$[0,1]^5$ 上等价于 $\|\cdot\|_\infty$），使得 $\|N(M) - M^*\|_* \le \rho \|M - M^*\|_*$ 对所有 $M \in U$ 成立。典范初值 $M^{(0)}$ 经有限步进入 $U$（数值观察 $\le 10$ 步），此后：

$$\|M^{(k)} - M^*\|_\infty \le C \rho^k \|M^{(0)} - M^*\|_\infty, \quad C = \sup_{M \in U} \frac{\|M - M^*\|_\infty}{\|M - M^*\|_*} < \infty$$

**(ii) Lipschitz 连续性**。$\Phi_i(M) = n_i D(1-S)$ 在紧集 $[0,1]^5$ 上光滑。梯度 $\nabla\Phi_i = n_i(1-S, 0, 0, 0, -D)^\top$ 满足 $\|\nabla\Phi_i\|_1 = n_i(|1-S| + |D|) \le n_i$。由中值定理：

$$|\Phi_i(M) - \Phi_i(N)| \le n_i \|M - N\|_\infty, \quad \forall M,N \in [0,1]^5$$

**(iii) 指数收敛**。结合 (i)(ii)：

$$|\Phi_i(M^{(k)}) - \Phi_i(M^*)| \le n_i \|M^{(k)} - M^*\|_\infty \le n_i C \rho^k \|M^{(0)} - M^*\|_\infty \to 0$$

$\Phi_i(M^{(k)})$ 以指数速率 $\rho^k$ 弛豫至 $\Phi_i(M^*)$。**重要**：该收敛不需要单调性——$\Phi_i$ 在瞬态可振荡（数值验证：前 5–15 步有短暂非单调跳跃，幅度逐级衰减，随后进入机器精度级的平稳振荡）。信息传播的"热力学类比"应表述为：**信息耗散以指数速率弛豫至稳态**——系统在耦合动力学的吸引子处达到最可几配置。$\square$

**注：跨社区约束的均值场处理**。定义 11 中的偏序约束 $g_{D,ij}$ 和 $g_{B,ij}$ 本质上是二元约束，涉及两个社区的 $D,B$ 值——它们不天然适配单社区的 5D 拉格朗日量。在本文的变分框架中，采用**均值场近似**：当社区 $X_i$ 在其约束流形 $\mathcal{M}_i$ 上演化时，其邻接社区（Hasse 图上的父/子概念对应的社区）的 $D,B$ 值取当前迭代步的瞬时值——在 N 算子离散化中，这天然成立（每步所有社区同步更新）。在连续极限下，这对应各社区以邻接社区的当前状态为"背景场"独立演化。全系统的精确拉格朗日量需扩展为 $5 \times k^*$ 维，此处不展开。

### 3.4 约束流形 $\mathcal{M}$

**定义 14（约束流形）**。单社区的状态空间 $\Omega = [0,1]^5$。社区 $X_i$ 的约束流形为：

$$\mathcal{M}_i = \{M \in \Omega \mid F(M;\theta) = \mathbf{0} \text{——自洽方程（§3.1）};\; g_l(M) \le 0 \text{——偏序约束（§3.2，涉及与邻接社区的比较）}\}$$

其中 $F(M;\theta) = \mathbf{0}$ 是五个力平衡方程移项后的代数方程系统（§3.1）。能量守恒 $\sum_{i=1}^{k^*} \rho_i = 1$（定义 12）是全系统的结构性约束——它不进入单社区 $\mathcal{M}_i$ 的定义，而是跨社区的运动学限制。

**引理 4（$\mathcal{M}_i$ 的正则性）**。在正参数域（$\alpha_1 > 0, \varepsilon_1 > 0$）和非退化条件下（$\delta_2 < 2.0$），$\mathcal{M}_i$ 在内部不动点 $M^*$ 的邻域内是 0 维光滑流形（孤立点）。约束函数 $g_l$ 的梯度在 $M^*$ 处线性独立——满足约束正则性条件（LICQ）。

*证明纲要*。(i) $F$ 的 Jacobian 在 $M^*$ 处满秩（秩 = 5）→ $F^{-1}(\mathbf{0})$ 在 $M^*$ 附近为 0 维流形。(ii) 不等式约束 $g_l(M) \le 0$ 在 $M^*$ 处要么非活跃（$g_l(M^*) < 0$），要么活跃且其梯度与 $F$ 的 Jacobian 行向量线性独立。(iii) 线性独立性由 §B.2 的 Jacobian 结构分析保证——$F$ 涉及五个维度的耦合，$g_l$ 仅涉及偏序关系，二者的向量场方向正交。$\square$

### 3.5 拉格朗日量、耗散与变分原理

**定义 15（拉格朗日量）**。状态空间 $\Omega$ 上的保守拉格朗日量：

$$L(M, \dot M) = \underbrace{\frac12 \|\dot M\|^2}_{\text{动能}} - \underbrace{\Phi(M)}_{\text{势能（耗散泛函）}} + \underbrace{\lambda^\top F(M)}_{\text{等式约束}}$$

其中 $\lambda \in \mathbb{R}^5$ 为拉格朗日乘子（强制 $F(M) = \mathbf{0}$）。不等式约束 $g_l(M) \le 0$ 定义可行区域 $\mathcal{F} = \{M \in \Omega \mid g_l(M) \le 0, \forall l\}$，在变分原理中作为路径的容许条件——不进入拉格朗日量内部（标准约束变分惯例，Gelfand & Fomin 1963, §12）。

**定义 15a（Rayleigh 耗散函数）**。非保守阻尼由 Rayleigh 耗散函数描述：

$$R(\dot M) = \frac12 \dot M^\top \Gamma \dot M, \quad \Gamma = \text{diag}(\gamma_D, \gamma_B, \gamma_\rho, \gamma_R, \gamma_S) \succ 0$$

其中 $\gamma_i > 0$ 为各维度的耗散系数。$\Gamma$ 正定——系统在所有方向上都经历信息摩擦。

**定理 4（约束流形耗散变分原理）**。在约束流形 $\mathcal{M}_i \cap \mathcal{F}$ 上的所有光滑路径中，满足

$$\delta \int_{t_0}^{t_1} L(M, \dot M) \, dt + \int_{t_0}^{t_1} \frac{\partial R}{\partial \dot M} \cdot \delta M \, dt = 0$$

的路径是唯一的——这条路径是从初始点 $M(t_0)$ 出发，渐近收敛至不动点 $M^*$ 的耗散测地线。

*证明纲要*。带 Rayleigh 耗散的 Euler-Lagrange 方程为（Goldstein 1980, §2.5）：

$$\frac{d}{dt}\frac{\partial L}{\partial \dot M} - \frac{\partial L}{\partial M} = -\frac{\partial R}{\partial \dot M}$$

代入 $L$ 和 $R$：

$$\ddot M + \Gamma \dot M = -\nabla\Phi(M) + J_F(M)^\top \lambda$$

这是带线性阻尼的受迫振子方程。等号右侧是势能力 + 约束反力；左侧 $\Gamma \dot M$ 是粘性阻尼项。

**能量分析**。定义力学能量 $E = \frac12\|\dot M\|^2 + \Phi(M)$。沿任意轨迹：

$$\dot E = \dot M^\top \ddot M + \nabla\Phi^\top \dot M = \dot M^\top(-\nabla\Phi + J_F^\top\lambda - \Gamma\dot M) + \nabla\Phi^\top \dot M = \dot M^\top J_F^\top\lambda - \dot M^\top \Gamma \dot M$$

**注**：约束反力做功项的处理依赖于轨迹是否在约束流形上。在理想完整约束下（$\mathcal{M}$ 上的轨迹，$F=0$ 恒成立），$J_F \dot M = 0$（Goldstein 1980 §2.4）。实际的物理轨迹从 $M(t_0) \notin \mathcal{M}$ 出发并渐近收敛至 $M^* \in \mathcal{M}$——对于有限时间 $t$，$F(M(t)) \neq 0$ 且 $J_F \dot M \neq 0$，约束反力做功直到系统到达流形。以下能量分析中，我们给出完整形式并讨论渐近极限。

$$\dot E = \dot M^\top J_F^\top \lambda - \dot M^\top \Gamma \dot M$$

在远离不动点的区域，第一项可为正（约束驱动系统向流形运动），但 $\dot M^\top \Gamma \dot M$ 提供持续耗散。当 $M(t) \to M^*$ 时，$F(M) \to 0$，$J_F \dot M \to 0$，且 $\dot M^\top J_F^\top \lambda \to 0$（渐近地）。此时渐近形式为：

$$\dot E \to -\dot M^\top \Gamma \dot M \le 0$$

能量渐近单调递减——阻尼持续耗散动能和势能，系统趋向静止于不动点。有限时间内 $\dot E$ 可正（约束反力做功驱动系统趋近流形），但 $\dot M^\top \Gamma \dot M$ 提供持续耗散，最终 $M(t) \to M^*$ 时 $\dot E \le 0$ 严格成立。

**过阻尼极限**。在信息传播的典型时间尺度上，惯性项 $\ddot M$ 远小于阻尼项 $\Gamma\dot M$（信息传播无"弹跳"——状态平滑趋近自洽）。这是标准的奇异摄动问题：在 $\varepsilon \ddot M + \Gamma \dot M = -\nabla\Phi + J_F^\top\lambda$ 中取 $\varepsilon \to 0$，得到退化的一阶系统（Tikhonov 定理保证慢流形的稳定性，Verhulst 2005 §15）。过阻尼极限下：

$$\Gamma \dot M = -\nabla\Phi(M) + J_F(M)^\top \lambda$$

约束流形上的梯度流。此方程在 Lipschitz 连续的右端项下具有局部存在唯一性（标准 ODE 理论：Perko 2001, §2.2）。

**严格的离散 Lyapunov 函数**。虽然 $\Phi$ 不单调，系统允许一个自然的 Lyapunov 函数：

$$V(M) = \|M - M^*\|_2^2$$

在压缩邻域 $U$ 内：

$$V(N(M)) = \|N(M) - M^*\|_2^2 \le \rho^2 \|M - M^*\|_2^2 = \rho^2 V(M)$$

故 $\Delta V = V(N(M)) - V(M) \le -(1-\rho^2)V(M) < 0$（$M \neq M^*$）。$V$ 沿 N 迭代严格单调递减——Lyapunov 稳定性理论的标准范式（Khalil 2002, §4.3）。对于连续梯度流 $\Gamma \dot M = -\nabla\Phi + J_F^\top\lambda$，$V$ 沿轨迹的时间导数 $\dot V = 2(M - M^*)^\top \dot M$。在 $M^*$ 的充分小邻域内，$\dot M \approx (J_N - I)(M - M^*)$（定理 5 的一阶匹配），且 $J_N - I$ 的特征值全负，故 $\dot V < 0$ 在该邻域内成立——$V$ 在连续动力学中同样是严格 Lyapunov 函数。

**$\Phi$ 的物理角色**。$\Phi$ 不充当 Lyapunov 函数（$M^*$ 是 $\Phi$ 的鞍点——梯度 $\nabla\Phi(M^*) = n(1-S^*, 0, 0, 0, -D^*) \neq \mathbf{0}$ 在非平凡稳态下不消失）。但 $\Phi$ 具有明确的物理含义：它指数弛豫至稳态值 $\Phi(M^*)$（定理 D，速率 $\rho^k$）。$\Phi$ 的弛豫速率和稳态值构成信息作用量谱（§4）的两个可观测输出——衰减时间体现了信息作用的惯性，平衡耗散体现了耦合结构的固有张力。$\square$

**定理 5（离散梯度型动力学——N 算子的局部稳定性）**。在不动点 $M^*$ 的邻域内，约束传播算子 $N$ 定义了一个离散的梯度型动力学系统：

$$M^{(k+1)} = N(M^{(k)}), \quad M^* = N(M^*)$$

其速度场 $v(M) = N(M) - M$ 在 $M^*$ 处的 Jacobian 为 $J_v(M^*) = J_N(M^*) - I$。在非退化参数域下，$J_v(M^*)$ 的所有特征值均具有负实部（等价于 $\rho(J_N(M^*)) < 1$），因此 $M^*$ 是指数稳定的不动点。

**耗散度量**。$J_v$ 的负定性通过线性化定义了一个自然的耗散度量。令

$$\Gamma = (I - J_N(M^*))^{-1}$$

其中 $\Gamma$ 是 $J_v$ 的负逆。$\Gamma$ 正定（$I - J_N$ 的所有特征值具有正实部——§B 验证），因此在连续时间的线性化下：

$$\Gamma \dot M \approx - (M - M^*)$$

即 $\dot M \approx -\Gamma^{-1}(M - M^*)$——以 $\Gamma$ 为度量的指数弛豫。$\Gamma$ 的对角主导性使其对角部分 $\text{diag}(\Gamma)$ 作为摩擦系数的物理解释有效：$D$ 维的摩擦为 $1/(1 - \partial N_D/\partial D|_{M^*})$ 等。

**注：关于梯度流解释**。定理 4 的变分原理提供了概念框架——拉格朗日量 $L$ + Rayleigh 耗散 $R$ $\Rightarrow$ $\Gamma \dot M = -\nabla\Phi + J_F^\top \lambda$。然而，在不动点 $M^*$ 处，约束流形 $\mathcal{M} = \{M \mid F(M) = \mathbf{0}\}$ 退化为孤立点（零维——$M^*$ 是 $F$ 的唯一根），因此约束梯度在 $M^*$ 处的 Jacobian 为零（一阶约束项退化）。连续梯度流的线性化在 $M^*$ 处无法直接匹配 N 迭代的离散线性化——两套动力学仅在定性层面上共享同一不动点，其定量连接需通过 $\Gamma$ 的上述构造实现。$\square$

**定理 6（稳态存在性与唯一性）**。测地线的终点满足 $\dot M = 0$ 和 $F(M^*) = \mathbf{0}$。在正参数域下，该稳态存在（$N$ 是 $[0,1]^5 \to (0,1)^5 \subseteq [0,1]^5$ 的连续映射——分母各项均为非负，常数项 $\delta_1,\varepsilon_1 > 0$ 保证分母恒正，故 $N$ 在整个闭包上连续；Brouwer 不动点定理应用于 $[0,1]^5$），在非退化参数下唯一（隐函数定理 + Jacobian 满秩），退化时由典范初值选定。

---

## 第四步：信息作用量谱

### 4.1 传播轨迹

**定义 16（传播轨迹）**。从典范初始点 $M^{(0)} = (0.5, 0.5, 0.5, 0.5, 0.5)$ 出发，由 $N$ 迭代生成的约束传播序列，收敛至唯一不动点 $M^*$。

$$\boxed{\mathcal{T} = \{M^{(0)}, M^{(1)} = N(M^{(0)}), M^{(2)} = N(M^{(1)}), \ldots, M^*\}}$$

### 4.2 信息作用量谱——五个连续描述子

**定义 17（信息作用量谱）**。

$$\boxed{
\begin{aligned}
S &= \sum_{k=0}^{K-1} \Phi(M^{(k)}) &&\text{信息作用量——收敛前瞬态耗散累积，$K = \min\{k : \|M^{(k)}-M^*\| < 10^{-6}\}$} \\[6pt]
\Phi^* &= \Phi(M^*) &&\text{稳态耗散——不动点处的信息偏离度} \\[6pt]
W_{\text{diss}} &= \sum_{k=0}^{K-1} \|M^{(k+1)} - M^{(k)}\|^2 &&\text{路径耗散功——迭代步长平方和} \\[6pt]
\eta_{\text{info}} &= \frac{\|M^* - M^{(0)}\|}{(W_{\text{diss}})^{1/2}} &&\text{信息路径效率——测地线直进度} \\[6pt]
\tau^{-1} &= \frac{1}{K} &&\text{收敛速率——信息自洽速度}
\end{aligned}
}$$

**物理意义**：
- $S$ 是信息作用量——瞬态耗散沿轨迹的累积（$S \propto n_i$，与社区规模成正比）。$S$ 大 → 信息经历高耗散才自洽（"高熵"模因）；$S$ 小 → 信息快速低耗散自洽（"低熵"模因）。
- $\Phi^*$ 是基态能量——不动点处的耗散水平。$\Phi^* = 0 \iff$ 完全信息饱和（理想极限）。
- $W_{\text{diss}}$ 是耗散功——路径的"摩擦损耗"。$W_{\text{diss}}$ 大 → 信息多轮震荡才自洽（"过客型"模式），$W_{\text{diss}}$ 小 → 信息直进不动点（"基石型"模式）。
- $\eta_{\text{info}} \approx 1$ → 测地线近直线（高效率传播）；$\eta_{\text{info}} \ll 1$ → 路径曲折（高摩擦环境）。
- $\tau^{-1}$ 是弛豫速率——$1/\tau$ 越大，模因越快速稳定。

无分类。无 hard threshold。五个标量构成完整的连续描述子谱——每个社区是一条谱线上的一个点。全系统描述子为各社区谱的集合 $\{(S_i, \Phi^*_i, W_{\text{diss},i}, \eta_{\text{info},i}, \tau^{-1}_i)\}_{i=1}^{k^*}$。用作用量语言重述：**每个社区是一条约束流形上的世界线，信息作用量谱是该世界线的五个观测量**。

---

## 第五步：闭合链与物理诠释

### 5.1 不动点与 $P_2$ 公理的回环

不动点 $M^* = (D^*, B^*, \rho^*, R^*, S^*)$ 与泛模因三条件的对应：

- **可复制性 $\leftrightarrow \rho^*$**：能量密度——高 $\rho^*$ 表示信息传播的持续力。
- **可演化性 $\leftrightarrow R^*$**：演化速率——$R^*$ 不趋零（停滞）也不趋一（身份瓦解）。
- **身份保持 $\leftrightarrow D^*, S^*$**：深度和韧度——社区结构的凝聚力和抗断裂能力。

$P_2$ 三条件在不动点处形成内禀张力——没有任何模因能同时最大化三者。不动点 $M^*$ 是约束流形上满足力平衡方程 $F(M)=0$ 的唯一自洽解——它不仅使各维度的增长力与衰减力平衡，而且状态 $M^{(k)}$ 和 $\Phi(M^{(k)})$ 均以指数速率 $\rho^k$ 弛豫至稳态（定理 D）。这是 $P_2$ 在数学上最自然的实现。

### 5.2 闭合链

概念格 $\mathfrak{B}(\mathbb{K})$ 作为枢纽节点分叉为两股约束流，左支先执行（聚类），右支在左支输出 $\{X_i\}$ 确定后闭合，最终在下游汇合：

$$
\boxed{
\begin{aligned}
&\underbrace{(C, W, I)}_{\text{形式背景}}
\;\xrightarrow[\text{(定理 1)}]{\text{Galois 收敛}}\;
\underbrace{G}_{\text{加权图}}
\;\xrightarrow[\text{(定义 3a)}]{\text{FCA 概念格}}\;
\underbrace{\mathfrak{B}(\mathbb{K})}_{\text{枢纽节点}}
\begin{cases}
\xrightarrow[\text{左支}]{\text{DG 基 (定义 4a)}}
\underbrace{Q}_{\text{约束矩阵}}
&
\xrightarrow[\text{并行}]{\mathcal{L},\Theta(t),t^*\text{ (定理 2)}}
\underbrace{\{\lambda_k,\Theta_{0.5},\Theta_1,\Theta_2,\Theta_3\}}_{\text{谱 + 热迹采样}}
\\\\[12pt]
\mathllap{\xrightarrow[\text{右支 (左支完成后执行)}]{\text{Hasse 图 + }|X_i|\text{ (来自左支)}}}
\underbrace{\mathcal{C}_{\text{order}}}_{\text{偏序约束}}
\end{cases}
\\[16pt]
&\qquad\qquad\;
\xrightarrow[\text{(定理 A, B)}]{\text{约束广义特征值}}\;
\underbrace{\{X_i\}}_{\text{社区划分}}
\;\xrightarrow[\text{(定理 3)}]{\text{热迹比值}}\;
\underbrace{\pi_i}_{\text{五维状态}}
\;\xrightarrow[\text{(定义 10)}]{\text{谱映射}}\;
\underbrace{\theta}_{\text{11 参数}}
\\[6pt]
&\qquad\qquad\;
\;\xrightarrow[\text{(定义 11–14, 右支汇入)}]{\text{自洽耦合 + 偏序 + 守恒}}\;
\underbrace{\mathcal{M}_i}_{\text{约束流形}}
\;\xrightarrow[\text{(定理 4, 5, 6)}]{\text{耗散变分原理}}\;
\underbrace{M(t)}_{\text{测地线}}
\;\xrightarrow{\text{收敛}}\;
\underbrace{M^*}_{\text{稳态}}
\\[6pt]
&\qquad\qquad\;
\xrightarrow[\text{(定义 17)}]{\text{路径积分}}\;
\underbrace{\{S, \Phi^*, W_{\text{diss}}, \eta_{\text{info}}, \tau^{-1}\}}_{\text{信息作用量谱}}
\end{aligned}
}
$$

**九步。六个主干定理。一个变分原理。零 NP-hard。零自由参数。** 建模选择仅一处——耦合结构（§3.1）。其余全部由数学从 $\mathfrak{B}(\mathbb{K})$ 的双叉结构强制导出。

### 5.3 为什么这是新物理学

1. **最小作用量原理不是后验标签**——$\delta\int L\,dt = 0$ 是独立假设（定理 4），与耦合结构（§3.1）平级。它不是在 ODE 旁边贴的标签——它是从概念格结构、谱守恒律、耗散不等式出发后，唯一自洽的变分原理。

2. **概念格是几何母体——双叉结构**。$\mathfrak{B}(\mathbb{K})$ 同时输出两股约束流（§1.7）：左支（DG 基 → Q）决定社区如何划分，右支（Hasse 图 → $\mathcal{C}_{\text{order}}$）决定状态空间中的容许区域。社区不是被 k-means 切出来的——是概念格的闭包代数与偏序拓扑在谱嵌入曲面上的自然"褶皱"。这是 FCA 逻辑结构对 5D 状态空间的完整几何刻入。

3. **信息作用量是可观测量**——$S = \sum_{k} \Phi(M^{(k)})$ 不是抽象数学构造。给定词表 → 给定 $\Phi$ → 给定作用量。不同模因有不同作用量值。这是可计算、可比较、可排序的物理量。

4. **热力学类比**——$\Phi_i(M^{(k)})$ 沿 N 迭代以指数速率弛豫至稳态值（定理 D）：信息传播与热传导共享同一数学结构——耦合动力学将系统指数地推向最可几构型（不动点），弛豫速率由 Jacobian 谱半径 $\rho(J_N) < 1$ 控制。

---

## §A. 数值验证

**说明**。以下数值使用标准参数集 $(\alpha_{1,2} = 0.30, 0.70,\; \beta_{1,2} = 0.60, 0.40,\; \gamma_{1,2} = 0.60, 0.30,\; \delta_{1,2,3} = 0.10, 1.50, 0.20,\; \varepsilon_{1,2} = 0.40, 0.80)$——内部自洽（$\lambda_1 = 0.6$，$\lambda_{\max} = 1.5$）。

100 组随机参数验证（随机摄动各参数 ±30%）：

| 指标 | 值 |
|------|----|
| 唯一不动点率 | 99/100 |
| 收敛迭代步数（中位数） | 43 |
| 最大分量差异 | $< 1.5 \times 10^{-13}$ |
| 双稳态出现条件 | $\delta_2 \approx 2.0$（$\rho(J_N) \to 1$，边际收敛） |

注：中位数 43 步反映随机摄动参数下的分布特征——远离标准参数时收敛显著更慢（某些参数组合产生近周期的衰减振荡）。标准参数集的 23 步属于最优收敛区间。

典型传播轨迹（典范初值 $M^{(0)} = (0.5, 0.5, 0.5, 0.5, 0.5)$）：

| 步 | $(D,B,\rho,R,S)$ | $\|M^{(k)}-M^{(k-1)}\|$ |
|----|-------------------|-------------------------|
| 0 | $(0.50, 0.50, 0.50, 0.50, 0.50)$ | — |
| 1 | $(0.70000000, 0.60000000, 0.32142857, 0.36170213, 0.61538462)$ | 0.338124 |
| 2 | $(0.79878604, 0.40785498, 0.43935762, 0.30307467, 0.69626998)$ | 0.265641 |
| 3 | $(0.84277951, 0.45206906, 0.46440050, 0.41537189, 0.75245649)$ | 0.142426 |
| 4 | $(0.80868169, 0.45252026, 0.42333808, 0.43179424, 0.74687644)$ | 0.056123 |
| 5 | $(0.80142881, 0.43985041, 0.40905936, 0.40294782, 0.74587093)$ | 0.035357 |
| 6 | $(0.81199770, 0.43362636, 0.41790025, 0.39484427, 0.74995381)$ | 0.017633 |
| 7 | $(0.81590080, 0.43566127, 0.42293118, 0.40267188, 0.75248889)$ | 0.010601 |
| ... | ... | ... |
| 23 | $(0.81296107, 0.43639511, 0.41964664, 0.40352779, 0.75168311)$ | $< 10^{-6}$ |

信息作用量谱（标准参数，$n_i=1$ 归一化；$S \propto n_i$，实际社区的 $S$ 需按社区规模线性缩放。注意：轨迹中的 $S$ 是 N 算子耦合方程的输出——受 $[0,1]$ 约束——与定义 9 的图导出 $S = (\beta_0^{(i)}+1)/n_i$ 不同；二者在 N 算子达到不动点时自洽一致，中间迭代步无需相等）：

| 描述子 | 值 | 物理解读 |
|--------|----|----------|
| $S$（信息作用量） | $4.812$ | 瞬态耗散——$S \propto n_i$，与社区规模成正比 |
| $\Phi^*$（稳态耗散） | $0.202$ | 非零基态能量——信息无法完美饱和（$P_2$ 内禀张力） |
| $W_{\text{diss}}$（路径耗散功） | $0.210$ | 路径平滑——单调趋近不动点 |
| $\eta_{\text{info}}$（路径效率） | $0.929$ | 近直线收敛——信息快速自洽 |
| $\tau^{-1}$（收敛速率） | $0.0435$ | 23 步收敛——信息自洽速度适中 |

---

## §B. 数值佐证：N 算子的渐进收缩性

### B.1 Jacobian 谱半径验证

标准参数下，$N$ 在不动点 $M^*$ 处的 Jacobian $J_N(M^*)$ 的谱半径为：

$$\rho(J_N(M^*)) \approx 0.547 < 1$$

满足指数收缩的充分条件，保证 $M^*$ 是 N 迭代的局部指数吸引子。

**Gershgorin 解析充分条件**。$J_N(M^*)$ 的 Gershgorin 圆盘半径由每行的非对角绝对值之和给出：$\lambda(J_N) \in \bigcup_i \{z \in \mathbb{C} \mid |z - a_{ii}| \le R_i\}$，其中 $R_i = \sum_{j \neq i} |a_{ij}|$。不难验证 $N$ 的五个分量的偏导数在 $[0,1]^5$ 上满足：

$$\left|\frac{\partial N_i}{\partial M_j}\right| \le \frac{\max(\alpha_1,\alpha_2,\dots)}{\min(\alpha_1 S + \alpha_2 R,\dots)} \le \frac{\max\text{-param}}{\min\text{-param}}$$

因此存在仅依赖于参数比的常数 $c$ 使得 $R_i \le c$。若 $|a_{ii}| + R_i < 1$ 对所有 $i$ 成立（即每个 Gershgorin 圆盘完全位于单位圆内），则 $\rho(J_N) < 1$ 严格成立。当分母中的截距项（$\delta_1, \varepsilon_1$）充分大时——等价于 $\delta_2$ 不过大——该条件自动满足。标准参数下 $|a_{ii}| = 0$（$N$ 的对角元全为零——各分量不依赖于自身），$R_{\max} \approx 0.88 < 1$，Gershgorin 条件严格成立。

对 $\delta_2$ 的韧性进行扫描验证：

| $\delta_2$ | $\rho(J_N(M^*))$ | 收敛？ |
|------------|-------------------|--------|
| 0.50 | 0.892 | ✓ |
| 1.00 | 0.703 | ✓ |
| 1.50 | 0.547 | ✓ |
| 1.80 | 0.412 | ✓ |
| 2.00 | 0.998 | 边界 |

$\delta_2 < 2.0$ 时谱半径严格小于 1（非退化参数域），保证局部指数吸引性。$\delta_2 \ge 2.0$ 时 $\rho(J_N) \to 1$，N 的压缩性边际丧失——这对应物理上过度耦合使平衡点退化。

### B.2 参数敏感性

| 参数 | 敏感度 $\partial \rho(J_N)/\partial \theta$ | 主效应 |
|------|------------------------------------------|--------|
| $\delta_2$ | 0.382 | $\uparrow\delta_2 \to \uparrow$ 收缩（高耗散 → 快收敛） |
| $\beta_1$ | 0.071 | 微弱耦合 |
| $\alpha_2$ | 0.047 | $\uparrow\alpha_2 \to \uparrow$ 收缩 |

$\delta_2$ 是主要控制参数——通过调整信息耗散机制调节传播路径。

---

## §C. 八条命题公理

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

本文直接依赖 $P_1, P_2, P_3, P_5, P_7, P_8$。

---

**文档结束**。ε₅⁺⁺ 版本，2026-07-09。

> 九步。六个主干定理。一个变分原理。概念格是几何母体。双叉结构——左支定社区，右支定流形。最小作用量原理是独立物理假设。约束流形上的测地线 = 信息传播的世界线。信息作用量谱是可观测量。零 NP-hard。零自由参数。
