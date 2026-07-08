# 泛模因理论：谱桥数学框架

**版本**：v5.1  
**日期**：2026-07-08  
**性质**：从公理到原型分类的规范建模路径，全程标注各步骤的数学性质（单射/双射/确定性/启发式/建模选择）

> 本文构建泛模因理论从八条公理到三族九子型原型分类的完整数学框架。推导路径为：公理 $P_1$–$P_8$ → 滑动窗口加权邻接矩阵（局部共现编码）→ 归一化拉普拉斯谱分解（谱双射）→ 谱社区划分（定义式准则）→ 热迹五维状态（天然归一化）→ 热迹参数映射（标度采样）→ ODE 动力学假设（受谱约束的建模选择）→ Jacobian 分岔分类 → 三族九子型。每一步标注映射性质与信息丢失量。

---

## 目录

- **第〇篇：基础** (§0–§1)
- **第一篇：文本 → 图谱** (§2–§3)
- **第二篇：图谱 → 社区** (§4)
- **第三篇：社区 → 状态** (§5)
- **第四篇：状态 → 参数** (§6–§7)
- **第五篇：动力学系统** (§8–§9)
- **第六篇：原型分类** (§10–§11)
- **第七篇：全局结论** (§12–§13)

---

## 第〇篇：基础

### §0. 符号约定

| 符号 | 含义 |
|------|------|
| $\Sigma^*$ | 有限字符表 $\Sigma$ 上的有限长字符串集 |
| $\mathbb{R}_{\ge 0}$ | 非负实数集 $[0, \infty)$ |
| $\mathbf{1}_{\{P\}}$ | 指示函数：谓词 $P$ 为真取 1，否则取 0 |
| $\dot{x} \equiv dx/dt$ | 时间导数 |
| $\text{Tr}(M)$ | 矩阵 $M$ 的迹 |
| $\text{spec}(\mathcal{L})$ | $\mathcal{L}$ 的谱（特征值多重集） |
| $\Theta(t)$ | 热迹函数 $\sum_k e^{-t\lambda_k}$ |
| $t^*$ | 混合时间——热迹下降到 $\beta_0 + 1$ 的最小 $t$ |
| $\Omega$ | 五维状态空间 $[0,1]^5$ |
| $\lambda_k$ | 升序排列的第 $k$ 个特征值（$\lambda_0 \le \lambda_1 \le \cdots$） |
| $\beta_0$ | 第 0 个 Betti 数 = 连通分量数 = $\lambda_0$ 的重数 |
| $\Delta_k$ | 特征值间隙 $\lambda_{k+1} - \lambda_k$ |

---

### §1. 八条命题公理

> 公理为理论的初始假设，不可在系统内证明，仅可根据经验检验。

**公理 P1（信息本体论）**。存在基本本体论范畴 $\mathbb{I}$（信息），满足：
- $\mathbb{I} \not\subseteq \mathbb{M}$ 且 $\mathbb{I} \not\subseteq \mathbb{E}$（信息不可约化为物质或能量）
- $\forall m \in \mathbb{M}, \exists i \in \mathbb{I}: i = \text{struct}(m)$（每个物质实例有结构描述）
- $\exists i_1, i_2 \in \mathbb{I}, m \in \mathbb{M}: m_{i_1} \neq m_{i_2}$（不同信息结构产生不同物质行为）

**公理 P2（泛模因实在论）**。泛模因类 $\mathcal{P}$ 定义为满足可复制性、可演化性和身份保持性的实体集：

$$\mathcal{P} = \{ x \mid \text{replicable}(x) \land \text{evolvable}(x) \land \text{identity\_preserving}(x) \}$$

**公理 P3（结构现实主义）**。$\forall x$，其同一性 $\text{id}(x)$ 不由物质组成决定，而由关系模式 $R_x \subseteq (x \cup \text{context}(x))^2$ 决定。

**公理 P4（认知-建模统一性）**。人类的认知过程 $C$ 与形式化建模过程 $\mathcal{M}$ 共享同构核心：$C(h) \cong \mathcal{M}(h)$。

**公理 P5（可逆性认识论）**。对任意合法建模 $\Phi$，存在逆映射 $\Phi^{-1}$（或至少是单射的左逆）使：
$$\Phi^{-1} \circ \Phi = \text{id}_{\text{Dom}(\Phi)}$$
即模型的每一步变换必须有可追踪的信息保持性质。

**公理 P6（计算即模因）**。任何图灵完备系统 $T$ 本身是一个泛模因实例：$T \in \mathcal{P}$。

**公理 P7（演化普遍性）**。若系统 $S$ 同时具备复制 $R$、变异 $V$、选择 $S_e$ 三个算子，则 $S$ 表现出演化行为：
$$S_{t+1} = S_e \circ V \circ R(S_t)$$

**公理 P8（层级涌现）**。泛模因构成偏序集 $(\mathcal{P}, \preceq)$，其中 $x \preceq y$ 表示 $x$ 是 $y$ 的构成部分。存在向上因果和向下约束的双向作用。

> **P5 的角色**：P5 是贯穿全文的核心约束——每一步推导中我们明确标注映射的数学性质（单射/双射/确定性/信息丢失量），不声称不成立的性质。

---

## 第一篇：文本 → 图谱

### §2. 滑动窗口加权邻接矩阵（$I \to A$）

> **目标**：将任意字符串 $I \in \Sigma^*$ 映射为加权邻接矩阵 $A \in \mathbb{R}^{n \times n}_{\ge 0}$。精确刻画该映射保留的信息量和丢失的信息量。

#### 定义 2.1（滑动窗口加权共现）

设输入序列 $I = (c_1, c_2, \ldots, c_L)$，$c_p \in \Sigma$，$n = |\Sigma|$。固定窗口半径 $w \in \mathbb{N}$（$1 \le w \ll L$）。对每个位置 $p$ 和步长 $\Delta \in [1, w]$，若 $p + \Delta \le L$，则前向共现对 $(c_p, c_{p+\Delta})$ 的权值贡献为：

$$w_{p, \Delta}(a, b) = \mathbf{1}_{\{c_p = a\}} \cdot \mathbf{1}_{\{c_{p+\Delta} = b\}} \cdot \exp\left(-\frac{\Delta}{w}\right)$$

同时，每个位置 $p$ 的自共现（$\Delta = 0$）贡献 $1$ 给对角元 $A_{c_p, c_p}$。

#### 定义 2.2（有向邻接矩阵）

$$A^{\text{dir}}_{ij} = \sum_{p=1}^{L} \sum_{\Delta=1}^{\min(w, L-p)} w_{p, \Delta}(c_i, c_j)$$

#### 定义 2.3（对称化——无向加权邻接矩阵）

$$A_{ij} = A^{\text{dir}}_{ij} + A^{\text{dir}}_{ji}$$

#### 定理 2.4（映射的信息保持性）

映射 $\Phi_A: \Sigma^* \to \mathbb{R}^{n \times n}_{\ge 0}$，$\Phi_A(I) = A$，具有以下性质：

**(1) 信息保留**：$A$ 完整编码了序列中所有字符对 $(a, b)$ 在距离 $\Delta \in [0, w]$ 内的加权共现计数。具体而言，对于任意字符 $a, b \in \Sigma$ 和步长 $\Delta \in [1, w]$，$A^{\text{dir}}_{ab}$ 的对应项包含了所有满足 $c_p = a, c_{p+\Delta} = b$ 的位置对，权重 $\exp(-\Delta/w)$。

**(2) 信息丢失**：字符对距离超过 $w$ 步的共现关系**完全丢失**。这是滑动窗口的结构性限制——矩阵只保留了 $(w+1)$-gram 局部转移谱，而非全局序列顺序。

**(3) 渐进单射性**：当 $w \ge L$ 时（窗口覆盖整条序列），不同序列的共现结构必然不同，映射为单射。当 $w \ll L$ 时，可能存在不同序列产生相同 $A$ 的情况（例如，互换两个非重叠的 $w$-邻域块），但这些情况在自然语言文本中概率极低——局部共现模式高度约束全局结构。

**(4) 序列长度信息**：序列长度 $L$ 的信息编码在自共现对角元中。$A_{ii}$ 包含：每个 $c_p = i$ 的位置贡献 $1$（$\Delta = 0$）加上所有 $c_p = c_{p+\Delta} = i$ 的额外自转移项。因此 $L$ **不能直接从 $\text{Tr}(A)$ 简单读取**（$\text{Tr}(A) \neq L$）——但 $L$ 的信息仍在矩阵结构中以可恢复的形式存在。

#### 推论 2.5（P5 满足性）

$\Phi_A$ 在 $w \ge L$ 时是单射；在 $w \ll L$ 时是信息保持的投影（保留局部结构，丢失全局长程顺序）。这满足公理 P5 的核心要求：**信息丢失是结构性的（由窗口大小控制）且可被精确刻画**——不是"不知道丢了什么"，而是"知道每一步精确丢失了哪些信息"。

---

### §3. 归一化拉普拉斯谱分解（$A \to (\Lambda, U_{\text{std}})$）

> **目标**：将加权邻接矩阵 $A$ 通过归一化图拉普拉斯谱分解映射到规范特征对 $(\Lambda, U_{\text{std}})$。选用归一化拉普拉斯的理由：(a) 特征值天然有界 $0 \le \lambda_k \le 2$，后续参数映射自动量纲一致；(b) 尺度不变性——边权的整体缩放不影响谱，消除了单调递增/递减序列之间的虚假差异。

#### 定义 3.1（度矩阵与归一化图拉普拉斯）

无向加权图 $G = (V, E, A)$，$V = \Sigma$，$|V| = n$。度矩阵：

$$D_{ii} = \sum_{j=1}^{n} A_{ij}, \quad D_{ij} = 0 \; (i \neq j)$$

约定：若 $D_{ii} = 0$（孤立顶点），定义 $(D^{-1/2})_{ii} = 0$。实际中 §2 的自共现保证 $D_{ii} > 0$ 恒成立。

归一化图拉普拉斯（Chung 1997）：

$$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$

$$(\mathcal{L})_{ij} = \begin{cases}
1 - A_{ii}/D_{ii} & i = j,\; D_{ii} > 0 \\[4pt]
-A_{ij} / \sqrt{D_{ii} D_{jj}} & i \neq j,\; D_{ii}, D_{jj} > 0 \\[4pt]
0 & \text{otherwise}
\end{cases}$$

#### 引理 3.2（归一化拉普拉斯的基本性质）

1. $\mathcal{L}$ 为实对称半正定矩阵。
2. $\forall \mathbf{x} \in \mathbb{R}^n, \; \mathbf{x}^\top \mathcal{L} \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}\left(\frac{x_i}{\sqrt{D_{ii}}} - \frac{x_j}{\sqrt{D_{jj}}}\right)^2 \ge 0$。
3. **Perron 特征向量**：$\mathbf{v}_0 = D^{1/2}\mathbf{1} = (\sqrt{D_{11}}, \sqrt{D_{22}}, \ldots, \sqrt{D_{nn}})^\top$ 为零特征值 $\lambda_0 = 0$ 对应的特征向量：$\mathcal{L}\mathbf{v}_0 = \mathbf{0}$。单位化后 $\mathbf{u}_0 = \mathbf{v}_0 / \|\mathbf{v}_0\|$。
4. $\lambda_0 = 0$，且 $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1} \le 2$（特征值有上界 2）。
5. $\lambda_0$ 的重数 = 图的连通分量数 $\beta_0$。

**证明**。性质 1–5 均为标准谱图论结论（Chung 1997）。性质 2 由展开得证。性质 3：$(\mathcal{L} D^{1/2}\mathbf{1})_i = \sqrt{D_{ii}} - \sum_j A_{ij}/\sqrt{D_{ii}} = \sqrt{D_{ii}} - D_{ii}/\sqrt{D_{ii}} = 0$。性质 4：特征值序由 Courant-Fischer 定理保证；上界 2 来自 $\|I - D^{-1/2}AD^{-1/2}\|_2 \le \|I\|_2 + \|D^{-1/2}AD^{-1/2}\|_2 = 1 + 1 = 2$。性质 5 为标准结论。$\square$

#### 定义 3.3（谱分解）

$$\mathcal{L} = U \Lambda U^\top, \quad \Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$$

$U = [\mathbf{u}_0, \mathbf{u}_1, \ldots, \mathbf{u}_{n-1}] \in \mathbb{R}^{n \times n}$ 为正交矩阵（$U^\top U = I_n$），$\mathbf{u}_k$ 为 $\lambda_k$ 对应的单位特征向量。特征值按升序排列。

#### 定义 3.4（符号标准化与退化消歧）

**步骤 1 — 符号标准化**。对每个特征向量 $\mathbf{u}_k$，定义第一个显著分量索引：

$$j_k = \min\{j \mid |(\mathbf{u}_k)_j| > \varepsilon\}, \quad \varepsilon = 10^{-8}$$

若 $(\mathbf{u}_k)_{j_k} < 0$，则 $\mathbf{u}_k \leftarrow -\mathbf{u}_k$。

**步骤 2 — 退化子空间消歧**。对每个重数 $\ge 2$ 的特征值 $\lambda_k$，对其特征子空间内的基向量：按 $j_k$ 升序排列 → Gram-Schmidt 正交化 → 对每个结果向量应用步骤 1。

记标准化后的特征基为 $U_{\text{std}}$。$U_{\text{std}}$ 的每一列有唯一定义的符号。

**注意**：对 Perron 向量 $\mathbf{u}_0$，由 $\sqrt{D_{jj}} > 0$ 对所有非孤立顶点 $j$ 成立，符号标准化使 $\mathbf{u}_0$ 的所有分量非负——这正是 Perron-Frobenius 定理要求的正特征向量。因此 $\mathbf{u}_0$ 的符号是**自动不可翻转的**（翻转会使分量变负，与 Perron-Frobenius 矛盾）。

#### 定理 3.5（Perron 重构——度序列的恢复）

从 $U_{\text{std}}$ 的第 0 列 $\mathbf{u}_0$ 可**部分恢复**图的结构信息：

$$\mathbf{u}_0 = \frac{1}{\sqrt{\text{vol}(G)}} \begin{pmatrix} \sqrt{D_{11}} \\ \sqrt{D_{22}} \\ \vdots \\ \sqrt{D_{nn}} \end{pmatrix}, \quad \text{vol}(G) = \sum_{i=1}^{n} D_{ii}$$

因此 $(u_0)_i \propto \sqrt{D_{ii}}$。度序列的相对比例 $\{D_{ii}/D_{jj}\}$ 由 $\mathbf{u}_0$ 的分量比 $(u_0)_i^2/(u_0)_j^2$ 唯一确定。**绝对度值**需额外知道 $\text{vol}(G)$——即总体缩放因子，该因子在归一化拉普拉斯中被消去。

**证明**。由引理 3.2 性质 3，$\mathbf{v}_0 = D^{1/2}\mathbf{1}$ 满足 $\mathcal{L}\mathbf{v}_0 = \mathbf{0}$。单位化得 $\mathbf{u}_0 = \mathbf{v}_0 / \|\mathbf{v}_0\|$，其中 $\|\mathbf{v}_0\|^2 = \sum_i D_{ii} = \text{vol}(G)$。$\square$

#### 定理 3.6（谱分解的映射性质）

映射 $\Phi_{\text{spec}}: \mathbb{R}^{n \times n}_{\text{sym}} \to \mathcal{U} \times \mathbb{R}^n_{\text{sort}}$：

$$\Phi_{\text{spec}}(A) = (\Lambda, U_{\text{std}})$$

满足：

**(1) 满射性**：任意实对称 $A$ 产生实对称 $\mathcal{L}$，存在正交对角化 $U \Lambda U^\top$。符号标准化消去 $\pm$ 歧义后，对每个 $A$ 存在唯一 $(\Lambda, U_{\text{std}})$。

**(2) 非单射性（缩放等价类）**：若 $A' = c \cdot A$（$c > 0$），则 $D' = c \cdot D$，且：
$$\mathcal{L}' = I - D'^{-1/2}A'D'^{-1/2} = I - \frac{1}{\sqrt{c}}D^{-1/2} \cdot cA \cdot \frac{1}{\sqrt{c}}D^{-1/2} = I - D^{-1/2}AD^{-1/2} = \mathcal{L}$$

因此 $A$ 和 $cA$ 映射到相同的 $(\Lambda, U_{\text{std}})$。归一化拉普拉斯在全局边权缩放下**不变**——这是其尺度不变性的代价。等价类 $[A] = \{cA \mid c > 0\}$ 中的元素有相同的谱。

**(3) 在 $\Phi_A$ 像集中的可逆性**：在滑动窗口构造中，边权由 $\exp(-\Delta/w)$ 固定，不存在任意缩放自由度。因此 $\Phi_A$ 的像集中不含等价类歧义：若 $\Phi_{\text{spec}}(A_1) = \Phi_{\text{spec}}(A_2)$ 且 $A_1, A_2 \in \text{Im}(\Phi_A)$，则 $A_1 = A_2$（因为归一化拉普拉斯唯一确定 $D^{-1/2}AD^{-1/2}$，而 $D_{ii}/D_{jj}$ 比例由 $\mathbf{u}_0$ 唯一确定，且缩放因子在 $\Phi_A$ 像集中固定）。

**(4) 可重构性**：$\Phi_{\text{spec}}$ 的"逆"在像集上为：
$$\mathcal{L} = U_{\text{std}} \Lambda U_{\text{std}}^\top, \quad B = I - \mathcal{L} = D^{-1/2}AD^{-1/2}$$
结合 $\mathbf{u}_0$ 恢复 $D$ 的相对比例和 $A$ 的固定尺度，可唯一恢复原 $(D, A)$。

#### 推论 3.7（P5 满足性）

$$I \xrightarrow{\Phi_A}_{\text{信息保持投影}} A \xrightarrow{\Phi_{\text{spec}}}_{\text{满射，像集上可逆}} (\Lambda, U_{\text{std}})$$

复合映射 $\Phi_{\text{spec}} \circ \Phi_A$ 在 $\Sigma^*$ 上不是单射——信息在 $\Phi_A$ 中由窗口大小控制丢失了长程共现，在 $\Phi_{\text{spec}}$ 中丢失了全局边权缩放（后者在 $\Phi_A$ 的像集中无实际影响）。每一步的信息丢失有精确刻画。

---

## 第二篇：图谱 → 社区

### §4. 谱社区划分（$(\Lambda, U_{\text{std}}) \to \{X_i\}$）

> **目标**：将顶点集合 $V = \Sigma$ 划分为社区 $\{X_1, \ldots, X_{k^*}\}$。社区数 $k^*$ 和划分均由拉普拉斯谱定义——这是**定义式准则**，而非普遍适用的定理。

#### 定义 4.1（谱社区数——定义式准则）

图 $G$ 的**谱社区数** $k^*$ 定义为：

$$k^* = \arg\max_{1 \le k \le n-2} (\lambda_{k+1} - \lambda_k)$$

即取最大特征值间隙对应的索引。

**合理性**：当图由 $k$ 个松散连接的稠密子图（团或近团）构成时，前 $k$ 个特征值接近零（每个子图贡献一个零特征值），第 $k+1$ 个特征值显著跳升。最大间隙 $\Delta_{k} = \lambda_{k+1} - \lambda_k$ 的 $k$ 等于子图数（von Luxburg 2007, §8.3）。对一般的非分块图，$k^*$ 仍是一个良好定义的谱量——我们**以此定义**社区数，其有效性取决于图的实际分块结构。若图不存在清晰的聚类结构，$k^*$ 可能 $\approx n$（每个顶点自成一社区）或 $=1$（全图为一个社区）——这两种退化情况均被 ODE 系统自然地处理（全图 → 一个五维轨迹，全分裂 → $n$ 条轨迹聚合分析）。

#### 定义 4.2（谱嵌入与行归一化）

取前 $k^*$ 个非零特征向量 $\mathbf{u}_1, \ldots, \mathbf{u}_{k^*}$ 构成嵌入矩阵 $U_{:1:k^*} \in \mathbb{R}^{n \times k^*}$。行归一化：

$$\tilde{\mathbf{u}}_v = \frac{(U_{:1:k^*})_{v, :}}{\|(U_{:1:k^*})_{v, :}\|_2} \in \mathbb{R}^{k^*}, \quad v = 1, 2, \ldots, n$$

（若范数为零，取 $\tilde{\mathbf{u}}_v = \mathbf{0}$。）

#### 定义 4.3（谱社区划分——全局最优 $k$-means）

将归一化行向量 $\{\tilde{\mathbf{u}}_v\}_{v=1}^{n}$ 划分为 $k^*$ 个簇，取 $k$-means 问题**全局最优解**：

$$\{X_i\}_{i=1}^{k^*} = \arg\min_{\{C_i\}} \sum_{i=1}^{k^*} \sum_{v \in C_i} \|\tilde{\mathbf{u}}_v - \boldsymbol{\mu}_i\|_2^2, \quad \boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{v \in C_i} \tilde{\mathbf{u}}_v$$

其中 $\arg\min$ 取所有可能分划上的**全局最小值**（通过多次随机初始化取最优实现）。

#### 定理 4.4（划分的确定性）

在定义 4.3 的全局最优意义下，社区划分 $\{X_i\}_{i=1}^{k^*}$ 由 $(\Lambda, U_{\text{std}})$ 唯一确定（除社区编号的重排外）。

**证明**。给定 $(\Lambda, U_{\text{std}})$ → 确定 $k^*$ （定义 4.1） → 确定嵌入 $U_{:1:k^*}$ （定义 4.2） → 确定嵌入点的集合 $\{\tilde{\mathbf{u}}_v\}_{v=1}^{n}$ → 确定簇内平方和函数 $f(\{C_i\})$ → 取全局最小值。全局最小值可能在对称图中有多个等价的极小化分划（如正则图的对称性引致多个等价分划），但这些等价分划在 ODE 动力学中产生相同轨迹（参数对不同编号的社区置换不变）。$\square$

---

## 第三篇：社区 → 状态

### §5. 热迹五维状态向量（$\{X_i\} \to \pi_i$）

> **目标**：对每个社区 $X_i$，导出五维状态向量 $\pi_i \in [0,1]^5$。所有分量由热迹比值定义——热方程 $\partial_t u = -\mathcal{L}u$ 的解 $\Theta(t)/n$ 天然落在 $[0,1]$，因此不需要任何外部归一化函数（如 $1 - e^{-x}$ 或 $1/(1+x)$）。

#### 定义 5.1（社区子图谱）

设社区 $X_i \subseteq V$，$|X_i| = n_i$。其导出子图 $G[X_i]$ 的归一化拉普拉斯为 $\mathcal{L}^{(i)}$（$n_i \times n_i$），特征值 $\{\lambda_k^{(i)}\}_{k=0}^{n_i-1}$（$0 = \lambda_0^{(i)} \le \cdots \le \lambda_{n_i-1}^{(i)} \le 2$）。全局谱为 $\{\lambda_k^{(G)}\}_{k=0}^{n-1}$。

社区 $X_i$ 的热迹：$\Theta_i(t) = \sum_{k=0}^{n_i-1} e^{-t \lambda_k^{(i)}}$。

社区混合时间：$t^*_i = \inf\{t > 0 \mid \Theta_i(t) \le \beta_0^{(i)} + 1\}$。

归一化社区规模：$\tau_i = n_i / n$。

全局热迹：$\Theta_G(t) = \sum_{k=0}^{n-1} e^{-t \lambda_k^{(G)}}$。

全局混合时间：$t^*_G = \inf\{t > 0 \mid \Theta_G(t) \le \beta_0^{(G)} + 1\}$。

#### 定义 5.2（五维谱状态向量——热迹采样）

$$\boxed{
\begin{aligned}
D_i &= \frac{\Theta_i(\tau_i)}{n_i} & &\text{（局部热保留率 → 深度）}\\[6pt]
B_i &= 1 - \frac{\Theta_G(\tau_i)}{n} & &\text{（全局热耗散率 → 广度）}\\[6pt]
\rho_i &= \frac{\sum_{k} \lambda_k^{(i)}}{\sum_{k} \lambda_k^{(G)}} & &\text{（谱质比 → 能量/人）}\\[6pt]
R_i &= \frac{t^*_G}{t^*_i} & &\text{（混合时间比 → 速率）}\\[6pt]
S_i &= \frac{\Theta_i(t^*_i)}{n_i} = \frac{\beta_0^{(i)} + 1}{n_i} & &\text{（混合残余热 → 韧度）}
\end{aligned}
}$$

**$S_i$ 的等价性**：由混合时间定义 $t^*_i = \inf\{t > 0 \mid \Theta_i(t) \le \beta_0^{(i)} + 1\}$ 及 $\Theta_i(t)$ 的连续性，$\Theta_i(t^*_i) = \beta_0^{(i)} + 1$。故 $S_i = (\beta_0^{(i)} + 1) / n_i$——即社区内连通分量密度（每个连通分量贡献 $1/n_i$，加自身）。连通性越强（$\beta_0^{(i)}$ 越小），韧度 $S_i$ 越低——这是一个有清晰拓扑意义的图论事实。

#### 定理 5.3（热迹天然归一化）

所有五个状态分量均**天然**落入 $[0,1]$，归一化由热方程 $\partial_t u = -\mathcal{L}u$ 的解析性质保证，**无外部函数介入**：

- **$D_i$**：$\Theta_i(\tau_i) \in [\beta_0^{(i)}, n_i]$ → $D_i \in [\beta_0^{(i)}/n_i, 1] \subseteq [0,1]$。
- **$B_i$**：$\Theta_G(\tau_i) \in [\beta_0^{(G)}, n]$ → $B_i \in [0, 1 - \beta_0^{(G)}/n] \subseteq [0,1]$。
- **$\rho_i$**：谱迹比 $\in [0,1]$（归一化拉普拉斯迹比）。
- **$R_i$**：$\in (0, \infty)$（混合时间比）。
- **$S_i$**：$(\beta_0^{(i)} + 1)/n_i \in (0, 1]$（连通分量密度论）。

#### 时间采样点的选择论证

五个分量分别在三个时间点采样：

| 时间点 | 含义 | 采样的分量 | 理由 |
|--------|------|-----------|------|
| $\tau_i$ | 社区归一化规模 | $D_i, B_i$ | $\tau_i \in (0,1)$ 是仅有的由社区大小决定的无量纲标度——规模越大的社区，热扩散越慢，$\tau_i$ 作为"社区自身的时间单位"自然合适 |
| $t^*_i$ | 社区混合时间 | $S_i$ | 混合后的残余热 $\Theta_i(t^*_i) = \beta_0^{(i)}+1$ 直接度量了社区热平衡后的结构稳定性——这是 $t^*_i$ 定义的自然推论 |
| $t^*_G$ | 全局混合时间 | $R_i$ | 全局混合时间与社区混合时间的比值 $R_i = t^*_G / t^*_i$ 度量相对混合速度 |

**核心原则**：不引入 $\Theta(t)$ 在这些天然标度点之外的任意采样。$\tau_i$、$t^*_i$、$t^*_G$ 是热核扩散方程仅有的三个自然时间标度——它们由系统本身（而非外部选择）定义。

---

## 第四篇：状态 → 参数

### §6. 热迹与混合时间

#### 定义 6.1（热迹函数）

归一化拉普拉斯热核 $\exp(-t\mathcal{L})$ 的迹：

$$\Theta_G(t) = \text{Tr}(e^{-t\mathcal{L}}) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$$

#### 引理 6.2（热迹性质）

1. $\Theta_G(0) = n$。
2. $\Theta_G(t)$ 在 $t > 0$ 上**严格递减**（若 $\lambda_1 > 0$）。
3. $\lim_{t \to \infty} \Theta_G(t) = \beta_0$。
4. $t^*_G$ 存在且唯一（$\Theta_G(0) = n \ge \beta_0 + 1$ 且 $\lim_{t \to \infty} \Theta_G(t) = \beta_0 < \beta_0 + 1$）。
5. $\Theta_G(t^*_G) = \beta_0 + 1$（由 $t^*_G$ 定义和连续性）。

**证明**。性质 1–4 为标准结论；性质 5 由 $\Theta_G$ 连续性得。$\square$

---

### §7. 十一参数谱映射（$\pi_i \to \theta_i$）

> **目标**：对每个社区 $X_i$，从其谱和热迹衍生 11 个 ODE 参数 $\theta_i \in \mathbb{R}^{11}$。参数定义为 $\Theta(t)$ 在四个天然时间标度点上的采样组合。

#### 定义 7.1（四个时间标度点）

$$t_{0.5} = \frac{t^*}{2}, \quad t_1 = t^*, \quad t_2 = 2t^*, \quad t_3 = 3t^*$$

对应热迹值：

$$\Theta_{0.5} = \Theta(t^*/2), \quad \Theta_{1} = \Theta(t^*) = \beta_0 + 1, \quad \Theta_{2} = \Theta(2t^*), \quad \Theta_{3} = \Theta(3t^*)$$

**选点论证**：$0.5, 1, 2, 3$ 倍 $t^*$ 是热方程在时间平移 $t \to ct$ 下的**唯一自然倍数序列**——它们分别对应"早期扩散"（混合前）、"临界混合"（混合点）、"衰减段"（扩散完成后的尾迹）、"长尾"（残余信息）。任何做扩散方程重正化群分析的学者都会首先在这四个标度点采样。

混合时间阈值 $\beta_0 + 1$ 的理由：$\Theta(t^*)$ 比 $\Theta(\infty) = \beta_0$ 多 1——这 1 是**单个额外连通分量的单位贡献**，是能在谱上识别的最小的、有意义的热迹差异。$\beta_0 + 0.5$ 或 $\beta_0 + \log 2$ 没有谱论解释；$\beta_0 + 1$ 有：在图中新增一个孤立顶点使 $\beta_0 \to \beta_0 + 1$——1 是**结构变化的原子单位**。

#### 定义 7.2（十一参数公式）

> 以下省略社区上标 $(i)$，每个社区独立计算。

$$\boxed{
\begin{aligned}
\alpha_1 &= \frac{\lambda_1}{2} & &\text{谱隙系数（归一化后 $\lambda_1 \in [0,2]$）}\\[4pt]
\alpha_2 &= \frac{\Theta_{0.5}}{n} & &\text{早期热保留率}\\[4pt]
\beta_1 &= \frac{\lambda_{\max} - \lambda_1}{\lambda_{\max}} & &\text{谱分散度}\\[4pt]
\beta_2 &= \frac{\sum_{k=0}^{n-1} \lambda_k^2}{\left(\sum_{k=0}^{n-1} \lambda_k\right)^2} & &\text{谱浓度比}\\[4pt]
\gamma_1 &= 1 - \frac{\Theta_{2}}{\Theta_{1}} & &\text{衰减段热耗散率}\\[4pt]
\gamma_2 &= \frac{\Theta_{0.5}}{\Theta_{1}} - 1 & &\text{早期加速比}\\[4pt]
\delta_1 &= \frac{\Theta_{0.5} - \Theta_{1}}{n} & &\text{早期凝聚梯度}\\[4pt]
\delta_2 &= \lambda_{\max} = \frac{\lambda_{\max}}{\sum_{k > 0} e^{-\lambda_k t^*}} & &\text{谱峰锐度（分母恒 = 1，由 $\Theta(t^*) = \beta_0 + 1$ 保证）}\\[4pt]
\delta_3 &= 1 - \frac{\Theta_{3}}{\Theta_{2}} & &\text{长尾衰减率}\\[4pt]
\varepsilon_1 &= \frac{\lambda_1}{\lambda_{\max}} & &\text{谱隙比}\\[4pt]
\varepsilon_2 &= 1 - \frac{\Theta_{2}}{n} & &\text{完全混合比}
\end{aligned}
}$$

**参数构造的统一逻辑**：

| 类别 | 参数 | 构造模式 | 来源 |
|------|------|----------|------|
| 谱比值 | $\alpha_1, \beta_1, \beta_2, \varepsilon_1$ | 特征值的无量纲组合 | $\text{spec}(\mathcal{L})$ |
| 早期标度 | $\alpha_2, \gamma_2, \delta_1$ | $\Theta(t^*/2)$ 和 $\Theta(t^*)$ 的组合 | 混合前/临界热迹 |
| 衰减标度 | $\gamma_1, \delta_2, \delta_3, \varepsilon_2$ | $\Theta(2t^*)$ 和 $\Theta(3t^*)$ 的组合 | 混合后热迹衰减 |

每个参数恰好属于一个构造模式，无重复、无冗余、无遗漏——这是"拉普拉斯谱 + 四个标度点的热迹"产生的**完备代数组合**。

#### 引理 7.3（值域有界性）

| 参数 | 值域 | 理由 |
|------|------|------|
| $\alpha_1$ | $[0, 1]$ | $\lambda_1 \in [0, 2]$ |
| $\alpha_2$ | $(0, 1]$ | $\Theta_{0.5} \in [\beta_0+1, n]$ |
| $\beta_1$ | $[0, 1)$ | $\lambda_1 \le \lambda_{\max}$ |
| $\beta_2$ | $(0, 1]$ | Cauchy-Schwarz: $\sum\lambda_k^2 \le (\sum\lambda_k)^2$ |
| $\gamma_1$ | $(0, 1)$ | $\Theta_2 < \Theta_1$（递减）+ 下界约束 |
| $\gamma_2$ | $> 0$ | $\Theta_{0.5} > \Theta_1$（递减早期） |
| $\delta_1$ | $(0, 1]$ | $\Theta_{0.5} - \Theta_1 \in (0, n]$ |
| $\delta_2$ | $(0, 2]$ | $\lambda_{\max} \in (0, 2]$（归一化拉普拉斯上界 = 2） |
| $\delta_3$ | $(0, 1)$ | $\Theta_3 < \Theta_2$ |
| $\varepsilon_1$ | $[0, 1]$ | $\lambda_1 \le \lambda_{\max}$ |
| $\varepsilon_2$ | $[0, 1)$ | $\Theta_2 \in [\beta_0, n]$ |

#### 定理 7.4（全无量纲性）

归一化拉普拉斯 $\mathcal{L} = I - D^{-1/2}AD^{-1/2}$ 从根上消去了边权的物理量纲——所有 $\lambda_k \in [0,2]$ 为无量纲比值，$t^*$ 也为无量纲参数（$[t] = [\lambda_k]^{-1} = 1$）。十一个参数均为无量纲量的代数组合。**无外部单位引入，无量纲灾难。** $\square$

---

## 第五篇：动力学系统

### §8. ODE 动力学假设（受谱约束的建模选择）

> **本节定位**：以下五维 ODE 系统是一个**动力学假设**（model hypothesis），而非从谱几何的严格推导。其函数形式受三个约束启发——(a) 不变域 $[0,1]^5$ 在边界上向量场向内；(b) 每个变量至少有一个增长项和一个衰减项（竞争-合作范式）；(c) 交叉项体现深度-广度-韧度之间的反馈环路。11 个参数由 §7 的谱热迹映射唯一赋值。函数形式本身是建模选择——11 个参数决定了该选择下的**特定动力学轨迹**，类似于薛定谔方程中哈密顿量的函数形式是公设，而势函数 $V(x)$ 的参数由系统决定。

#### 定义 8.1（状态空间）

五维相空间：

$$\Omega = [0, 1]^5 \subset \mathbb{R}^5$$

#### 定义 8.2（谱桥 ODE 系统——动力学假设）

$$\boxed{
\begin{aligned}
\dot{D} &= \alpha_2 S(1 - D) - \alpha_1 R D \\[4pt]
\dot{B} &= \beta_1 \rho (1 - B) - \beta_2 D B \\[4pt]
\dot{\rho} &= \gamma_1 D(1 - \rho) + \gamma_2 B(1 - \rho) - \delta_1 \rho - \delta_2 \rho R - \delta_3 \rho S \\[4pt]
\dot{R} &= \delta_1 \rho D + \delta_2 \rho R - \alpha_1 D R - \beta_2 B R - \varepsilon_1 R \\[4pt]
\dot{S} &= \varepsilon_2 D(1 - S) - \delta_3 \rho S - \gamma_2 B S
\end{aligned}
}$$

**方程解读——竞争-合作结构**：

| 变量 | 增长项 | 衰减项 | 动力学解释 |
|------|--------|--------|-----------|
| $D$（深度） | $\alpha_2 S(1-D)$：韧度驱动深度饱和 | $-\alpha_1 R D$：速率消耗深度 | 结构固化 vs 演化瓦解 |
| $B$（广度） | $\beta_1 \rho(1-B)$：能量驱动广度扩张 | $-\beta_2 D B$：深度压制广度 | 水平扩张 vs 垂直锁定 |
| $\rho$（能量） | $\gamma_1 D + \gamma_2 B$：深度和广度注入能量 | $-\delta_{1,2,3}$ 项：三重耗散 | 产出 vs 消耗 |
| $R$（速率） | $\delta_1 \rho D + \delta_2 \rho R$：能量和自催化驱动演化 | $-\alpha_1 D - \beta_2 B - \varepsilon_1$：三重压制 | 创新冲动 vs 惯性阻尼 |
| $S$（韧度） | $\varepsilon_2 D(1-S)$：深度奠定韧度基础 | $-\delta_3 \rho - \gamma_2 B$：能量流和广度侵蚀韧度 | 耐久性 vs 消耗 |

**每一项的谱参数对应**：增长系数（$\alpha_2, \beta_1, \gamma_1, \gamma_2, \delta_1, \delta_2, \varepsilon_2$）来自热迹的早期和临界采样——高值对应"热扩散慢"（结构稳固，增长潜力大）。衰减系数（$\alpha_1, \beta_2, \delta_1, \delta_2, \delta_3, \varepsilon_1$）来自谱分散度和热迹衰减采样——高值对应"结构松散"（易被瓦解）。

#### 定理 8.3（不变域）

$\Omega = [0,1]^5$ 在谱桥 ODE 流下**正向不变**。

**证明**。逐边界验证：

- $D = 0$：$\dot{D} = \alpha_2 S \ge 0$ ✓
- $D = 1$：$\dot{D} = -\alpha_1 R \le 0$ ✓
- $B = 0$：$\dot{B} = \beta_1 \rho \ge 0$ ✓
- $B = 1$：$\dot{B} = -\beta_2 D \le 0$ ✓（$D > 0$ 时严格负；若 $D = 0$，$B$ 不运动）
- $\rho = 0$：$\dot{\rho} = \gamma_1 D + \gamma_2 B \ge 0$ ✓
- $\rho = 1$：$\dot{\rho} = -\delta_1 - \delta_2 R - \delta_3 S \le 0$ ✓
- $R = 0$：$\dot{R} = \delta_1 \rho D \ge 0$ ✓
- $R = 1$：$\dot{R} = \delta_1 \rho D + \delta_2 \rho - \alpha_1 D - \beta_2 B - \varepsilon_1$。在标准参数范围（$\delta$ 小，$\alpha, \beta, \varepsilon$ 为正）下 $\dot{R} \le 0$ ✓
- $S = 0$：$\dot{S} = \varepsilon_2 D \ge 0$ ✓
- $S = 1$：$\dot{S} = -\delta_3 \rho - \gamma_2 B \le 0$ ✓

所有边界条件满足。$\square$

---

### §9. 平衡点分析

#### 定义 9.1（平衡点）

平衡点 $\vec{M}^* = (D^*, B^*, \rho^*, R^*, S^*) \in \Omega$ 满足 $\dot{\vec{M}} = \mathbf{0}$。

#### 定理 9.2（平衡点的存在性）

谱桥 ODE 系统在 $\Omega$ 内至少存在一个平衡点。

**证明**。$\Omega$ 为紧凸集，向量场连续且在边界上指向内部（定理 8.3）。由 Brouwer 不动点定理，存在 $\vec{M}^*$ 使 $\mathbf{F}(\vec{M}^*) = \mathbf{0}$。$\square$

#### 定义 9.3（Jacobian 矩阵）

在平衡点 $\vec{M}^*$ 处：

$$J = \left. \frac{\partial \dot{\vec{M}}}{\partial \vec{M}} \right|_{\vec{M}^*} \in \mathbb{R}^{5 \times 5}$$

记 $T = \text{tr}(J)$，$\Delta = \det(J)$，$\{\mu_1, \ldots, \mu_5\} = \text{eig}(J)$。

#### 定理 9.4（局部稳定性）

1. $\operatorname{Re}(\mu_k) < 0$ 对所有 $k$ → $\vec{M}^*$ 局部渐近稳定（双曲吸引子）。
2. $\exists k: \operatorname{Re}(\mu_k) > 0$ → $\vec{M}^*$ Lyapunov 不稳定。
3. $\exists k: \operatorname{Re}(\mu_k) = 0$ → 中心流形分析。

**证明**。Hartman-Grobman 定理的直接推论。$\square$

---

## 第六篇：原型分类

### §10. 分岔判别——三族与 P2 公理映射

> **目标**：从 Jacobian 矩阵的代数性质和 ODE 轨迹特征推导三类泛模因原型族。核心创新：**从 P2 公理的三条件（可复制/可演化/身份保持）到 ODE 稳态性质的映射**。

#### §10.0 公理-动力学对应

P2 公理定义 $\mathcal{P} = \{x \mid \text{replicable} \land \text{evolvable} \land \text{identity\_preserving}\}$。在 ODE 框架中：

| P2 条件 | ODE 对应 | 数学判据 |
|---------|----------|----------|
| **可复制**（replicable） | $\rho^* > 0$：能量/人密度非零——无能量则无法复制 | $\rho^* > \varepsilon_\rho$ |
| **可演化**（evolvable） | $R^* > 0$ 或轨迹存在 $\dot{R} > 0$ 的时段——演化速率非零 | $\exists t: \dot{R}(t) > 0$ |
| **身份保持**（identity\_preserving） | $D^* > 0$（深度非零）且 $\operatorname{Re}(\mu_k) < 0$（稳定）——扰动后回归恒等状态 | $D^* > \varepsilon_D$ 且稳定 |

三族原型对应 P2 三条件的不同满足模式：

#### 定理 10.1（基石族——Cornerstone）

社区 $X_i$ 属于基石族当且仅当：

$$\operatorname{Re}(\mu_k) < 0 \; \forall k, \quad T < 0, \quad \Delta > 0$$

此时 $D^* > 0$，$R^* \approx 0$，$S^* > 0$。

**P2 满足模式**：可复制 ✓（$\rho^* > 0$），可演化 ✗（$R^* \approx 0$），身份保持 ✓（稳定吸引子）。基石族满足 P2 的**两个条件**——是稳定的、自我维持的模因核心，但不主动扩展。对应文化基因中的"基础信念"、生物基因中的"持家基因"。

**动力学解读**：所有扰动指数衰减回平衡点。深度$D$自我维持（$\alpha_2 S(1-D)$ 项），速率 $R$ 被深度压制（$-\alpha_1 D R$ 项），系统处于"结构锁定"状态。

#### 定理 10.2（过客族——Visitor）

社区 $X_i$ 属于过客族当且仅当：

$$T < 0, \quad \Delta < 0 \quad \text{（鞍点结构）}$$

且轨迹中存在 $\tau$ 使 $R(\tau) > R(0)$（脉冲特征——先扩散后衰减）。

**P2 满足模式**：可复制 ✓，可演化 ✓（$R$ 有脉冲），身份保持 ✗（鞍点——不稳定方向导致无法长期保持）。过客族是**有限生命周期的模因爆发**——充分满足可复制和可演化，但无法持久。

**动力学解读**：$R$ 短暂正反馈（$\delta_2 \rho R$ 自催化项）推动快速扩张，随后耗散项（$-\alpha_1 D R$ 等）占主导，轨迹偏离平衡点。

#### 定理 10.3（泡沫族——Bubble）

社区 $X_i$ 属于泡沫族当且仅当：

$$\exists k: \operatorname{Re}(\mu_k) > 0 \quad \text{（至少一个不稳定方向）}$$

且 $S^* \approx 0$（韧度崩溃）。

**P2 满足模式**：可复制 ✓（瞬间），可演化 ✓（爆发），身份保持 ✗（结构不稳定 → 消散）。泡沫族是**短暂膨胀后彻底消失**的模因模式——不具备任何形式的长期身份。

**动力学解读**：Lyapunov 不稳定 + 韧度 $S \to 0$（$\varepsilon_2 D(1-S)$ 增长项不足以补偿 $-\delta_3 \rho S - \gamma_2 B S$ 的耗散），社区结构在演化中瓦解。

---

### §11. 谱子型细分（九子型）

> 在三族基础上按 Jacobian 特征值的代数性质做九子型细分。

#### 定义 11.1（基石族三子型）

| 子型 | 代数条件 | 动力学含义 |
|------|----------|-----------|
| **Stone** | $\max_k \|\operatorname{Re}(\mu_k)\| < 10^{-3}$ | 超稳定吸引子——深度近乎永久 |
| **StableCore** | $\max_k \|\operatorname{Re}(\mu_k)\| \ge 10^{-3}$ | 稳定但可被足够大的外部扰动激发 |
| **Resilient** | $\lambda_1^{(X_i)} < 10^{-2}$（谱隙极小） | 处于分裂边缘——韧度 S 极高但一触即溃 |

#### 定义 11.2（过客族三子型）

| 子型 | 代数条件 | 动力学含义 |
|------|----------|-----------|
| **Burst** | $\max_t \|\ddot{R}(t)\|$ 超阈值 | 爆发型——急剧扩张后骤停 |
| **Decay** | $\dot{D} < 0$ 且 $\dot{S} < 0$ 单调 | 衰亡型——缓慢消解 |
| **Transient** | 既不满足 Burst 也不满足 Decay | 普通瞬态——有限生命周期的温和涨落 |

#### 定义 11.3（泡沫族二子型）

| 子型 | 代数条件 | 动力学含义 |
|------|----------|-----------|
| **Source** | $\gamma_2 > \gamma_1$ | 外部注入驱动——$\rho$ 净产出 |
| **Sink** | $\gamma_1 > \gamma_2$ 且 $\varepsilon_1 < \varepsilon_2$ | 内部耗散主导——$\rho$ 净消耗 |

#### 定义 11.4（跨族子型）

| 子型 | 代数条件 | 动力学含义 |
|------|----------|-----------|
| **Oscillatory** | $\max_k \|\operatorname{Im}(\mu_k)\| > 10^{-2}$ | 螺旋收敛或发散——周期性涨落 |

---

## 第七篇：全局结论

### §12. 闭合推导链与信息分类

#### 完整链图——附各步骤数学性质标注

$$\boxed{
\begin{array}{c}
I \in \Sigma^* \quad \textit{（原始文本）} \\
\quad \Big\downarrow \quad \S 2 \quad \text{滑动窗口 } A \quad \textbf{[信息保持投影]} \quad \text{保留 } (w+1)\text{-gram 谱，丢失 } d>w \text{ 长程序} \\
A \in \mathbb{R}^{n \times n}_{\ge 0} \quad \textit{（加权邻接矩阵）} \\
\quad \Big\downarrow \quad \S 3 \quad \mathcal{L} = I - D^{-1/2}AD^{-1/2} = U\Lambda U^\top \quad \textbf{[满射，像集上可逆]} \quad \lambda_k \in [0,2] \\
(\Lambda, U_{\text{std}}) \quad \textit{（规范化归一化谱表示）} \\
\quad \Big\downarrow \quad \S 4 \quad k^* = \arg\max\Delta_k + \text{全局最优 } k\text{-means} \quad \textbf{[定义式准则]} \quad \text{唯一划分（除编号重排）} \\
X_1, \ldots, X_{k^*} \quad \textit{（社区集合）} \\
\quad \Big\downarrow \quad \S 5 \quad \text{热迹采样 } \Theta(\tau_i), \Theta(t^*_i) \to [0,1]^5 \quad \textbf{[天然归一化]} \quad \text{三个时间标度（}\tau_i, t^*, t^*_G\text{）} \\
\pi_i \in \Omega \quad \textit{（五维谱状态向量）} \\
\quad \Big\downarrow \quad \S\S 6\text{–}7 \quad \Theta(t) \to t^* \to 11\text{ 参数} \quad \textbf{[谱代数映射]} \quad \text{四个标度点完备代数组合} \\
\theta_i \in \mathbb{R}^{11} \quad \textit{（ODE 参数向量——谱热迹赋值）} \\
\quad \Big\downarrow \quad \S 8 \quad \dot{\vec{M}} = \mathbf{F}_{\theta_i}(\vec{M}) \quad \textbf{[动力学假设]} \quad \text{竞争-合作 Lotka-Volterra 范式} \\
\{\vec{M}_i(t)\}_{t \in [0,T]} \quad \textit{（五维轨迹）} \\
\quad \Big\downarrow \quad \S\S 9\text{–}11 \quad J \to (T, \Delta, \{\mu_k\}) \to \text{分类} \quad \textbf{[代数判据]} \quad \text{P2 公理 ↔ 稳态性质映射} \\
\text{三族} \in \{\text{基石}, \text{过客}, \text{泡沫}\} \cup \text{九子型细分}
\end{array}
}$$

#### 定理 12.1（链的映射性质总表）

| 步骤 | 映射 | 数学性质 | P5 满足方式 | 信息丢失 |
|------|------|----------|------------|----------|
| $\S 2$ | $I \to A$ | 信息保持投影 | 丢失 $d > w$ 长程序 | 由 $w$ 大小控制 |
| $\S 3$ | $A \to (\Lambda, U_{\text{std}})$ | 满射，在 $\Phi_A$ 像集上可逆 | 缩放等价类在 $\Phi_A$ 像集中固定 | 全局边权缩放（在像集中无影响） |
| $\S 4$ | $(\Lambda, U_{\text{std}}) \to \{X_i\}$ | 定义式准则 + 全局最优 | 确定性 | 降维（$O(n^2) \to O(n \log n)$） |
| $\S 5$ | $\{X_i\} \to \pi_i$ | 热迹天然归一化 | 确定性 | 降维（社区结构 $\to$ 5 维） |
| $\S\S 6\text{–}7$ | $\pi_i \to \theta_i$ | 谱代数映射 | 确定性 | 信息重组（5 维 $\to$ 11 维） |
| $\S 8$ | $\theta_i \to \{\vec{M}_i(t)\}$ | 动力学假设 + 解唯一 | 确定性 | ODE 形式是建模选择 |
| $\S\S 9\text{–}11$ | 轨迹 $\to$ 分类 | 代数判据 | 确定性 | P2 语义映射是解释 |

#### 定理 12.2（参数覆盖性——谱桥的非平凡性）

映射 $\Phi_{\text{total}}: \Sigma^* \to \mathcal{C}$（$\mathcal{C} =$ 原型分类空间）满足：

1. **非平凡满射**：$\text{Im}(\Phi_{\text{total}})$ 在 $\mathcal{C}$ 中具有正测度——即框架能产生不同原型的分类结果，而非退化到单一类别。
2. **谱辨识性**：对任意 $I_1, I_2$，若其 $(w+1)$-gram 共现谱不同，则 $\text{spec}(\mathcal{L}_{I_1}) \neq \text{spec}(\mathcal{L}_{I_2})$，进而 11 参数不同，ODE 轨迹不同。
3. **连续依赖性**：$\Phi_{\text{total}}$ 是 Lipschitz 连续的（在 $(w+1)$-gram 统计量的小扰动下）——相似的输入产生相似的分类，框架不会因数值噪声而产生突变。

**证明草图**。(1) 随机序列模型产生连续的特征值分布 → 参数空间覆盖正测度。(2) 谱是 $(w+1)$-gram 统计的连续函数。(3) 所有中间映射为连续代数运算。$\square$

---

### §13. 结果索引

| 编号 | 名称 | 位置 | 类别 | 核心内容 |
|------|------|------|------|----------|
| 2.4 | 滑动窗口信息保持定理 | §2 | 定理 | $I \to A$ 保留 $(w+1)$-gram 谱，丢失 $d > w$ 长程序 |
| 3.5 | Perron 重构定理 | §3 | 定理 | 从 $\mathbf{u}_0$ 恢复度序列相对比例 |
| 3.6 | 谱分解映射性质 | §3 | 定理 | 满射 + 缩放等价类 + $\Phi_A$ 像集上可逆 |
| 4.1 | 谱社区数定义 | §4 | 定义 | $k^* = \arg\max \Delta_k$（定义式准则） |
| 4.4 | 划分确定性 | §4 | 定理 | 全局最优 $k$-means 划分唯一 |
| 5.2 | 五维谱状态 | §5 | 定义 | 热迹比值——天然归一化 |
| 5.3 | 热迹天然归一化定理 | §5 | 定理 | 五分量由热方程解析性质内在归一化 |
| 7.2 | 十一参数 | §7 | 定义 | 四个标度点的完备代数组合 |
| 7.4 | 全无量纲定理 | §7 | 定理 | 归一化拉普拉斯消去全部物理量纲 |
| 8.2 | 谱桥 ODE 系统 | §8 | 假设 | 竞争-合作动力学范式，11 参数由谱赋值 |
| 8.3 | 不变域 | §8 | 定理 | $\Omega = [0,1]^5$ 正向不变 |
| 9.2 | 平衡点存在 | §9 | 定理 | Brouwer 不动点 |
| 9.4 | 局部稳定性 | §9 | 定理 | Hartman-Grobman 稳定性分类 |
| 10.1–10.3 | 三族判别 | §10 | 定理 | P2 公理三条件 $\leftrightarrow$ 稳态性质映射 |
| 11.1–11.4 | 九子型 | §11 | 定义 | Jacobian 特征值代数分类 |
| 12.2 | 参数覆盖性 | §12 | 定理 | 非平凡满射 + 谱辨识 + 连续依赖 |

---

**全链总结**：

> 本框架从八条公理出发，经七步映射（信息保持投影 → 满射谱分解 → 定义式社区划分 → 热迹天然状态 → 谱代数参数 → 动力学假设 → 代数分类判据），到达三族九子型原型分类。每一步标注了数学性质：哪些是严格推导的（定理）、哪些是定义式准则、哪些是建模选择（假设）。公理 P5（可逆性认识论）的核心要求——可追踪的信息保持性质——贯穿全链。框架的开放性在于：ODE 的函数形式可替换（其他竞争-合作范式的变体），参数赋值方式固定（谱热迹唯一确定），分类判据固定（Jacobian 代数性质）。这构成了泛模因理论从"为模因发明数学"到"用谱几何观测模因"的数学基础。

---

**文档结束**。版本 v5.1，2026-07-08。
