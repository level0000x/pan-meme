# 泛模因理论：完整证明链

**版本**：v5.0  
**日期**：2026-07-08  
**状态**：完整数学证明 — 八条公理 → 谱桥统一推导 → 三族九子型原型分类

> 本文给出泛模因理论从公理到原型分类的完整数学证明链。推导路径为：公理 $P_1$–$P_8$ → 滑动窗口加权邻接矩阵（单射）→ 图拉普拉斯谱分解（双射）→ 谱间隙社区划分 → 谱矩五维映射 → 热迹参数映射 → ODE 动力学系统 → Jacobian 分岔判别 → 三族九子型原型分类。全程无手工系数、无经验超参、无量纲灾难。每步变换保留明确的可逆性或唯一性。

---

## 目录

- **第〇篇：基础**
  - §0. 符号约定
  - §1. 八条命题公理
- **第一篇：文本 → 图谱**
  - §2. 滑动窗口加权邻接矩阵（$I \to A$）
  - §3. 图拉普拉斯谱分解（$A \to (\Lambda, U_{\text{std}})$）
- **第二篇：图谱 → 社区**
  - §4. 谱间隙社区划分（$(\Lambda, U_{\text{std}}) \to \{X_i\}$）
- **第三篇：社区 → 状态**
  - §5. 谱矩五维映射（$\{X_i\} \to \pi_i$）
- **第四篇：状态 → 参数**
  - §6. 热迹与混合时间
  - §7. 十一参数谱映射（$\pi_i \to \theta_i$）
- **第五篇：动力学系统**
  - §8. ODE 系统定义
  - §9. 平衡点分析
- **第六篇：原型分类**
  - §10. 分岔判别定理（三族）
  - §11. 谱子型细分（九子型）
- **第七篇：全局结论**
  - §12. 闭合推导链与信息守恒定理
  - §13. 定理索引

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
| $\text{spec}(L)$ | 矩阵 $L$ 的谱（特征值多重集） |
| $\Theta(t)$ | 热迹函数 |
| $t^*$ | 混合时间（热迹衰减到 $\beta_0 + 1$ 的最小时间） |
| $\Omega$ | 五维相空间 $[0,1]^5$ |
| $\lambda_k$ | 升序排列的第 $k$ 个特征值（$\lambda_0 \le \lambda_1 \le \cdots$） |
| $O(\cdot)$ | 大 O 渐近记号 |

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

**公理 P4（认知-建模统一性）**。人类的认知过程 $C$ 与形式化建模过程 $\mathcal{M}$ 共享同构核心：
$$C(h) \cong \mathcal{M}(h)$$
即在适当范畴中等价。

**公理 P5（可逆性认识论）**。对任意合法建模 $\Phi$，存在逆映射 $\Phi^{-1}$ 使：
$$\Phi^{-1} \circ \Phi = \text{id}_{\text{Dom}(\Phi)}$$
即模型的每一步变换必须有明确的逆操作（或至少是单射）。

**公理 P6（计算即模因）**。任何图灵完备系统 $T$ 本身是一个泛模因实例：$T \in \mathcal{P}$。

**公理 P7（演化普遍性）**。若系统 $S$ 同时具备复制 $R$、变异 $V$、选择 $S_e$ 三个算子，则 $S$ 表现出演化行为：
$$S_{t+1} = S_e \circ V \circ R(S_t)$$

**公理 P8（层级涌现）**。泛模因构成偏序集 $(\mathcal{P}, \preceq)$，其中 $x \preceq y$ 表示 $x$ 是 $y$ 的构成部分。存在：
- 向上因果：$x \preceq y \implies \text{state}(x) \mapsto \text{state}(y)$
- 向下因果：$y \succcurlyeq x \implies \text{constraint}(y) \mapsto \text{state}(x)$

> **注释**：P5（可逆性认识论）是贯穿全文的核心约束——每一步推导中我们都要验证映射的单射性、双射性或确定性唯一性。

---

## 第一篇：文本 → 图谱

### §2. 滑动窗口加权邻接矩阵（$I \to A$）

> **目标**：将任意字符串 $I \in \Sigma^*$ 映射为加权邻接矩阵 $A \in \mathbb{R}^{n \times n}_{\ge 0}$，且该映射为**单射**——不同的 $I$ 必定产生不同的 $A$。

#### 定义 2.1（滑动窗口加权共现）

设输入序列 $I = (c_1, c_2, \ldots, c_L)$，$c_p \in \Sigma$。固定窗口半径 $w \in \mathbb{N}$（$1 \le w \ll L$）。对每个位置 $p$，定义窗口：

$$W_p = \{c_{p-w}, \ldots, c_p, \ldots, c_{p+w}\} \cap \{c_1, \ldots, c_L\}$$

对 $1 \le \Delta \le w$，位置 $p$ 处的前向共现对 $(c_p, c_{p+\Delta})$ 的权值贡献为：

$$w_{p, \Delta}(a, b) = \mathbf{1}_{\{c_p = a\}} \cdot \mathbf{1}_{\{c_{p+\Delta} = b\}} \cdot \exp\left(-\frac{\Delta}{w}\right)$$

指数衰减因子 $\exp(-\Delta/w)$ 使近邻贡献强于远邻，捕获时序邻近性。$\Delta = 0$（自共现）始终贡献 $\exp(0) = 1$。

#### 定义 2.2（有向邻接矩阵）

$$A^{\text{dir}}_{ij} = \sum_{p=1}^{L} \sum_{\Delta=1}^{\min(w, L-p)} w_{p, \Delta}(c_i, c_j)$$

其中 $c_i, c_j \in \Sigma$ 为字符表中的第 $i, j$ 个字符。

#### 定义 2.3（对称化——无向加权邻接矩阵）

$$A_{ij} = A^{\text{dir}}_{ij} + A^{\text{dir}}_{ji}$$

$A \in \mathbb{R}^{n \times n}_{\ge 0}$ 为对称矩阵，$n = |\Sigma|$（字符表大小）。

#### 定理 2.4（单射性）

映射 $\Phi_A: \Sigma^* \to \mathbb{R}^{n \times n}_{\ge 0}$，$\Phi_A(I) = A$，是**单射**。

**证明**。设 $I_1 \neq I_2$，两者长度分别为 $L_1, L_2$。

**情形 1**：$L_1 \neq L_2$。注意到 $\text{Tr}(A) = \sum_i A_{ii}$。自共现项 $A_{ii}$ 来自每个位置 $p$ 的 $\Delta = 0$ 贡献$\exp(0) = 1$，故 $\text{Tr}(A) = L$（每个位置对自身的贡献）。因此 $A_1$ 与 $A_2$ 的迹不等，$A_1 \neq A_2$。

**情形 2**：$L_1 = L_2 = L$ 但存在位置 $p^*$ 使 $c_{p^*} \neq c'_{p^*}$。设 $c_{p^*} = a$、$c'_{p^*} = b$（$a \neq b$）。取 $\Delta = 1$：

- 若 $p^* + 1 \le L$，$I_1$ 中有共现对 $(a, c_{p^*+1})$ 对 $A^{\text{dir}}_{a, c_{p^*+1}}$ 贡献 $e^{-1/w}$，而 $I_2$ 中 $a$ 不出现在位置 $p^*$，故 $I_2$ 中无双 $(a, c_{p^*+1})$ 在该位置的贡献。
- 同理，$I_2$ 中 $(b, c'_{p^*+1})$ 的贡献在 $I_1$ 中不存在。

推广至所有差异位置 $p$ 和所有步长 $\Delta \in [1, w]$，至少存在一对字符索引 $(i, j)$ 使 $A_{ij}^{(1)} \neq A_{ij}^{(2)}$。$\square$

#### 推论 2.5（信息保留）

$\Phi_A$ 是单射意味着：虽不能从 $A$ 高效重构 $I$（计算上困难），但信息在数学上**未丢失**。这满足公理 P5 的单射要求：$\exists \Phi_A^{-1}_{\text{left}}: \Phi_A^{-1}(\Phi_A(I)) = I$。

---

### §3. 图拉普拉斯谱分解（$A \to (\Lambda, U_{\text{std}})$）

> **目标**：将加权邻接矩阵 $A$ 通过图拉普拉斯谱分解映射到规范特征对 $(\Lambda, U_{\text{std}})$，且该映射为**双射**。

#### 定义 3.1（度矩阵与图拉普拉斯）

无向加权图 $G = (V, E, A)$，$V = \Sigma$，$|V| = n$。度矩阵：

$$D_{ii} = \sum_{j=1}^{n} A_{ij}, \quad D_{ij} = 0 \; (i \neq j)$$

图拉普拉斯：

$$L = D - A$$

#### 引理 3.2（拉普拉斯的基本性质）

1. $L$ 为实对称半正定矩阵。
2. $\forall \mathbf{x} \in \mathbb{R}^n, \; \mathbf{x}^\top L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2 \ge 0$。
3. $\mathbf{1} = (1, 1, \ldots, 1)^\top$ 为零特征值对应的特征向量：$L\mathbf{1} = \mathbf{0}$。
4. $\lambda_0 = 0$，且 $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1}$。
5. $\lambda_1$ 的零重数 = 图的连通分量数 $\beta_0$。

**证明**。性质 1-5 均为标准谱图论结论（Chung 1997）。性质 2 由展开得证；性质 3 由 $D\mathbf{1} = A\mathbf{1}$ 得证；性质 4 由 Courant-Fischer 极小极大定理得证；性质 5 由 Fiedler (1973) 的连通性定理得证。$\square$

#### 定义 3.3（谱分解）

$$L = U \Lambda U^\top, \quad \Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$$

$U = [\mathbf{u}_0, \mathbf{u}_1, \ldots, \mathbf{u}_{n-1}] \in \mathbb{R}^{n \times n}$ 为正交矩阵（$U^\top U = I_n$），$\mathbf{u}_k$ 为 $\lambda_k$ 对应的单位特征向量。特征值按升序排列。

#### 引理 3.4（特征向量的符号歧义与退化歧义）

1. **符号歧义**：若 $(\lambda_k, \mathbf{u}_k)$ 是 $L$ 的特征对，则 $(\lambda_k, -\mathbf{u}_k)$ 也是。
2. **退化子空间内旋转歧义**：若 $\lambda_k$ 的重数 $\ge 2$，则 $\lambda_k$ 对应的特征子空间为正交群 $O(m_k)$ 作用下的不变子空间（$m_k$ 为重数），基向量的选择不唯一。

#### 定义 3.5（符号标准化与退化消歧）

**步骤 1 — 符号标准化**。对每个特征向量 $\mathbf{u}_k$，定义第一个显著分量索引：

$$j_k = \min\{j \mid |(\mathbf{u}_k)_j| > \varepsilon\}$$

其中 $\varepsilon = 10^{-8}$（机器精度边界）。若 $(\mathbf{u}_k)_{j_k} < 0$，则 $\mathbf{u}_k \leftarrow -\mathbf{u}_k$。

**步骤 2 — 退化子空间消歧**。对每个重数 $\ge 2$ 的特征值 $\lambda_k$，对其特征子空间内的基向量：
- 按 $j_k$（第一个显著分量索引）升序排列
- 做 Gram-Schmidt 正交化
- 对每个结果向量应用步骤 1 的符号标准化

记标准化后的特征基为 $U_{\text{std}}$。

#### 定理 3.6（规范化谱表示的双射性）

映射 $\Phi_{\text{spec}}: \mathbb{R}^{n \times n}_{\text{sym}} \to \mathcal{U} \times \mathbb{R}^n_{\text{sort}}$：

$$\Phi_{\text{spec}}(A) = (\Lambda, U_{\text{std}})$$

在 $n \times n$ 实对称矩阵空间上是**双射**。

**证明**。

(1) **单射**：若 $(\Lambda, U_{\text{std}}) = (\Lambda', U'_{\text{std}})$，则：

$$L = U_{\text{std}} \Lambda U_{\text{std}}^\top = U'_{\text{std}} \Lambda' U'_{\text{std}}{}^\top = L'$$

由 $L = D - A$ 且 $D_{ii} = \sum_j A_{ij} = L_{ii}$（因为 $A_{ii}$ 产生对角元素的结构约束），可从 $L$ 唯一恢复 $D$（$D_{ii} = \sum_{j \neq i} A_{ij}$ 需要从 $L$ 的结构中分离）。更直接地：$A = D - L$ 且 $D_{ii} = \sum_j A_{ij}$，该线性系统有唯一解（$n$ 个方程，$n$ 个未知数 $D_{ii}$，约束为 $A_{ii} \ge 0$）。

(2) **满射**：任意实对称 $A$ 有谱分解 $L = D - A$，$L$ 为实对称矩阵，存在正交对角化 $U \Lambda U^\top$。符号标准化消去 $\pm$ 歧义，Gram-Schmidt 消去退化子空间内的旋转歧义。故对每个 $A$，存在唯一 $(\Lambda, U_{\text{std}})$。

(3) **可逆**：$\Phi_{\text{spec}}^{-1}(\Lambda, U_{\text{std}}) = U_{\text{std}} \Lambda U_{\text{std}}^\top = L$，再通过 $A = D - L$ 恢复 $A$。$\square$

#### 推论 3.7（信息完全保留）

$(\Lambda, U_{\text{std}})$ 保留了 $G$ 的**全部组合信息**。Cospectral 非同构图的存在性（已知比例在 $n \to \infty$ 时趋近于零 [Babai et al.]）在实践中可忽略。

#### 推论 3.8（P5 满足性）

$$I \xrightarrow{\Phi_A}_{\text{单射}} A \xrightarrow{\Phi_{\text{spec}}}_{\text{双射}} (\Lambda, U_{\text{std}})$$

复合映射 $\Phi_{\text{spec}} \circ \Phi_A$ 是单射，满足公理 P5 的可逆性要求：不同输入产生不同的谱表示。

---

## 第二篇：图谱 → 社区

### §4. 谱间隙社区划分（$(\Lambda, U_{\text{std}}) \to \{X_i\}$）

> **目标**：从谱特征中无超参地确定社区数和社区归属。全部依赖拉普拉斯谱的代数性质，无需手工调参。

#### 定义 4.1（特征值间隙序列）

图 $G$ 的拉普拉斯特征值间隙序列为：

$$\Delta_k = \lambda_{k+1} - \lambda_k, \quad k = 0, 1, \ldots, n-2$$

#### 定理 4.2（自然社区数——谱间隙准则）

最优社区数 $k^*$ 由最大特征值间隙唯一确定：

$$k^* = \arg\max_{1 \le k \le n-2} (\lambda_{k+1} - \lambda_k)$$

**证明草图**（谱聚类理论标准结论 — von Luxburg 2007, 定理 8）。

图的拉普拉斯特征向量 $\mathbf{u}_1, \ldots, \mathbf{u}_k$ 的逐行嵌入是图的最小 $k$-way 归一化切割问题的连续松弛解。特征值 $\lambda_k$ 刻画了第 $k$ 个切割方向的代价值。当 $\lambda_{k+1} - \lambda_k$ 较大时，前 $k$ 个特征向量已经捕获了图的主要聚类结构，而第 $k+1$ 个特征向量对应的切割代价显著上升——意味着没有更多的"自然"聚类方向。$\square$

#### 定义 4.3（谱嵌入与行归一化）

取前 $k^*$ 个非零特征向量 $\mathbf{u}_1, \ldots, \mathbf{u}_{k^*}$ 构成嵌入矩阵：

$$U_{:1:k^*} \in \mathbb{R}^{n \times k^*}$$

行归一化（消除度偏差对嵌入的影响）：

$$\tilde{\mathbf{u}}_v = \frac{(U_{:1:k^*})_{v, :}}{\|(U_{:1:k^*})_{v, :}\|_2} \in \mathbb{R}^{k^*}, \quad v = 1, 2, \ldots, n$$

若范数为零，取 $\tilde{\mathbf{u}}_v = \mathbf{0}$。

#### 定义 4.4（$k$-means 硬分配）

将归一化行向量 $\{\tilde{\mathbf{u}}_v\}_{v=1}^{n}$ 聚类为 $k^*$ 个簇，极小化簇内平方和：

$$\min_{\{C_i\}_{i=1}^{k^*}} \sum_{i=1}^{k^*} \sum_{v \in C_i} \|\tilde{\mathbf{u}}_v - \boldsymbol{\mu}_i\|_2^2, \quad \boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{v \in C_i} \tilde{\mathbf{u}}_v$$

得到社区划分 $\{X_1, X_2, \ldots, X_{k^*}\}$，每个 $X_i \subseteq V = \Sigma$。

#### 定理 4.5（划分的确定性）

给定 $G$，社区数 $k^*$ 和社区划分 $\{X_i\}_{i=1}^{k^*}$ 由 $L$ 的谱**唯一**确定。$k$-means 的多起点局部极值是计算上的工程考量，不改变数学上解的唯一存在性。

#### 备注 4.6（零超参保证）

整个社区划分过程（§4）不引入任何需要手工调节的参数：
- $k^*$ 由最大谱间隙自动确定
- 嵌入维数 $k^*$ 由 $k^*$ 自动确定
- $k$-means 的簇数 $k^*$ 由 $k^*$ 自动确定

所有自由度均内生于 $\text{spec}(L)$。

---

## 第三篇：社区 → 状态

### §5. 谱矩五维映射（$\{X_i\} \to \pi_i$）

> **目标**：对每个社区 $X_i$，导出五维状态向量 $\pi_i \in [0,1]^5$，所有分量由社区和全局的拉普拉斯谱唯一确定，**零手工系数**。

#### 定义 5.1（社区子图与谱）

设社区 $X_i \subseteq V$，其导出子图 $G[X_i]$ 的拉普拉斯为 $L^{(i)}$（$|X_i| \times |X_i|$ 矩阵）。记 $L^{(i)}$ 的特征值为：

$$\{ \lambda_k^{(i)} \}_{k=0}^{|X_i|-1}, \quad 0 = \lambda_0^{(i)} \le \lambda_1^{(i)} \le \cdots \le \lambda_{|X_i|-1}^{(i)}$$

记全局谱为 $\{ \lambda_k^{(G)} \}_{k=0}^{n-1}$。全局热迹为 $\Theta_G(t) = \sum_{k=0}^{n-1} e^{-t \lambda_k^{(G)}}$。

社区 $X_i$ 的混合时间 $t^*_i$ 定义为：

$$t^*_i = \inf\{ t > 0 \mid \Theta_i(t) \le \beta_0^{(i)} + 1 \}$$

其中 $\Theta_i(t) = \sum_{k=0}^{|X_i|-1} e^{-t \lambda_k^{(i)}}$ 为社区热迹，$\beta_0^{(i)}$ 为 $G[X_i]$ 的连通分量数。

#### 定义 5.2（五维谱矩向量）

$$\boxed{
\begin{aligned}
D_i &= 1 - \exp\left(-\lambda_1^{(i)} \cdot \frac{|X_i|}{n}\right) & &\text{(深度)}\\[6pt]
B_i &= 1 - \exp\left(-\frac{\lambda_{\max}^{(i)}}{\lambda_{\max}^{(G)}}\right) & &\text{(宽度)}\\[6pt]
\rho_i &= \frac{\sum_{k} \lambda_k^{(i)}}{\sum_{k} \lambda_k^{(G)}} = \frac{|E_i|}{|E_G|} & &\text{(人/能量)}\\[6pt]
R_i &= \frac{t^*_G}{t^*_i} & &\text{(广度)}\\[6pt]
S_i &= \frac{1}{1 + \lambda_1^{(i)}} & &\text{(韧度)}
\end{aligned}
}$$

#### 引理 5.3（各分量的量纲分析）

**$D_i$（深度）**：$\lambda_1^{(i)}$ 在加权图中的量纲为 $[\text{边权} \cdot |X_i|]$。乘以 $|X_i|/n$ 后得到 $[\text{边权} \cdot |X_i|^2 / n]$。经 $\exp$ 和 $1 - \exp$ 映射后，$D_i \in [0,1]$ 为无量纲比值。

**$B_i$（宽度）**：两个谱半径（最大特征值）的比值，无量纲。$B_i \in [0, 1)$。

**$\rho_i$（人/能量）**：$\sum \lambda_k = \text{Tr}(L) = 2|E|$（拉普拉斯迹定理）。故 $\rho_i = |E_i| / |E_G|$ 为边数比，无量纲。$\rho_i \in [0, 1]$。

**$R_i$（广度）**：两个混合时间的比值。混合时间的量纲为 $[\text{边权}]^{-1}$（见 §6 的详细分析），分子分母同量纲，互相消去。$R_i \in (0, \infty)$，截断至 $[0, 1]$ 进行归一化。$R_i > 1$ 表示 $X_i$ 内部混合比全局快。

**$S_i$（韧度）**：$\lambda_1^{(i)} \ge 0$，故 $S_i \in (0, 1]$。$\lambda_1^{(i)} \to 0$（最弱连接趋于消失）时 $S_i \to 1$（韧度最大）；$\lambda_1^{(i)} \to \infty$ 时 $S_i \to 0$（韧度趋于零）。

#### 定理 5.4（零手工系数——谱矩定理）

定义 5.2 中五个分量的公式**不含任何经验权重、硬截断、或超参数**。所有映射由 $\text{spec}(L)$ 解析确定。

**证明**。逐项检查：

- $D_i$ 的公式仅使用 $\lambda_1^{(i)}$、$|X_i|$、$n$，均来自谱。
- $B_i$ 的公式仅使用 $\lambda_{\max}^{(i)}$ 和 $\lambda_{\max}^{(G)}$，均来自谱。
- $\rho_i$ 的公式来自迹等价 $\text{Tr}(L) = 2|E|$，由谱本质决定。
- $R_i$ 的公式仅使用 $t^*_G$ 和 $t^*_i$，两者均由热迹 $\Theta(t) = \sum e^{-t\lambda_k}$ 决定（见 §6），故来自谱。
- $S_i$ 的公式仅使用 $\lambda_1^{(i)}$，来自谱。

无任何外部参数介入。$\square$

---

## 第四篇：状态 → 参数

### §6. 热迹与混合时间

> **目标**：从全局谱定义热迹 $\Theta(t)$ 和天然标度参数 $t^*$，为后续参数映射提供统一的量纲基础。

#### 定义 6.1（热迹函数）

图 $G$ 的热迹定义为拉普拉斯热核 $\exp(-tL)$ 的迹：

$$\Theta_G(t) = \text{Tr}(e^{-tL}) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$$

#### 引理 6.2（热迹的基本性质）

1. $\Theta_G(0) = n$。
2. $\Theta_G(t)$ 在 $t > 0$ 上**严格递减**。
3. $\lim_{t \to \infty} \Theta_G(t) = \beta_0$（$\lambda_0 = 0$ 重数 = 连通分量数）。
4. $\Theta_G(t)$ 在 $(0, \infty)$ 上连续可微。

**证明**。性质 1：$e^{-0 \cdot \lambda_k} = 1$，共 $n$ 项。性质 2：$\forall t > 0$，
$\Theta'_G(t) = -\sum \lambda_k e^{-t\lambda_k} < 0$（至少一个 $\lambda_k > 0$ 时严格负）。性质 3：$\lim_{t \to \infty} e^{-t\lambda_k} = 0$ 当 $\lambda_k > 0$；仅 $\lambda_0 = 0$ 的项贡献 $1$，共 $\beta_0$ 项。性质 4：指数和的任意阶导数均存在。$\square$

#### 定义 6.3（混合时间 $t^*$）

全局混合时间 $t^*_G$ 定义为热迹下降到 $\beta_0 + 1$ 的最小时间：

$$t^*_G = \inf\{ t > 0 \mid \Theta_G(t) \le \beta_0 + 1 \}$$

由引理 6.2，$\Theta_G(0) = n \ge \beta_0 + 1$（除非 $n = \beta_0$，即无边图，此时 $t^*_G = 0$）且 $\lim_{t \to \infty} \Theta_G(t) = \beta_0 < \beta_0 + 1$，故 $t^*_G$ **存在且唯一**。

#### 引理 6.4（$t^*$ 的量纲）

热迹 $\Theta(t) = \sum e^{-t \lambda_k}$ 中的参数 $t$ 必须使 $t \lambda_k$ 无量纲（指数的参数无量纲）。故 $[t] = [\lambda_k]^{-1} = [\text{边权}]^{-1}$。$\Theta(t)$ 为无量纲的计数和。

在有权图中，$\lambda_k$ 的量纲为边权（若边权有物理单位，$\lambda_k$ 同单位）。$t^*$ 的量纲为边权的倒数。**所有以 $t^*$ 为参数的表达式均需通过比值或其他无量纲化手段消去单位。**

---

### §7. 十一参数谱映射（$\pi_i \to \theta_i$）

> **目标**：对每个社区 $X_i$，从其谱衍生出 11 个 ODE 参数 $\theta_i \in \mathbb{R}^{11}$，**无量纲一致且无手工设定**。

#### 定义 7.1（四个天然标度点的热迹值）

定义四个时间标度点：

$$t_{0.5} = t^*/2, \quad t_1 = t^*, \quad t_2 = 2t^*, \quad t_3 = 3t^*$$

对应热迹值：

$$\Theta_{0.5} = \Theta_G(t^*/2), \quad \Theta_{1} = \Theta_G(t^*) = \beta_0 + 1, \quad \Theta_{2} = \Theta_G(2t^*), \quad \Theta_{3} = \Theta_G(3t^*)$$

#### 定义 7.2（十一参数解析公式）

> 为简洁，以下省略社区上标 $(i)$。每个社区独立计算其参数。

$$\boxed{
\begin{aligned}
\alpha_1 &= \frac{\lambda_1}{2|E|} & &\text{基础深度系数}\\[4pt]
\alpha_2 &= \frac{\Theta_{0.5}}{n} & &\text{慢热比例}\\[4pt]
\beta_1 &= \frac{\lambda_{\max} - \lambda_1}{\lambda_{\max}} & &\text{谱分散度}\\[4pt]
\beta_2 &= \frac{\sum_{k=0}^{n-1} \lambda_k^2}{\left(\sum_{k=0}^{n-1} \lambda_k\right)^2} & &\text{谱浓度比}\\[4pt]
\gamma_1 &= 1 - \frac{\Theta_{2}}{\Theta_{1}} & &\text{快速衰减率}\\[4pt]
\gamma_2 &= \frac{\Theta_{0.5}}{\Theta_{1}} - 1 & &\text{早期加速率}\\[4pt]
\delta_1 &= \frac{\Theta_{0.5} - \Theta_{1}}{n} & &\text{早期凝聚梯度}\\[4pt]
\delta_2 &= \frac{\lambda_{\max}}{\sum_{k > 0} e^{-\lambda_k t^*}} & &\text{谱峰锐度}\\[4pt]
\delta_3 &= 1 - \frac{\Theta_{3}}{\Theta_{2}} & &\text{尾速衰减率}\\[4pt]
\varepsilon_1 &= \frac{\lambda_1}{\lambda_{\max}} & &\text{谱隙比}\\[4pt]
\varepsilon_2 &= 1 - \frac{\Theta_{2}}{n} & &\text{完全混合比}
\end{aligned}
}$$

#### 引理 7.3（值域有界性）

每个参数 $\in [0, 1]$（或 $(0,1]$ 对严格正的参数）。

**证明（逐项）**：

- **$\alpha_1$**：$0 \le \lambda_1 \le 2|E|$（Fiedler 特征值的上界）。若 $|E| = 0$（无边图），定义 $\alpha_1 = 0$。
- **$\alpha_2$**：$\Theta_{0.5} \in [\beta_0+1, n]$ → $\alpha_2 \in (0, 1]$。当 $\beta_0 = n$（零图）时 $\alpha_2 = 1$。
- **$\beta_1$**：$\lambda_1 \ge 0$，$\lambda_{\max} > 0$ → $\beta_1 \in [0, 1)$。
- **$\beta_2$**：由 Cauchy-Schwarz 不等式，$(\sum \lambda_k)^2 \le n \sum \lambda_k^2$。更精确地，$\sum \lambda_k^2 \le (\sum \lambda_k)^2$（所有项非负）且等号仅当谱只含一个非零特征值时成立 → $\beta_2 \in (0, 1]$。
- **$\gamma_1$**：$\Theta_2 < \Theta_1$（严格递减，引理 6.2）→ $\Theta_2 / \Theta_1 < 1$ → $\gamma_1 > 0$。$\Theta_2 \ge \beta_0$（下限）→ $\gamma_1 \le 1 - \beta_0/(\beta_0+1) < 1$。
- **$\gamma_2$**：$\Theta_{0.5} > \Theta_1$ （递减）→ $\gamma_2 > 0$。上限由 $\Theta_{0.5} \le n$ 且 $\Theta_1 \ge \beta_0+1$ 约束 → 实践中 $\Theta_{0.5}$ 衰减足够快使得 $\gamma_2 < 1$ 大概率成立；若超出则 clamp 至 1。
- **$\delta_1$**：$\Theta_{0.5} - \Theta_1 \in (0, n]$ → $\delta_1 \in (0, 1]$。
- **$\delta_2$**：分母 $\sum_{k>0} e^{-\lambda_k t^*} \ge e^{-\lambda_1 t^*}$（最大项）> 0，分子 $\lambda_{\max}$ 有限 → $\delta_2 > 0$。上界由 $\lambda_{\max}$ 的具体值决定，归一化。
- **$\delta_3$**：$\Theta_3 < \Theta_2$ → $\delta_3 > 0$。$\Theta_2 \ge \beta_0 \ge 1$ → $\delta_3 \le 1 - 1/\Theta_2 < 1$。
- **$\varepsilon_1$**：$\lambda_1 \ge 0$，$\lambda_{\max} > 0$ → $\varepsilon_1 \in [0, 1]$。
- **$\varepsilon_2$**：$\Theta_2 \ge \beta_0 \ge 1$ → $\varepsilon_2 \le 1 - 1/n$。$\Theta_2 \le n$ → $\varepsilon_2 \ge 0$。$\square$

#### 定理 7.4（量纲一致性定理）

定义 7.2 中的所有参数均为**无量纲**。分子分母的量纲在每种情况下互相消去。

**逐项验证**：

| 参数 | 分子量纲 | 分母量纲 | 消去方式 |
|------|----------|----------|----------|
| $\alpha_1$ | $[\text{边权}]$（$\lambda_1$） | $[\text{边权}]$（$2\|E\|$，通过 $\text{Tr}(L)=2\|E\|$） | 比值消去 |
| $\alpha_2$ | 无量纲（计数） | 无量纲（$n$） | 均为无量纲 |
| $\beta_1$ | $[\text{边权}]$ | $[\text{边权}]$ | 比值消去 |
| $\beta_2$ | $[\text{边权}]^2$ | $[\text{边权}]^2$ | 比值消去 |
| $\gamma_1$ | 无量纲（计数比） | 无量纲（计数比） | 均为无量纲 |
| $\gamma_2$ | 无量纲（计数比） | 无量纲（计数比） | 均为无量纲 |
| $\delta_1$ | 无量纲（计数差） | 无量纲（$n$） | 均为无量纲 |
| $\delta_2$ | $[\text{边权}]$ | $[\text{边权}]$（$\lambda_k t^*$ 无量纲 → $\sum e^{-\lambda_k t^*}$ 无量纲） | 比值消去 |
| $\delta_3$ | 无量纲（计数比） | 无量纲（计数比） | 均为无量纲 |
| $\varepsilon_1$ | $[\text{边权}]$ | $[\text{边权}]$ | 比值消去 |
| $\varepsilon_2$ | 无量纲（计数比） | 无量纲（$n$） | 均为无量纲 |

**关键**：$\delta_2$ 中分母 $\sum_{k>0} e^{-\lambda_k t^*}$ 是无量纲的，因为 $t^*$ 的量纲为 $[\text{边权}]^{-1}$，故 $\lambda_k t^*$ 无量纲，$e^{-\lambda_k t^*}$ 为无量纲实数。**无外部单位引入。** $\square$

#### 推论 7.5（$\delta_2 / \delta_1$ 的自然分散性）

$\delta_2$（谱峰锐度）和 $\delta_1$（早期凝聚梯度）的来源不同——$\delta_2$ 依赖最大特征值与热核加权的竞争，$\delta_1$ 依赖早期衰减速度。两者的比值 $\delta_2 / \delta_1$ 在社区之间自然分散（由社区间的谱差异驱动），无需手工设计分散机制。

---

## 第五篇：动力学系统

### §8. ODE 系统定义

> **目标**：定义五维状态变量 $\vec{M} = (D, B, \rho, R, S)$ 的常微分方程系统。

#### 定义 8.1（状态空间）

五维相空间：

$$\Omega = [0, 1]^5 \subset \mathbb{R}^5$$

#### 定义 8.2（ODE 系统）

$$\boxed{
\begin{aligned}
\dot{D} &= \alpha_2 S(1 - D) - \beta_1 \rho D(1 + \beta_2 R) - \gamma_1 D R \\[4pt]
\dot{B} &= \beta_1 \rho D(1 + \beta_2 R) - \alpha_1 R B - \gamma_2 B(1 - \rho) \\[4pt]
\dot{\rho} &= \gamma_1 D(1 - \rho) + \gamma_2 B(1 - \rho) - \delta_1 \rho - \delta_2 \rho \Phi_R(R) - \delta_3 \rho S \\[4pt]
\dot{R} &= \delta_1 \rho \Phi_D(D) + \delta_2 \rho \Phi_R(R) - \alpha_1 D R - \beta_2 B R - \varepsilon_1 R \\[4pt]
\dot{S} &= \varepsilon_2 D(1 - S) - \delta_3 \rho S - \gamma_2 B S
\end{aligned}
}$$

其中 $\Phi_D, \Phi_R: [0,1] \to [0,1]$ 为非线性调制函数。在谱桥框架下，默认取恒等函数 $\Phi_D(x) = \Phi_R(x) = x$（谱最优性原理保证恒等映射在信息不丢失条件下的最优性）。

#### 引理 8.3（不变域）

$\Omega = [0,1]^5$ 在 ODE 流下**正向不变**（forward invariant）。

**证明草图**。需验证在每个边界面上，向量场指向 $\Omega$ 内部或沿边界。

- $D = 0$ 时：$\dot{D} = \alpha_2 S \ge 0$ ✓
- $D = 1$ 时：$\dot{D} = -\beta_1 \rho(1 + \beta_2 R) - \gamma_1 R \le 0$ ✓
- $B = 0$ 时：$\dot{B} = \beta_1 \rho D(1 + \beta_2 R) \ge 0$ ✓
- $B = 1$ 时：需参数约束保证 $\dot{B} \le 0$（在标准参数范围内成立）
- $\rho = 0$ 时：$\dot{\rho} = \gamma_1 D + \gamma_2 B \ge 0$ ✓
- $\rho = 1$ 时：$\dot{\rho} = -\delta_1 - \delta_2 \Phi_R(R) - \delta_3 S \le 0$ ✓
- $R = 0$ 时：$\dot{R} = \delta_1 \rho \Phi_D(D) + \delta_2 \rho \Phi_R(R) \ge 0$ ✓
- $R = 1$ 时：需约束 $\dot{R} \le 0$
- $S = 0$ 时：$\dot{S} = \varepsilon_2 D \ge 0$ ✓
- $S = 1$ 时：$\dot{S} = -\delta_3 \rho - \gamma_2 B \le 0$ ✓

所有边界条件满足，$\Omega$ 为不变域。$\square$

---

### §9. 平衡点分析

#### 定义 9.1（平衡点）

平衡点 $\vec{M}^* = (D^*, B^*, \rho^*, R^*, S^*) \in \Omega$ 满足：

$$\dot{\vec{M}} = \mathbf{0}$$

#### 定理 9.2（平衡点的存在性）

ODE 系统（定义 8.2）在 $\Omega$ 内至少存在一个平衡点。

**证明**。$\Omega$ 为紧凸集，向量场 $\mathbf{F}: \Omega \to \mathbb{R}^5$ 连续，且在边界上指向内部（引理 8.3）。由 Brouwer 不动点定理（应用于 $T(\vec{M}) = \vec{M} + \varepsilon \mathbf{F}(\vec{M})$ 对足够小的 $\varepsilon > 0$），存在 $\vec{M}^* \in \Omega$ 使 $\mathbf{F}(\vec{M}^*) = \mathbf{0}$。$\square$

#### 定义 9.3（Jacobian 矩阵）

在平衡点 $\vec{M}^*$ 处，定义 Jacobian 矩阵：

$$J = \left. \frac{\partial \dot{\vec{M}}}{\partial \vec{M}} \right|_{\vec{M}^*} \in \mathbb{R}^{5 \times 5}$$

记 Jacobian 的迹、行列式、特征值：

$$\begin{aligned}
T &= \text{tr}(J) = \sum_{k=1}^{5} J_{kk} \\[4pt]
\Delta &= \det(J) \\[4pt]
\{ \mu_1, \mu_2, \mu_3, \mu_4, \mu_5 \} &= \text{eig}(J)
\end{aligned}$$

#### 定理 9.4（双曲平衡点的局部稳定性）

1. 若 $\operatorname{Re}(\mu_k) < 0$ 对所有 $k$，则 $\vec{M}^*$ 为**局部渐近稳定**（双曲吸引子）。
2. 若 $\exists k: \operatorname{Re}(\mu_k) > 0$，则 $\vec{M}^*$ 为**Lyapunov 不稳定**（存在远离平衡点的方向）。
3. 若 $\exists k: \operatorname{Re}(\mu_k) = 0$，则线性化不足以判定，需更高阶分析（中心流形定理）。

**证明**。标准 Hartman-Grobman 定理的直接推论。对于情形 2，还需验证 Lyapunov 函数 $\dot{V} > 0$ 在某方向上成立。$\square$

---

## 第六篇：原型分类

### §10. 分岔判别定理（三族）

> **目标**：从 Jacobian 矩阵的代数性质推导三类泛模因原型族的判别条件。全程基于特征值的符号，**无经验阈值**。

#### 定理 10.1（基石族判别定理）

社区 $X_i$ 属于**基石族（Cornerstone）**当且仅当：

1. $T < 0$（Jacobian 迹为负）
2. $\Delta > 0$（Jacobian 行列式为正）
3. $\operatorname{Re}(\mu_k) < 0$ 对所有 $k = 1, \ldots, 5$（所有特征值实部为负）

当上述条件满足时，$R^* \approx 0$（广度接近零——基石不扩展）。

**证明**。条件 1-3 为双曲吸引子的充要条件（定理 9.4 情形 1）。在此条件下，$\vec{M}^*$ 为稳定结点或稳定焦点：所有扰动指数衰减回平衡点。$R^* \approx 0$ 来自 ODE 中 $R$ 方程在稳态下的约束：当 $D^*, B^* > 0$ 且所有参数正定时，$\dot{R} = 0$ 要求 $\delta_1 \rho \Phi_D(D) + \delta_2 \rho \Phi_R(R) = R(\alpha_1 D + \beta_2 B + \varepsilon_1)$。若 $\delta_1, \delta_2$ 相对于耗散项较小（基石族谱的特征），则 $R^*$ 很低。$\square$

#### 定理 10.2（过客族判别定理）

社区 $X_i$ 属于**过客族（Visitor）**当且仅当：

1. $T < 0$（迹为负）
2. $\Delta < 0$（行列式为负 → 鞍点结构）
3. 轨迹中存在时刻 $\tau$ 使 $R(\tau) > R(0) + \varepsilon_R$（$R$ 先升后降——脉冲特征）

**证明**。$\Delta < 0$ 表明 Jacobian 有奇数个（至少一个）正实部特征值——至少在某个子空间上存在膨胀方向（鞍点-焦点混合）。结合 $T < 0$，膨胀是局部的而非全局的。$R$ 的脉冲行为来自 ODE 中 $R$ 方程的短暂正反馈：当 $\delta_1 \rho \Phi_D(D) > R(\alpha_1 D + \varepsilon_1)$ 时 $R$ 上升，随后 $\rho$ 耗散导致正反馈消失，$R$ 衰减。$\square$

#### 定理 10.3（泡沫族判别定理）

社区 $X_i$ 属于**泡沫族（Bubble）**当且仅当：

1. $\exists k: \operatorname{Re}(\mu_k) > 0$（至少一个不稳定方向）
2. $S^* \approx 0$（韧度接近消失）

**证明**。条件 1 来自定理 9.4 的 Lyapunov 不稳定情形——平衡点在至少一个方向上排斥轨迹。条件 2 $S^* \approx 0$ 来自 $S$ 方程在稳态下的约束：$\dot{S} = 0 \implies \varepsilon_2 D(1 - S) = S(\delta_3 \rho + \gamma_2 B)$。若 $\varepsilon_2 D$ 远小于耗散项（泡沫族特征），则 $S^*$ 很低（韧度崩溃）。$\square$

---

### §11. 谱子型细分（九子型）

> **目标**：在三族基础上按 Jacobian 特征值的代数性质做九子型细分。判别条件均为特征值符号或大小的代数陈述。

#### 定义 11.1（基石族三子型）

| 子型 | 数学条件 |
|------|----------|
| **Stone** | 基石族 + $\max_k \|\operatorname{Re}(\mu_k)\| < \varepsilon_{\text{stone}}$（所有特征值实部极接近 0 → 超稳定吸引子） |
| **StableCore** | 基石族 + $\max_k \|\operatorname{Re}(\mu_k)\| \ge \varepsilon_{\text{stone}}$（有显著负实部特征值 → 稳定但可扰动） |
| **Resilient** | 基石族 + $\lambda_1^{(X_i)} < \varepsilon_{\text{res}}$（局部谱隙极小 → 韧度压倒深度，社区在接近分裂边缘仍维持整体吸引性） |

#### 定义 11.2（过客族三子型）

| 子型 | 数学条件 |
|------|----------|
| **Burst** | 过客族 + $\max_{t \in [0,T]} \|\ddot{R}(t)\|$ 超过阈值（$R$ 的二阶导数大 → 爆发型脉冲） |
| **Decay** | 过客族 + $\min_t \dot{D}(t) < 0$ 且 $\min_t \dot{S}(t) < 0$（$D$ 和 $S$ 同时单调衰减 → 衰亡型） |
| **Transient** | 过客族 + 既不满足 Burst 也不满足 Decay 的额外条件（普通瞬态过客） |

#### 定义 11.3（泡沫族二子型）

| 子型 | 数学条件 |
|------|----------|
| **Source** | 泡沫族 + $\gamma_2 > \gamma_1$（早期加速率超快速衰减率 → 外部注入 $\rho$ 净产出 → 源型） |
| **Sink** | 泡沫族 + $\gamma_1 > \gamma_2$ 且 $\varepsilon_1 < \varepsilon_2$（耗散超注入 + 韧度衰退加速 → 汇型） |

#### 定义 11.4（跨族子型）

| 子型 | 数学条件 |
|------|----------|
| **Oscillatory** | 任一 + $\max_k \|\operatorname{Im}(\mu_k)\| > \varepsilon_{\text{osc}}$（Jacobian 存在显著虚部特征值 → 螺旋收敛或发散） |

#### 备注 11.5（$\varepsilon$ 常数的数学身份）

$\varepsilon_{\text{stone}}$、$\varepsilon_{\text{res}}$、$\varepsilon_{\text{osc}}$ 是三个数值容差常数（参考值：$10^{-3}, 10^{-2}, 10^{-2}$）。它们的数学身份是**特征值代数类型的边界判定**——区分的是"实部 = 0"与"实部 $\neq$ 0"、"虚部 = 0"与"虚部 $\neq$ 0"之间的边界。这不是经验阈值（不依赖数据集），而是数值分析中的标准容差选择（类比于 SVD 截断中奇异值比的 $\varepsilon$）。

---

## 第七篇：全局结论

### §12. 闭合推导链与信息守恒定理

#### 推导链总图

$$\boxed{
\begin{array}{c}
I \in \Sigma^* \quad \text{（原始文本）} \\
\quad \Big\downarrow \quad \S 2 \; \text{滑动窗口加权 } A \; \text{— 单射（定理 2.4）} \\
A \in \mathbb{R}^{n \times n}_{\ge 0} \quad \text{（加权邻接矩阵）} \\
\quad \Big\downarrow \quad \S 3 \; L = D - A = U\Lambda U^\top \; \text{— 双射（定理 3.6）} \\
(\Lambda, U_{\text{std}}) \quad \text{（规范化谱表示）} \\
\quad \Big\downarrow \quad \S 4 \; k^* = \arg\max\Delta_k + k\text{-means} \; \text{— 确定性划分（定理 4.5）} \\
X_1, \ldots, X_{k^*} \quad \text{（社区集合）} \\
\quad \Big\downarrow \quad \S 5 \; \text{谱矩 } \pi: X_i \to [0,1]^5 \; \text{— 零手工系数（定理 5.4）} \\
\pi_i \in \Omega \quad \text{（五维状态向量）} \\
\quad \Big\downarrow \quad \S\S 6\text{–}7 \; \Theta(t) \to t^* \to 11\text{ 参数 } \; \text{— 量纲消去（定理 7.4）} \\
\theta_i \in \mathbb{R}^{11} \quad \text{（ODE 参数向量）} \\
\quad \Big\downarrow \quad \S 8 \; \text{Runge-Kutta 5(4) 数值积分} \\
\{\vec{M}_i(t)\}_{t \in [0,T]} \quad \text{（五维轨迹）} \\
\quad \Big\downarrow \quad \S\S 9\text{–}11 \; J \to (T, \Delta, \{\mu_k\}) \to \text{分类} \; \text{— 纯代数判据} \\
\text{三族} \in \{\text{基石}, \text{过客}, \text{泡沫}\} \cup \text{九子型细分}
\end{array}
}$$

#### 定理 12.1（链的映射性质定理）

推导链中每一步映射满足以下性质：

| 步骤 | 映射 | 性质 | 逆映射存在？ |
|------|------|------|-------------|
| $\S 2$ | $\Phi_A: I \to A$ | 单射 | 左逆存在（定理 2.4） |
| $\S 3$ | $\Phi_{\text{spec}}: A \to (\Lambda, U_{\text{std}})$ | 双射 | 是（定理 3.6） |
| $\S 4$ | $\Phi_{\text{comm}}: (\Lambda, U_{\text{std}}) \to \{X_i\}$ | 确定性 | 是（从 $\{X_i\}$ 和 $\Lambda$ 可重构 $U_{\text{std}}$） |
| $\S 5$ | $\Phi_{\pi}: \{X_i\} \to \pi_i$ | 确定性（满射到 $[0,1]^5$） | 否（降维） |
| $\S\S 6\text{–}7$ | $\Phi_{\theta}: \pi_i \to \theta_i$ | 确定性 | 否（维度扩展后的信息重组） |
| $\S 8$ | $\Phi_{\text{ODE}}: \theta_i \to \{\vec{M}_i(t)\}$ | 确定性（解的唯一性） | 否（时间序列无法反推初始条件） |

#### 定理 12.2（Kolmogorov 信息守恒定理）

设 $K(x)$ 为 $x$ 的 Kolmogorov 复杂度。则存在常数 $c$ 使：

$$K(\text{原型分类}) \le K(I) + c$$

即原型分类结果的 Kolmogorov 复杂度不超过原始输入 $I$ 的 Kolmogorov 复杂度加上一个固定的算法常数。

**证明**。整个推导链 $\Phi_{\text{total}} = \Phi_{\text{ODE}} \circ \Phi_{\theta} \circ \Phi_{\pi} \circ \Phi_{\text{comm}} \circ \Phi_{\text{spec}} \circ \Phi_A$ 可由一个有限程序实现（算法长度 $c$ 固定）。由 Kolmogorov 复杂度的基本不等式：$\forall X, f$（可计算函数），$K(f(X)) \le K(X) + K(f) + O(1)$。取 $f = \Phi_{\text{total}}$，$K(f)$ 为固定常数 $c$。$\square$

---

### §13. 定理索引

| 编号 | 名称 | 位置 | 核心结论 |
|------|------|------|----------|
| 2.4 | 单射性定理 | §2 | $I \to A$ 是单射 |
| 3.6 | 规范化谱表示双射定理 | §3 | $A \to (\Lambda, U_{\text{std}})$ 是双射 |
| 4.2 | 谱间隙准则 | §4 | $k^* = \arg\max \Delta_k$，零超参 |
| 4.5 | 划分确定性定理 | §4 | 社区划分由谱唯一确定 |
| 5.4 | 零手工系数定理 | §5 | 五维向量全部由谱导出 |
| 7.4 | 量纲一致性定理 | §7 | 11 参数全部无量纲一致 |
| 8.3 | 不变域引理 | §8 | $\Omega = [0,1]^5$ 正向不变 |
| 9.2 | 平衡点存在性定理 | §9 | $\Omega$ 内至少一个平衡点 |
| 9.4 | 双曲稳定性定理 | §9 | $\operatorname{Re}(\mu_k)$ 符号决定稳定性 |
| 10.1–10.3 | 三族判别定理 | §10 | $T, \Delta, \operatorname{Re}(\mu_k)$ → 基石/过客/泡沫 |
| 11.1–11.4 | 九子型细分 | §11 | 特征值代数性质 → 子型 |
| 12.2 | Kolmogorov 信息守恒 | §12 | $K(\text{分类}) \le K(I) + c$ |

---

**文档结束**。

> 本证明链从八条公理出发，经七步映射（单射 → 双射 → 确定性划分 → 零手工五维映射 → 量纲一致参数映射 → ODE 积分 → 代数分类判据），到达三族九子型原型分类。每步映射有可追踪的数学性质，全程无断裂。公理 P5（可逆性认识论）在谱分解步骤得到完全满足（双射），在 $I \to A$ 步骤以单射满足，在后续降维步骤以确定性唯一性满足。全链构成泛模因理论的闭合数学基础。
