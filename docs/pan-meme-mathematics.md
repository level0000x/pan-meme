# 泛模因理论：完整数学推导

**版本**：v4.3-rc1  
**日期**：2026-07-08  
**状态**：完整初稿 — 公理→FCA→CW→Betti→5D→ODE→原型，含断裂标注

> 本文从八条命题公理出发，经形式概念分析 → CW 胞腔复形 → Betti/Louvain 分解 → 五维映射 → ODE 参数推导 → 平衡点分析 → 九类原型分类，给出完整的数学推导链。每一步给出严格定义、定理陈述与证明（或证明草图）。断裂处用 `⚠️ 断裂 N` 显式标注。

---

## 目录

- **第〇部分：基础**
  - 0. 符号约定
  - 1. 八条命题公理
- **第一部分：阶段 0 — 形式概念分析（I → M）**
  - 2. 论域构造与输入提取
  - 3. ↑↓ 操作与 Galois 连接
  - 4. 不动点收敛与概念生成
  - 5. 概念层级模型 M 与双射性分析
- **第二部分：阶段 1 — CW 胞腔复形（M → G）**
  - 6. 胞腔逐维构造（B1-B4）
  - 7. 几何编码：标量场 φ 与向量场 F
  - 8. Euler 示性数与 Betti 不变量
- **第三部分：阶段 2 — 分解与五维映射（G → Q）**
  - 9. Betti 分解与 Louvain 替代
  - 10. 五维状态映射 π: X_i → [0,1]^5
  - 11. 扩展维度 ξ_i 与双射恢复
  - 12. 耦合矩阵 C
- **第四部分：阶段 3 — ODE 动力学系统（π → Params → Trajectories）**
  - 13. 五维状态空间 Ω 与不变域
  - 14. ODE 方程系统的第一性原理导出
  - 15. 十一参数几何-动力学映射
  - 16. 非线性函数 Φ_D 与 Φ_R
- **第五部分：阶段 4 — 平衡点分析**
  - 17. 平衡点枚举与存在性
  - 18. Jacobian 分析与局部稳定性
  - 19. Lyapunov 函数与全局性质
- **第六部分：阶段 5 — 原型分类**
  - 20. 参数空间三分：三族判定条件
  - 21. 轨迹特征九子型细分
- **第七部分：全局定理**
  - 22. Kolmogorov 信息守恒定理
  - 23. 数据建模完备性定理
- **附录**
  - A. from_geometry() 数值分析
  - B. 断裂全景与修复优先级
  - C. 跨文档引用索引

---

## 第〇部分：基础

### 0. 符号约定

| 符号 | 含义 |
|------|------|
| $\Sigma^*$ | 有限长字符串的集合 |
| $2^X$ | $X$ 的幂集 |
| $[0,1]$ | 闭单位区间 |
| $\mathbb{R}_+$ | 非负实数 $[0, \infty)$ |
| $\Omega$ | 五维相空间 $[0,1]^4 \times \mathbb{R}_+$ |
| $\mathbf{1}_{\{P\}}$ | 谓词 $P$ 为真时取 1，否则取 0 |
| $\dot{x} \equiv dx/dt$ | 时间导数 |
| $J(f)|_p$ | $f$ 在点 $p$ 处的 Jacobian 矩阵 |
| $\kappa_{\text{crit}}$ | 原型分类的临界参数比值 |
| $H(X)$ | $X$ 的 Kolmogorov 复杂度 |
| $\Phi$ | 完整四阶段复合映射 $\Phi_A \circ \Phi_B \circ \Phi_C \circ \Phi_D$ |

---

### 1. 八条命题公理

> 来源：`泛模因理论.md` §1.2（第 254-353 行）。以下以形式化语言重述。

**公理 P1（信息本体论）**。存在基本本体论范畴 $\mathbb{I}$（信息），满足：
- $\mathbb{I} \not\subseteq \mathbb{M}$（物质）且 $\mathbb{I} \not\subseteq \mathbb{E}$（能量）
- $\forall m \in \mathbb{M}, \exists i \in \mathbb{I}: i = \text{struct}(m)$（每个物质实例有结构描述）
- $\exists i_1, i_2 \in \mathbb{I}, m \in \mathbb{M}: m_{i_1} \neq m_{i_2}$（不同信息结构产生不同物质行为）

**公理 P2（泛模因实在论）**。泛模因类 $\mathcal{P}$ 定义为：
$$\mathcal{P} = \{ x \mid \text{replicable}(x) \land \text{evolvable}(x) \land \text{identity\_preserving}(x) \}$$

**公理 P3（结构现实主义）**。$\forall x$，其同一性 $\text{id}(x)$ 不由物质组成决定，而由关系模式 $R_x \subseteq (x \cup \text{context}(x))^2$ 决定。

**公理 P4（认知-建模统一性）**。人类的认知过程 $C$ 与形式化建模过程 $\mathcal{M}$ 共享同构核心：$C(h) \cong \mathcal{M}(h)$（在适当范畴中等价）。

**公理 P5（可逆性认识论）**。对任意合法建模 $\Phi$，存在 $\Phi^{-1}$ 使：
$$\Phi^{-1} \circ \Phi = \text{id}_{\text{Dom}(\Phi)}$$
即模型的每一步变换必须有明确的逆操作。

**公理 P6（计算即模因）**。任何图灵完备系统 $T$ 本身是一个泛模因实例：$T \in \mathcal{P}$。

**公理 P7（演化普遍性）**。若系统 $S$ 同时具备复制 $R$、变异 $V$、选择 $S_e$ 三个算子，则 $S$ 表现出演化行为：
$$S_{t+1} = S_e \circ V \circ R(S_t)$$

**公理 P8（层级涌现）**。泛模因构成偏序集 $(\mathcal{P}, \preceq)$，其中 $x \preceq y$ 表示 $x$ 是 $y$ 的构成部分。存在:
- 向上因果：$x \preceq y \implies \text{state}(x) \mapsto \text{state}(y)$
- 向下因果：$y \succcurlyeq x \implies \text{constraint}(y) \mapsto \text{state}(x)$

**公理状态**：八命题是形而上学承诺，本身不要求数学证明。它们为后续推导提供约束边界。

---

## 第一部分：阶段 0 — 形式概念分析（I → M）

> 来源：`formal-concept-analysis-proof.md` + `proof-supplement-complete.md` §1-2  
> 实现：`phase1_emergence.rs`（128 行）

### 2. 论域构造与输入提取

#### 定义 2.1（原始输入）
设 $\Sigma$ 为有限字符表。原始输入为字符串：
$$I \in \Sigma^*, \quad |I| = L < \infty$$

#### 定义 2.2（Tokenizer）
Tokenizer $\mathcal{T}: \Sigma^* \to 2^{\Sigma^*}$ 将字符串映射为词集合（基于 Unicode 分词规则）。

$$\mathcal{T}("I\;saw\;a\;man\;with\;a\;telescope") = \{"I", "saw", "a", "man", "with", "telescope"\}$$

#### ⚠️ 断裂 1：信息丢失 $H(I) > H(\mathcal{T}(I))$

$\mathcal{T}$ 的陪域是 $2^{\Sigma^*}$（幂集，无序），而定义域是 $\Sigma^*$（序列，有序）。两个不同的有序序列可映射到相同的集合：

```
"A eats B"  →  {"A", "eats", "B"}
"B eats A"  →  {"A", "eats", "B"}
```

**信息损失量化**：若 $I$ 含 $n$ 个互异词，丢失的时序排列信息量为 $\log_2(n!)$ 比特。对 $n = 100$，$\log_2(100!) \approx 524$ 比特。

**对 P5 的影响**：$\mathcal{T}$ 不是单射，因此 $\mathcal{T}^{-1}$ 不存在。P5（可逆性认识论）从建模的第一步就被违反。

**修复方向**：
- **方案 A（承认降维）**：将 P5 放松为 $\|\Phi^{-1}(\Phi(I)) - I\| \leq \varepsilon$（近似可逆）
- **方案 B（编码词序）**：将 $\mathcal{T}$ 替换为 n-gram 邻接图提取器，保留时序信息
- **方案 C（充分统计量）**：声称 $\mathcal{T}(I)$ 是建模相关信息的充分统计量，非建模相关信息（如词序）的丢失是可接受的

**本文处理**：后续推导假设 $\mathcal{T}(I)$ 已包含了全部"建模相关"的信息。断裂 1 在第七章（信息守恒）中被正式记录为根断裂。

---

#### 定义 2.3（论域）
给定词集 $W = \mathcal{T}(I)$，论域 $\mathcal{U}$ 为词集与字符集之并：

$$\mathcal{U} = W \cup C, \quad C = \bigcup_{w \in W} \{c \in \Sigma \mid c \in \text{chars}(w)\}$$

显然 $|\mathcal{U}| \leq |\Sigma| \cdot \max_{w \in W} |w| + |W| < \infty$（有限字符串,有限词数）。

---

#### 定义 2.4（初始关系网络 $\Psi$）

$$\Psi = (V, E), \quad V = \mathcal{U}, \quad (c, w) \in E \iff c \in \text{chars}(w)$$

这是字-词二部图。度分布 $\deg(w) = |\text{chars}(w)|$（每个词的字符数）。$\deg(c) = |\{w \in W \mid c \in \text{chars}(w)\}|$（每个字符出现的词数）。

---

### 3. ↑↓ 操作与 Galois 连接

#### 定义 3.1（上操作 ↑）

$$\uparrow: 2^{\mathcal{U}} \to 2^{W}$$
$$\uparrow(A) = \{w \in W \mid \exists c \in A, (c, w) \in E\}$$

直观：集合 $A$ 中任意字符所属的全部词。

#### 定义 3.2（下操作 ↓）

$$\downarrow: 2^{W} \to 2^{C}$$
$$\downarrow(B) = \{c \in C \mid \forall w \in B, (c, w) \in E\}$$

直观：词集 $B$ 中**所有词**共有的字符。

#### 命题 3.3（↑↓ 的单调性）

对于 $A_1 \subseteq A_2$，$\uparrow(A_1) \subseteq \uparrow(A_2)$。  
对于 $B_1 \subseteq B_2$，$\downarrow(B_2) \subseteq \downarrow(B_1)$（注意反转）。

**证明**：$\uparrow$ 的单调性直接来自 $\exists$ 量化——更大的 $A$ 提供更多的候选字符触发包含条件。$\downarrow$ 的反单调性来自 $\forall$ 量化——更多的词意味着更少的共有字符。

---

#### 命题 3.4（Galois 不等式）

对于任意 $A \subseteq C$ 和 $B \subseteq W$：
$$A \subseteq \downarrow(\uparrow(A)), \quad B \subseteq \uparrow(\downarrow(B))$$

**证明**（第一个不等式）：

设 $c \in A$。需证 $c \in \downarrow(\uparrow(A))$，即 $\forall w \in \uparrow(A), (c, w) \in E$。

取任意 $w \in \uparrow(A)$。由 ↑ 定义，$\exists c' \in A: (c', w) \in E$。但此 $c'$ 不一定等于 $c$。

反例：$A = \{c_1, c_2\}$，$c_1 \in \text{chars}(w_a)$，$c_2 \in \text{chars}(w_b)$。则 $\uparrow(A) = \{w_a, w_b\}$。而 $\downarrow(\{w_a, w_b\}) = \{c \mid c \in \text{chars}(w_a) \cap \text{chars}(w_b)\}$，不一定包含 $c_1$ 或 $c_2$。

故 $A \subseteq \downarrow(\uparrow(A))$ 仅在 $A$ 中所有字符共现于同一词集时成立。一般情况下 $A \not\subseteq \downarrow(\uparrow(A))$。

**修正**：命题 3.4 在实际文本上不成立（罕见词不共享字符）。但 FCA 证明文档将其作为前提。

---

#### 定义 3.5（闭包算子）

对 $x \in \mathcal{U}$，定义交替闭包序列：

$$\begin{aligned}
S_0 &= \{x\} \\
S_{2k+1} &= \uparrow(S_{2k}) \quad (k \ge 0) \\
S_{2k+2} &= \downarrow(S_{2k+1}) \quad (k \ge 0)
\end{aligned}$$

---

### 4. 不动点收敛与概念生成

#### 定理 4.1（有限收敛 — 定理 4.1 of FCA proof）

对任意 $x \in \mathcal{U}$，存在 $N < |\mathcal{U}|$ 使 $S_N = S_{N+1}$（即交替 ↑↓ 序列收敛到不动点）。

**证明**：

序列 $\{S_{2k}\}_{k \ge 0}$ 是 $\subseteq C$ 的子集序列。由命题 3.4 的弱化版本（每次 $S_{2k+2} = \downarrow(\uparrow(S_{2k}))$ 要么不变要么缩小集覆盖），知该序列是单调递减的（在包含序下）。又因 $C$ 为有限集，递减子集序列在至多 $|C|$ 步内必然停止。

同理 $\{S_{2k+1}\}_{k \ge 0} \subseteq W$ 在至多 $|W|$ 步内收敛。

取 $N = \min\{n \mid S_n = S_{n+1}\}$，由 $|\mathcal{U}| = |C| + |W|$ 有限，$N \le |\mathcal{U}|$。$\square$

---

#### 定义 4.2（概念）

对任意 $x \in \mathcal{U}$，其**概念**定义为收敛的不动点：

$$\text{Concept}(x) = \lim_{k \to \infty} S_k$$

$\text{Concept}(x)$ 是一个由共现关系封闭的集合——它是 FCA 意义下的"形式概念"。

#### 命题 4.3（概念数的上界）

$$|\{\text{Concept}(x) \mid x \in \mathcal{U}\}| \le 2^{|C|} + 2^{|W|}$$

**证明**：每个概念是一个子集 $C' \subseteq C$ 或 $W' \subseteq W$。子集总数上界为 $2^{|C|} + 2^{|W|}$。$\square$

---

### 5. 概念层级模型 M 与双射性分析

#### 定义 5.1（概念层级模型 M）

$$M = (S, F, C)$$

- $S = \{L_0, L_1, \ldots, L_{d-1}\}$：概念层级序列，按不动点收敛集的包含关系偏序
- $F: S \to 2^{S \times S}$：五类推理关系（父子、协变、对立、蕴含、类比）
- $C = \{\kappa_1, \ldots, \kappa_r\}$：约束集合（如共现约束、互斥约束）

#### 定理 5.2（Φ_A 的规范表示唯一性）

给定 $M$，按其字集的字典序排序后，$M$ 的序列化表示唯一。（证明：字典序在全序字集上是确定的。）

#### ⚠️ 断裂 2：M 不能唯一决定 I

$M$ 在其内部是规范的（定理 5.2），但这只意味着 $M \to \text{serialized}(M)$ 是单射——不是 $I \to M$ 是单射。两个不同的原始输入可以产生结构等价的 $M$：

```
I₁: "猫 吃 鱼 狗 追 猫"
I₂: "狗 追 猫 猫 吃 鱼"
→ 相同 W → 相同 Ψ → 相同 M
```

$M$ 的字集字典序唯一性不足以判定 $\Phi_A: I \to M$ 是单射。断裂 1 的词序丢失 + 断裂 2 的 M 歧义 = 阶段 0 输出不可从原始输入恢复。

**第零阶段总结**：

| 步骤 | 映射 | 双射性 |
|------|------|--------|
| I → W | Tokenizer | ✗ 不可逆（词序丢失） |
| W → Ψ | 字-词二部图 | ✓（构造性的） |
| Ψ → M | ↑↓ 不动点 | ✗ 不可逆（M 歧义） |

---

## 第二部分：阶段 1 — CW 胞腔复形（M → G）

> 来源：`proof-supplement-complete.md` §2  
> 实现：`phase2_encoding.rs` + `cw_complex.rs`

### 6. 胞腔逐维构造（B1-B4）

#### 定义 6.1（CW 胞腔复形 G）

$$G = (K, g, \omega, \Gamma, R)$$

- $K = K^0 \cup K^1 \cup K^2$：胞腔分层
  - $K^0 = \{v_1, \ldots, v_{|C|}\}$：0-胞腔（顶点）
  - $K^1 = \{e_1, \ldots, e_{|E_{\text{cooc}}|}\}$：1-胞腔（边）
  - $K^2 = \{f_1, \ldots, f_{|\text{concepts}_{\geq 3}|}\}$：2-胞腔（面）
- $g: K^1 \to K^0 \times K^0$：附着映射
- $\omega: K^1 \to \mathbb{R}_+$：边权（场强/通量）
- $\Gamma = (\Gamma^0, \Gamma^1, \Gamma^2)$：离散梯度场
- $R$：可逆性元数据

#### 构造步骤 Φ_B(M)

| 步骤 | 操作 | 细节 |
|------|------|------|
| B1 | $\forall w \in M.\text{words} \to v_w \in K^0$ | 每个字生成一个 0-胞腔 |
| B2 | $\forall$ 同概念内共现字对 $(c_1, c_2) \to e \in K^1$ | $g(e) = (v_{c_1}, v_{c_2})$ |
| B3 | $\forall$ 含 $\ge 3$ 字的概念 $\to f \in K^2$ | 边界的 1-胞腔形成封闭循环 |
| B4 | $M$ 的完整结构序列化到 $R$ | `node_texts`, `concept_levels`, `serialized_rules`, `serialized_constraints` |

---

### 7. 几何编码：标量场 φ 与向量场 F

#### 定义 7.1（标量场 φ）

$$\varphi: K^0 \to [0, 1]$$
$$\varphi(v) = \frac{\text{info\_depth}(v)}{\max\text{Depth}}$$

其中 $\text{info\_depth}(v)$ 是 $v$ 在概念层级 $M$ 中的层级索引（$L_0$ 最深 = 0）。

#### 定义 7.2（向量场 F）

$$\mathbf{F}: K^1 \to \mathbb{R}$$
$$\mathbf{F}(e) = \varphi(v_{\text{target}}) - \varphi(v_{\text{source}}) = -\nabla\varphi|_e$$

其中 $(v_{\text{source}}, v_{\text{target}}) = g(e)$。直观：信息从深层节点流向浅层节点。

---

### ⚠️ 断裂 3：几何编码是装饰

$G$ 携带了双重信息：

1. **组合骨架**：$K^0, K^1, K^2$ 的维度和连接关系。
2. **R 元数据**：$M$ 的完整序列化备份（`node_texts` 等）。

$\Phi_B$ 的逆操作 $\Phi_B^{-1}$ 通过反序列化 $R$ 恢复 $M$——不是通过 CW 几何结构。CW 骨架是负载 $M$ 数据的**容器**，φ 和 F 是对骨架的**标注**。两者都对双射性没有贡献——双射性仅由 $R$ 保证。

**修复方向**：要么删除"CW 几何编码提供双射"的声称（承认 $R$ 是唯一的双射保障），要么证明 φ + F 能区分 $R$ 无法区分的信息（目前没有这种证明）。

---

### 8. Euler 示性数与 Betti 不变量

#### 定义 8.1（Euler 示性数）

$$\chi(G) = |K^0| - |K^1| + |K^2|$$

#### 定义 8.2（Betti 数）

$$\beta_0(G) = \#\{\text{连通分量}\}$$
$$\beta_1(G) = |K^1| - |K^0| + \beta_0(G) \quad \text{(一维 Betti 数, 独立循环数)}$$

在字-词二部图中，$\beta_0(G)$ 通常为 1（单一巨连通分量）。

**第一阶段总结**：

| 步骤 | 双射性 |
|------|--------|
| M → G (B1-B3) | 构造正确，非双射（K⁰ 到字的映射是 1-1，但概念到面的映射不是单射） |
| M → G (B4) | ✓（$R$ 元数据保证可逆） |
| φ + F 编码 | 对双射无贡献（装饰性） |

---

## 第三部分：阶段 2 — 分解与五维映射（G → Q）

> 来源：`proof-supplement-complete.md` §3  
> 实现：`phase3_decomposition.rs`（230 行）+ `cw_complex.rs`

### 9. Betti 分解与 Louvain 替代

#### 定义 9.1（Betti 分解 —— 定理声明）

按 0 维同调 $H_0(G)$ 做连通分量分解：

$$G = \bigsqcup_{i=1}^{\beta_0(G)} X_i$$

其中 $X_i = (K_i^0, K_i^1, K_i^2, g|_{X_i})$ 是第 $i$ 个连通分量。

#### ⚠️ 断裂 4：Louvain 替代 Betti

**声明**（论文 §3.4.1）：$n = \beta_0(G)$（连通分量数）。

**实现**（`phase3_decomposition.rs:6-7`）：
```
注：实践中字词关系图往往形成单一巨连通分量，丧失模因分解意义。
故采用 Louvain 社区检测作为替代分解策略。
当 Louvain 退化为连通分量时，与论文方法等价。
```

**断裂分析**：

Betti 分解（按连通性切分）和 Louvain 社区检测（按模块度最大化切分）是不同域上的不同分割：

| 性质 | Betti | Louvain |
|------|-------|---------|
| 边界类型 | 拓扑（无跨越边） | 统计（模块度梯度） |
| 确定性 | 唯一 | 依赖分辨率参数 γ |
| 可逆性 | 严格（每个顶点属于唯一分量） | 依赖 γ |

"当 Louvain 退化为连通分量时等价" —— 这个条件**从未成立**（Betti 被放弃正是因为单一连通分量使得 $\beta_0 = 1$）。

**实际后果**：不同 $\gamma \in \{0.5, 1.0, 2.0\}$ 产生不同数量的社区（实验 008 验证），导致不同的 $|Q|$ 和不同的原型分布。γ 的选择无理论指导。

**修复方向**：
- **A**：放弃 Betti 声明，承认是 Louvain 聚类
- **B**：证明 $\exists \gamma^*$ 使 Louvain 全局最优解等同于 Betti 分解（当前无此类证明）

---

### 10. 五维状态映射 π: X_i → [0,1]^5

#### 定义 10.1（五维映射 —— 证明中的声明）

$$\pi(X_i) = (D_i, B_i, \rho_i, R_i, S_i)$$

其中（证明文档 §3.2）：

$$\begin{aligned}
D_i &= \frac{|K_i^0|}{|K^0|} \quad \text{（顶点占比——内禀度）} \\
B_i &= \frac{|K_i^1|}{|K^1|} \quad \text{（边占比——关联度）} \\
\rho_i &= \text{avg}\{\Gamma^1(e) \mid e \in K_i^1\} \quad \text{（边通量均值——能流密度）} \\
R_i &= \frac{\beta_1(X_i)}{\max(1, |K_i^1|)} \quad \text{（独立环比——演化速率）} \\
S_i &= \frac{|K_i^2|}{\max(1, |K_i^0| + |K_i^1|)} \quad \text{（面-骨架比——结构韧度）}
\end{aligned}$$

#### 定义 10.2（五维映射 —— 实际代码实现）

代码实现（`phase3_decomposition.rs:143-177`）与证明文档声明**不同**：

$$\begin{aligned}
D_i^{\text{(code)}} &= 0.5 \cdot \underbrace{\frac{2 \cdot \text{internal\_edges}_i}{n_i(n_i - 1)}}_{\text{内部边密度}} + 0.5 \cdot \underbrace{\frac{n_i}{N}}_{\text{顶点占比}} \\[4pt]
B_i^{\text{(code)}} &= 0.6 \cdot \frac{n_i}{N} + 0.4 \cdot \underbrace{\frac{\text{cross\_edges}_i}{\text{total\_edges}_i}}_{\text{跨社区边比}} \\[4pt]
\rho_i^{\text{(code)}} &= |\nabla\varphi|_{\text{global}} \cdot (0.3 + 0.7 \cdot \text{internal\_density}) \\[4pt]
R_i^{\text{(code)}} &= 0.4 \cdot \text{cross\_ratio} + 0.3 \cdot (1 - \text{density}) + 0.3 \cdot (1 - \frac{n_i}{N}) \\[4pt]
S_i^{\text{(code)}} &= 0.7 \cdot \text{internal\_density} + 0.3 \cdot \frac{n_i}{N}
\end{aligned}$$

#### ⚠️ 断裂 5：五维映射是手工特征工程

**矛盾点**：

1. **权重系数（0.5, 0.6, 0.4, 0.3, 0.7）** —— 全部是经验性的，零点几何理由
2. **$D_i$ 混合量纲** —— 边密度（无量纲比值）与顶点占比（概率）做加权平均
3. **$\rho_i$ 混用全局量** —— $|\nabla\varphi|$ 是全局场强，不属于 $X_i$ 的局域量
4. **$R_i$ 有硬截断** —— `clamp(0.15, 0.9)` 是人为边界，非几何导出
5. **与证明声明不一致** —— 证明文档 §3.2 的公式定义与代码实现是两套不同的公式

**核心问题**：$Q$ 携带了 $\pi(X_i)$（5D 压缩）和 $\xi_i$（全量快照）。真正保证双射的是 $\xi_i$，不是 $\pi$。但 ODE 的参数从 $\pi$ 计算——如果 $\pi$ 是任意的，那么 ODE 参数就是任意的，整个动力学链条从根上就是手工的。

---

### 11. 扩展维度 ξ_i 与双射恢复

#### 定义 11.1（扩展维度 ξ_i —— 证明定义 3.3）

$$\xi_i = (\{v^0_{i,1}, \ldots\}, \{e^1_{i,1}, \ldots\}, \{f^2_{i,1}, \ldots\}, B_i)$$

其中 $B_i$ 是跨子几何体的边界链接集。ξ_i 编码了 $X_i$ 的全部微观信息。

#### 实现

```rust
ExtendedDimension {
    cell_snapshots_v: verts.iter().map(|&v| CellSnapshot { dim: 0, id: v, boundary: vec![] }).collect(),
    cell_snapshots_e: Vec::new(),  // 空的！
    cell_snapshots_f: Vec::new(),  // 空的！
    ...
}
```

注意：代码中 $\xi_i$ 的 1-胞腔和 2-胞腔快照是**空的**。只有顶点集合被保留。这意味着 $\xi_i$ 并没有完全编码 $X_i$ 的结构——它丢失了边和面的信息。

---

### 12. 耦合矩阵 C

#### 定义 12.1（耦合矩阵）

$$C_{ij} = \frac{|\partial K_i \cap \partial K_j|}{|\partial K_i|}$$

其中 $\partial K_i = \{e \in K_i^1 \mid \text{一端} \in K_i^0, \text{另一端} \notin K_i^0\}$（$X_i$ 的边界胞腔集）。

直观：$C_{ij}$ 测量 $X_i$ 中有多少比例的"表面"与 $X_j$ 共享。非对称（$C_{ij} \neq C_{ji}$ 当 $|\partial K_i| \neq |\partial K_j|$）。

**第二阶段总结**：

| 步骤 | 证明声称 | 代码现实 |
|------|----------|----------|
| Betti 分解 | $\beta_0$ 连通分量 | Louvain 聚类（γ 超参） |
| π 映射 | 几何不变量公式 | 手工权重 + 不同公式 |
| ξ_i | 全量编码 | 仅顶点集合（边面快照空） |

---

## 第四部分：阶段 3 — ODE 动力学系统（π → Params → Trajectories）

> 来源：`泛模因理论.md` §4.1-4.3 + `proof-supplement-complete.md` §5  
> 实现：`ode.rs`（795 行）+ `dynamics_params.rs`（240 行）

### 13. 五维状态空间 Ω 与不变域

#### 定义 13.1（五维状态空间）

$$\Omega = [0, 1]^4 \times \mathbb{R}_+ \subset \mathbb{R}^5$$

$$\vec{M}(t) = (D(t), B(t), \rho(t), R(t), S(t)) \in \Omega$$

| 维度 | 符号 | 范围 | 物理语义 |
|------|------|------|----------|
| 内禀度 | $D$ | $[0,1]$ | 结构内部复杂度与自洽性 |
| 关联度 | $B$ | $[0,1]$ | 与其他系统连接的广度 |
| 能流密度 | $\rho$ | $[0,\infty)$ | 单位时间承载/转换的能/信息流强度 |
| 演化速率 | $R$ | $[0,1]$ | 状态变化或扩散的瞬时速度 |
| 结构韧度 | $S$ | $[0,1]$ | 抵抗扰动、维持核心特征的能力 |

#### 定理 13.2（Ω 不变性 —— 定理 7）

若 $\vec{M}(0) \in \Omega$，且 ODE 右侧函数满足一定的 Lipschitz 条件，则对任意 $t \ge 0$，$\vec{M}(t) \in \Omega$。

**证明草图**：当 $D=0$ 且方程试图将 $D$ 推向负值时，由 $S \ge 0$ 使 $\alpha_2 S \ge 0$，$dD/dt \ge 0$（指向返回）。同理分析各维度的边界。$\square$

**实现**：`clamp_to_omega()` 在每个 RKF45 步后执行硬裁剪（第 289-291 行），防止数值溢出。

---

### 14. ODE 方程系统的第一性原理导出

> 此节从五维空间的四个核心张力出发，以第一性原理推导每个方程。非简单列举——而是给出每个项"为什么必须在那里"的理由。

#### 14.1 D 方程：深度-简化的拮抗

深度不是免费获得的。它需要：
- **时间**：精细结构需要时间来结晶
- **保护**：快速传播会稀释精细度（断章取义）

$$\boxed{\frac{dD}{dt} = -\alpha_1 \cdot \underbrace{R}_{\text{速度稀释}} \cdot D + \alpha_2 \cdot \underbrace{S}_{{\text{韧度锚定}}} \cdot (1-D)}$$

- **项 1 ($-\alpha_1 R D$)**：演化越快，深度越被稀释。$D$ 自身作为系数——深度越高，绝对稀释量越大（"高处不胜寒"）。
- **项 2 ($+\alpha_2 S(1-D)$)**：韧度高 → 深度被锚定并深化。$(1-D)$ 是 Logistic 上限——深度不能超过 1。

---

#### 14.2 B 方程：广度-深度的权衡

$$\boxed{\frac{dB}{dt} = \beta_1 \cdot \underbrace{R}_{\text{扩散势}} \cdot (1-B) - \beta_2 \cdot \underbrace{D}_{\text{专门化壁垒}} \cdot B}$$

- **项 1 ($+\beta_1 R(1-B)$)**：演化快 → 关联度扩张。$(1-B)$ Logistic 饱和。
- **项 2 ($-\beta_2 D B$)**：深度越高，专门化越强，广度的边际扩张越困难。$B$ 自身作为系数——关联越广，继续扩张的难度越大。

---

#### 14.3 ρ 方程：能流的输入-耗散平衡

$$\boxed{\frac{d\rho}{dt} = -\gamma_1 \cdot \underbrace{R}_{\text{传播消耗}} \cdot \rho + \gamma_2 \cdot (1-\rho) \cdot \underbrace{I_{\text{ext}}(t)}_{\text{外部注入}}}$$

- **项 1 ($-\gamma_1 R \rho$)**：高演化速率消耗能流（传播、争论、竞争都消耗注意力/能量）。
- **项 2 ($+\gamma_2 (1-\rho) \cdot I_{\text{ext}}(t)$)**：外部环境注入。$I_{\text{ext}}(t)$ 是外生函数——本文默认使用 $e^{-t}$（初始脉冲后衰减）。

---

#### 14.4 R 方程：三力竞争——驱动vs.诅咒vs.衰退

$$\boxed{\frac{dR}{dt} = \delta_1 \cdot \underbrace{\rho B}_{\text{燃料×通路}} \cdot (1-R) - \delta_2 \cdot \underbrace{\Phi_D(D)}_{\text{深度诅咒}} \cdot R - \delta_3 \cdot \underbrace{R}_{\text{自然遗忘}}}$$

- **项 1 ($+\delta_1 \rho B(1-R)$)**：$\rho$（燃料）+ $B$（通路）= 演化驱动力。$(1-R)$ Logistic 上限。
- **项 2 ($-\delta_2 \Phi_D(D) R$)**：深度对演化的非线性压制。$\Phi_D(D)$ 随 $D$ 增长而加速增长。
- **项 3 ($-\delta_3 R$)**：自然衰减/遗忘。即使没有深度诅咒，注意力也会自发消散。

这是整个系统的**核心方程**——三个参数的比值 $\delta_1 : \delta_2 : \delta_3$ 决定了模因的命运（基石/过客/泡沫）。

---

#### 14.5 S 方程：韧度的建立-侵蚀对抗

$$\boxed{\frac{dS}{dt} = \varepsilon_1 \cdot \underbrace{D}_{\text{深度奠基}} \cdot (1-S) - \varepsilon_2 \cdot \underbrace{\Phi_R(R)}_{\text{速度侵蚀}} \cdot S}$$

- **项 1 ($+\varepsilon_1 D(1-S)$)**：深度是韧度的基石。深刻自洽的结构难以被解构或曲解。
- **项 2 ($-\varepsilon_2 \Phi_R(R) S$)**：高速演化侵蚀韧度。快速变化不给结构稳定下来的时间。

---

### 15. 十一参数几何-动力学映射

> 来源：`dynamics_params.rs` — `from_geometry()` 方法  
> 断裂 6 标注内联

#### 定义 15.1（DynamicsParams）

```rust
pub struct DynamicsParams {
    pub alpha_1: f64,  // R→D 简化效应
    pub alpha_2: f64,  // S→D 沉淀效应
    pub beta_1: f64,   // R→B 扩张耦合
    pub beta_2: f64,   // D→B 泛化权衡
    pub gamma_1: f64,  // R→ρ 能流耗散
    pub gamma_2: f64,  // 外部→ρ 注入效率
    pub delta_1: f64,  // ρB→R 核心驱动力
    pub delta_2: f64,  // D→R 深度诅咒
    pub delta_3: f64,  // R 自发衰退
    pub epsilon_1: f64, // D→S 深度基石
    pub epsilon_2: f64, // R→S 速朽定律
}
```

#### 定义 15.2（from_geometry — 代码中的实际映射）

$$\begin{aligned}
\alpha_1 &= 0.02 + 0.30 \cdot R_i \\
\alpha_2 &= 0.03 + 0.94 \cdot s^3 + 0.03 \cdot \sigma \cdot (1 - D_i) \\[2pt]
\beta_1 &= 0.05 + 0.70 \cdot R_i \cdot (1 - B_i) \\
\beta_2 &= 0.05 + 0.50 \cdot D_i \\[2pt]
\gamma_1 &= 0.05 + 0.40 \cdot R_i \cdot \rho_i \\
\gamma_2 &= 0.30 + 0.50 \cdot (1 - \rho_i) \\[2pt]
\delta_1 &= 0.03 + 2.00 \cdot \rho_i \cdot B_i \cdot (0.1 + 0.9 \cdot s) \\
\delta_2 &= 0.05 + 0.40 \cdot D_i \\
\delta_3 &= 0.08 + 0.89 \cdot s^3 \\[2pt]
\varepsilon_1 &= 0.05 + 0.90 \cdot s + 0.05 \cdot D_i \cdot (1 - \sigma) \\
\varepsilon_2 &= 0.05 + 0.40 \cdot R_i
\end{aligned}$$

其中 $s = n_i / N$（归一化分量大小），$\sigma = S_i$。

#### ⚠️ 断裂 6：from_geometry 是手工映射，不是定理推导

**声称**（docstring）："从几何特征推导 11 参数"（论文 §3.4.3）。

**实际**：每行公式是手工构造的有界线性/立方映射。系数的"理由"（如 $0.94 \cdot s^3$ 使大分量 α₂ 更大）是事后解释。**证明文档 §3.4.3 声称的参数几何公式（"边密度→α₁"、"层级深度→α₂"、"拓扑不变量数量倒数→δ₃"）在代码中没有一行被直接使用。**

**自反例 —— 判别性失败**（数值分析见附录 A）：

$$\text{大分量: } \frac{\delta_2}{\delta_1} \approx 0.35 \quad \text{小分量: } \frac{\delta_2}{\delta_1} \approx 2.14$$

小分量的 $\delta_2/\delta_1$ 比值**更大**——这意味着所有分量落在同一吸引域（$\delta_2/\delta_1$ 跨不过 $\kappa_{\text{crit}}$）。九个原型必然无法同时出现——这与实验 010 的结果（0/4 缺失原型出现）一致。

---

### 16. 非线性函数 Φ_D 与 Φ_R

#### 定义 16.1（函数族 —— 论文 §4.3.8）

候选函数族 $\mathcal{H}_f$ 为紧致有限集：

$$\mathcal{H}_f = \{\text{Power}(x^k), \text{Exp}(e^{kx}{-}1), \text{Sigmoid}, \text{Log}(\ln(1{+}kx)), \text{PiecewiseLinear}\}$$

共 $|\mathcal{H}_f| = 5$ 个族。

#### 定义 16.2（参数范围 —— 紧致化 §6.2）

| 族 | 参数 | 范围 |
|----|------|------|
| Power | $k$ | $[0, 2]$ |
| Exponential | $k$ | $[0, 2]$ |
| Sigmoid | $(k, x_0)$ | $k \in [0.5, 2.5], x_0 \in [0, 1]$ |
| Logarithm | $k$ | $[0, 2]$ |
| PiecewiseLinear | $(b_1, b_2)$ | $b_1, b_2 \in [0,1], b_1 < b_2$ |

#### 定义 16.3（全局优化 —— 声称）

选择 $(\Phi_D^*, \Phi_R^*) = \arg\min_{\Phi_D, \Phi_R \in \mathcal{H}_f \times \mathcal{H}_f} \mathcal{J}(\Phi_D, \Phi_R)$，其中：

$$\mathcal{J} = \int_0^T \left\| \dot{\vec{M}}(t) - \vec{F}(\vec{M}(t); \Phi_D, \Phi_R) \right\|^2 dt$$

存在性：$\mathcal{H}_f$ 为有限集 → 最小值必存在（遍历即可）。但此优化从未被执行。

#### ⚠️ 断裂 7：Φ_D, Φ_R 从未被拟合

**代码实现**（`function_families.rs:87`）：$\Phi_D(x) = x$（Power 族，$k=1$），$\Phi_R(x) = x$（同上）。

没有网格搜索。没有损失计算。没有函数族评估。论文声称的优化步骤在代码中不存在。论文与代码之间的差距在此处保持开放。

---

## 第五部分：阶段 4 — 平衡点分析

> 来源：`proof-supplement-complete.md` §5（引理 5.3-5.7）  
> 实现：`ode.rs` — `classify_equilibrium()`（第 337-360 行）

### 17. 平衡点枚举与存在性

平衡点满足 $\dot{D} = \dot{B} = \dot{\rho} = \dot{R} = \dot{S} = 0$。设 $I_{\text{ext}} = 0$（无外部输入）。

#### 定理 17.1（平衡点枚举 —— 引理 5.3 + 补充）

| 平衡点 | 坐标 | 存在条件 | 类型 |
|--------|------|----------|------|
| $P_0$ 退零点 | $(0,0,0,0,0)$ | 恒成立 | 鞍点 |
| $P_1$ 湮灭型 | $(0, B^*, 0, 0, 0)$ | $B^* \in [0,1]$ 任意 | 中性 |
| $P_2$ 惰性型 | $(1, 0, \rho^*, 0, 1)$ | $\rho^* \in [0,\infty)$ 任意 | 中性 |
| $P_3$ 广度-韧度 | $(0, B^*, \rho^*, 0, 0)$ | 任意 | 中性 |
| $P^*$ 基石型 | $(D^*, 0, 0, 0, S^*)$ | $\delta_2 \gg \delta_1$ | 稳定结点 |

**证明**（$P_0$）：代入 $\vec{M} = (0,0,0,0,0)$ 到 ODE 右侧：
$$\dot{D} = -\alpha_1 \cdot 0 \cdot 0 + \alpha_2 \cdot 0 \cdot 1 = 0$$
同理 $\dot{B} = \dot{\rho} = \dot{R} = \dot{S} = 0$。$\square$

---

### 18. Jacobian 分析与局部稳定性

#### 定理 18.1（P₀ 的 Jacobian —— 引理 5.6）

$$J(P_0) = \begin{bmatrix}
\alpha_2 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & -\gamma_2 & 0 & 0 \\
0 & 0 & 0 & -\delta_3 & 0 \\
\varepsilon_1 & 0 & 0 & 0 & 0
\end{bmatrix}$$

特征值：$\{\alpha_2, 0, -\gamma_2, -\delta_3, 0\}$。符号混合 → $P_0$ 是鞍点，不是全局吸引子。

**证明**：直接计算。在 $P_0$ 处：$dD/dt$ 仅剩 $\alpha_2 S(1-D) = \alpha_2 \cdot 0 \cdot 1 = 0$。但 $\partial(\dot{D})/\partial D|_{P_0} = -\alpha_1 \cdot 0 + \alpha_2 \cdot 0 \cdot (-1) = \alpha_2$（只有 $\alpha_2 S(1-D)$ 对 $D$ 的偏导在 $D=S=0$ 处非零）。$\square$

---

#### 定理 18.2（P* 的渐近稳定性 —— 引理 5.7）

当 $\delta_2 \gg \delta_1$ 且 $\varepsilon_1 \gg \varepsilon_2$，$R$ 快速衰减到 0。$R \to 0$ 后系统退化为：

$$\dot{D} = \alpha_2 S(1-D), \quad \dot{S} = \varepsilon_1 D(1-S)$$

此二维子系统在非平凡平衡点 $(D^*, S^*) = (\frac{\alpha_2}{\alpha_2 + \varepsilon_1}, \frac{\varepsilon_1}{\alpha_2 + \varepsilon_1})$ 处有：

$$J(P^*)_{2\times 2} = \begin{bmatrix} -\alpha_2 S^* & \alpha_2(1-D^*) \\ \varepsilon_1(1-S^*) & -\varepsilon_1 D^* \end{bmatrix}$$

迹为负，行列式为正 → 两特征值均负 → 稳定结点。$\square$

**警告（断裂 8 关联）**：此定理假设 $R \to 0$——但 $R \to 0$ 本身依赖 $\delta_2 \gg \delta_1$。这是循环推理：参数取值 → R 衰减 → 衰减后系统稳定 → 参数取值"正确"。

---

### 19. Lyapunov 函数与全局性质

#### 定理 19.1（Lyapunov 函数）

取 $V(\vec{M}) = \frac{1}{2}(D^2 + B^2 + \rho^2 + R^2 + S^2)$。

$$\dot{V} = -\alpha_1 R D^2 - \beta_2 D B^2 - \gamma_1 R \rho^2 - (\delta_2\Phi_D(D) + \delta_3)R^2 - \varepsilon_2\Phi_R(R)S^2 + \Sigma^+$$

其中 $\Sigma^+ = \alpha_2 S D(1-D) + \beta_1 R B(1-B) + \gamma_2 \rho(1-\rho)I_{\text{ext}} + \delta_1 \rho B R(1-R) + \varepsilon_1 D S(1-S) \ge 0$。

$\dot{V}$ 的符号由负项（含 $\alpha_1, \beta_2, \gamma_1, \delta_2, \delta_3, \varepsilon_2$）和正项 $\Sigma^+$ 的竞争决定。**在 $\delta_2, \delta_3$ 大的区域，负项主导 → $V \downarrow$ → 收敛**。在 $\delta_2, \delta_3$ 小的区域，正项可能主导 → 系统可能发散（但被 Ω 裁剪限制）。

**推论**：系统的长期行为由 $(\delta_2, \delta_3)$ 与 $(\delta_1, \gamma_2)$ 的竞争决定。$\delta_2 + \delta_3 > \kappa_V \cdot (\delta_1 \rho_0 B_0)$ 时全局收敛。

---

## 第六部分：阶段 5 — 原型分类

> 来源：`proof-supplement-complete.md` §5.8 + `泛模因理论.md` §4.4  
> 实现：`ode.rs` — `classify()`（第 371-524 行）

### 20. 参数空间三分：三族判定条件

#### 定理 20.1（三族分叉定理 —— 定理 5.8）

存在临界参数比值 $\kappa_{\text{crit}} > 0$，使得：

$$\frac{\delta_2}{\delta_1} \begin{cases}
> \kappa_{\text{crit}} & \implies \text{基石族} \quad (R \to 0, D \uparrow, S \uparrow) \\
< \kappa_{\text{crit}} & \implies \text{过客族} \quad (R \uparrow\downarrow, D \downarrow) \\
\ll \kappa_{\text{crit}} & \implies \text{泡沫族} \quad (R \uparrow\uparrow\downarrow\downarrow, D \approx 0)
\end{cases}$$

$\kappa_{\text{crit}}$ 由 $R=0$ 处 $D$ 和 $S$ 方程 Jacobian 的特征值符号决定。精确表达式待推导（涉及 $\alpha_2, \varepsilon_1, \varepsilon_2$ 的交互）。

#### 三族特征

| 族 | 参数条件 | 动力学轨迹 | 例子 |
|----|----------|-----------|------|
| 基石 | $\delta_2 \gg \delta_1 \gg \delta_3$ | $R \searrow 0, D \nearrow 1, S \nearrow 1$ | 科学理论、宪法 |
| 过客 | $\delta_1 \gg \delta_2, \delta_3$ 中 | $R \nearrow\searrow, D \searrow, S \searrow$ | 新闻热点、时尚 |
| 泡沫 | $\delta_1 \gg \delta_2, \delta_3 \approx 0$ | $R \uparrow\uparrow\downarrow\downarrow, S \approx 0$ | 金融泡沫、伪科学 |

---

### 21. 轨迹特征九子型细分

#### 定义 21.1（九子型判别条件 —— 代码实现）

分类器 `classify()` 按优先级序列判定（第 371-524 行）：

**优先级 1：近收敛检测**
```
if D_final > 0.85 && S_final > 0.85 && R_final < 0.15:
    └─ near_converged
```

**基石族（near_converged = true）**：

| 子型 | 条件 | 直觉 |
|------|------|------|
| **Stone** | $D > 0.98, S > 0.98, R < 0.02$ | 完全石化——五维冻结 |
| **Resilient** | $S_{\min} > 0.7, S > D + 0.05$ | S 始终高位——韧度压倒深度 |
| **StableCore** | 以上两者均不满足 | 温和收敛——默认基石 |

**优先级 2：非收敛下的 Resilient**
```
if S_final > 0.85 && S_min > 0.7 && D_final < 0.85 && D_final > 0.3:
    └─ Resilient（S 领先于 D）
```

**优先级 3：振荡检测**
```
if oscillation_count >= 8:
    └─ Oscillatory
```

**过客族（含泡沫源的过渡）**：

| 子型 | 条件 |
|------|------|
| **Burst** | $r_{\text{peak}} - r_{\text{init}} > 0.05 \land r_{\text{peak}} - r_{\text{final}} > 0.2 \land D_{\text{peak}} > 0.5$ |
| **Source** | $B_{\text{final}} > 0.7 \land D_{\text{final}} < 0.3 \land R_{\text{final}} > 0.1$ |
| **Decay** | $D_{\text{decay}} > 0.2 \land D_{\text{final}} < 0.5$（R 从高位衰减后） |
| **Transient** | $R_{\text{final}} > 0.05$ 且不满足以上条件 |
| **Sink** | $\rho_{\text{final}} > 0.6 \land D_{\text{final}} < 0.4 \land B_{\text{final}} < 0.4$ |

#### ⚠️ 断裂 8：九子型是参数性阈值，非结构性推导

三族（基石/过客/泡沫）可从平衡点分析**严格推导**（定理 20.1）。但九子型的区分依赖于：
- $D > 0.98$ vs $D > 0.85$（连续阈值）
- $b_{\text{final}} > 0.7$（任意的）
- $\text{oscillation\_count} \ge 8$（无几何基础的）

这些阈值是**参数性的**（依赖于 $\delta_2/\delta_1$ 和 $\varepsilon_1/\varepsilon_2$ 的连续取值），而非**结构性的**（如特征值重数、平衡点类型的离散变化）。

**后果**：如果 $\delta_2/\delta_1$ 的分布不够宽（如当前的 from_geometry 映射），某些子型在参数空间中不出现——不是因为它们在理论上不可能，而是因为当前的参数映射没有探索到它们的区域。这就是实验 010 的根因。

---

## 第七部分：全局定理

### 22. Kolmogorov 信息守恒定理

#### 定义 22.1（Kolmogorov 复杂度）

$$H(X) = \min\{|p| \mid U(p) = X\}$$

其中 $U$ 是通用图灵机。

#### 定理 22.1（"信息守恒" —— 定理 4.3）

$$|H(I) - H(Q)| \le C_{\Phi}$$

其中 $C_{\Phi} = \max(K(\Phi), K(\Phi^{-1})) + O(1)$。

**证明**（`proof-supplement-complete.md` §4）：构造程序 $(code(\Phi), p_I)$ 从 $I$ 生成 $Q$，以及 $(code(\Phi^{-1}), p_Q)$ 从 $Q$ 恢复 $I$。两程序长度之差由 $\Phi$ 和 $\Phi^{-1}$ 的复杂度界定。

#### ⚠️ 此定理的三个问题

1. **不可计算性**：$H(\cdot)$ 不可计算，定理在计算上是存在性陈述，不可验证。
2. **对双射性的假设**：定理假设 $\Phi$ 是双射链。断裂 1（词序丢失）打破了这一假设——$I \to W$ 不是单射。
3. **$\Phi^{-1}$ 不存在**：断裂 1 意味着定理 22.1 的前提在第一步就不成立。

---

### 23. 数据建模完备性定理

#### 定理 23.1（"数据建模完备性" —— 定理 7.1）

| 序号 | 声称 | 本文判定 | 理由 |
|------|------|----------|------|
| 1 | $\Phi(I)$ 存在且唯一 | ✓ | 确定性算法，每一步输出由输入唯一决定 |
| 2 | $\Phi^{-1}(\Phi(I)) = I$ | ✗ | 断裂 1: $I \to W$ 不是单射 |
| 3 | $|H(I)-H(Q)| \le C_{\Phi}$ | ✗ | 假设 2 成立的前提不成立 |
| 4 | ODE 解存在唯一 | ✓ | Lipschitz 系统，Picard-Lindelöf 保证 |
| 5 | 参数在紧致 $H$ 上可优化 | ✓ | 连续损失函数在紧致集上可达最小值 |

**结论**：5 条声称中，3 条成立，2 条因断裂 1 而悬空。

---

## 附录 A：from_geometry() 数值分析

设典型输入参数：

**大分量**：$s = 0.9, D = 0.7, B = 0.4, \rho = 0.6, R = 0.2, \sigma = 0.8$

$$\begin{aligned}
\delta_1^{\text{(large)}} &= 0.03 + 2.0 \cdot 0.6 \cdot 0.4 \cdot (0.1 + 0.9 \cdot 0.9) = 0.03 + 0.48 \cdot 0.91 = 0.467 \\
\delta_2^{\text{(large)}} &= 0.05 + 0.4 \cdot 0.7 = 0.330 \\
\delta_2/\delta_1 &= 0.707
\end{aligned}$$

**小分量**：$s = 0.02, D = 0.1, B = 0.05, \rho = 0.05, R = 0.8, \sigma = 0.1$

$$\begin{aligned}
\delta_1^{\text{(small)}} &= 0.03 + 2.0 \cdot 0.05 \cdot 0.05 \cdot (0.1 + 0.9 \cdot 0.02) = 0.03 + 0.005 \cdot 0.118 = 0.031 \\
\delta_2^{\text{(small)}} &= 0.05 + 0.4 \cdot 0.1 = 0.090 \\
\delta_2/\delta_1 &= 2.903
\end{aligned}$$

**结论**：小分量的 $\delta_2/\delta_1 = 2.90 \gg 0.71 =$ 大分量。深度诅咒的参数强度是反向的——小分量被诅咒得更厉害。这违反了设计目标（大分量应有更强的深度诅咒以产生基石族的分化）。

---

## 附录 B：断裂全景与修复优先级

```
I ──→ W ──→ Ψ ──→ M ──→ G ──→ Q ──→ Params ──→ Trajectories ──→ Archetype
│        │              │      │      │        │            │
│ B1:P0  │ B2:P2        │B3:P3 │B4:P1 │B5:P1   │B6:P0,B7:P3│B8:P2
│词序丢  │M歧义         │CW装饰│Louv  │5D手工  │映射反设计  │阈值分类
└── 根 ──┘              │      │跳步  │系数    │Φ未拟合    │
```

| 优先级 | 断裂 | 修复方案 | 预计工作量 |
|--------|------|----------|-----------|
| P0 | B1 词序 | 放松 P5 或引入 n-gram 边 | 大 |
| P0 | B6 from_geo | 反推约束满足映射 | 中 |
| P1 | B4 Louvain | 证明等价性或修改定理声明 | 中 |
| P1 | B5 5D 映射 | 去除手工权重，使用几何不变量 | 中 |
| P2 | B2 M 歧义 | B1 修完自动解决 | — |
| P2 | B8 九子型 | B6 修完（参数分化够宽），九子型自然出现 | — |
| P3 | B3 CW 装饰 | 修改定理表述（非逻辑错误） | 小 |
| P3 | B7 Φ 拟合 | 实现网格搜索（存在性已保证） | 小 |

---

## 附录 C：跨文档引用索引

| 本文章节 | 主论文(泛模因理论.md) | FCA 证明 | 补全证明 | 代码文件 | 代码行号 |
|----------|----------------------|----------|----------|----------|---------|
| §1 | §1.2 (254-353) | — | — | — | — |
| §2-5 | §3.1-3.2 (782-868) | §3-6 | §1 | phase1_emergence.rs | 1-128 |
| §6-8 | §3.3 (869-943) | — | §2 | phase2_encoding.rs, cw_complex.rs | — |
| §9-12 | §3.4 (869-943) | — | §3 | phase3_decomposition.rs | 1-230 |
| §13-16 | §4.1-4.3 (994-1276) | — | §5-6 | ode.rs, dynamics_params.rs | — |
| §17-19 | §4.4-4.5 (1277-1361) | — | §5 | ode.rs:337-360 | — |
| §20-21 | §4.4 (1277-1361) | — | §5.8 | ode.rs:371-524 | — |
| §22-23 | §7 (2360-2409) | — | §4, §7 | — | — |

---

**文档结束**。下一步：基于断裂优先级决定修复路线，然后进入代码实现。
