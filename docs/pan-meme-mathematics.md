# 泛模因理论：统一数学推导

**版本**：v4.3 → v4.4 预备  
**日期**：2026-07-08  
**依赖**：`泛模因理论.md`（§1-4）+ `formal-concept-analysis-proof.md`（阶段 0）+ `proof-supplement-complete.md`（阶段 1-3 + ODE 分类）

> 本文梳理从八条命题公理到 ODE 参数推导再到原型分类的完整数学链。目标：使参数推导 `from_geometry()` 不再基于启发式，而基于这条链提供的有界性约束和映射方向。

---

## 第〇章：公理基座 — 八命题

取自 `泛模因理论.md` §1.2。

**P1（信息本体论）**：信息是现实的独立构成维度，不可还原为物质-能量。  
**P2（泛模因实在论）**：一切可复制、可演化、可保持同一性的信息-结构模式，都是泛模因。  
**P3（结构现实主义）**：持久存在的是实体间的稳定关系模式（结构），而非具体实体。  
**P4（认知-建模统一性）**：认知即建模。理论是认知建模的形式化延伸。  
**P5（可逆性认识论）**：模型必须可逆映射回现实——建模的每一步都是可追溯的。  
**P6（计算即模因）**：图灵完备性是泛模因系统的逻辑延伸，不是外部能力。  
**P7（演化普遍性）**：复制 + 变异 + 选择 → 演化。不限于生物。  
**P8（层级涌现）**：低层模因组合形成高层模因，双向因果，无本体论特权层。

**约束**：P5（可逆性）要求建模的每一步都是双射，信息不可丢失。

---

## 第一章：阶段 0 — 形式概念分析（FCA）

> 来源：`formal-concept-analysis-proof.md` + `proof-supplement-complete.md` §2-3

### 1.1 论域定义

给定输入字符串 $I \in \Sigma^*$（有限长）。提取：
- 字集 $C = \bigcup_{w \in W} \text{chars}(w)$ —— 从词自暴露
- 词集 $W = \{w_1, \ldots, w_m\}$ —— tokenizer 输出

论域 $\mathcal{U} = W \cup C$。

### 1.2 初始关系网络

$$\Psi = A(W) = (V, E), \quad V = \mathcal{U}, \quad (c, w) \in E \iff c \in \text{chars}(w)$$

### 1.3 ↑↓ 操作

$$\uparrow(c) = \{w \in W \mid (c, w) \in E\}$$
$$\downarrow(w) = \{c \in C \mid (c, w) \in E\} = \text{chars}(w)$$

互逆性（P5）：$\uparrow \circ \downarrow$ 和 $\downarrow \circ \uparrow$ 在集合论意义上是严格恒等。

### 1.4 不动点收敛

对任意 $x \in \mathcal{U}$，↑↓ 交替循环：
$$S_0 = \{x\}, \quad S_{k+1} = \begin{cases} \uparrow(S_k) & k \text{ even} \\ \downarrow(S_k) & k \text{ odd} \end{cases}$$

由前提 0（有限层级），$\exists N$ 使 $S_N = S_{N+1}$。收敛的不动点即为**概念**——由共现关系封闭的字集。

### 1.5 输出：概念层级模型 $M$

$$M = (S, F, C)$$

- $S = \{L_0, L_1, \ldots, L_{d-1}\}$：概念层级序列，$L_k = \{c_{k,1}, \ldots, c_{k,n_k}\}$
- $F: S \to 2^{S \times S}$：五类推理关系（父子、协变、对立、蕴含、类比）
- $C = \{\kappa_1, \ldots, \kappa_r\}$：约束集合

**双射性**（证明 §2.3-2.4）：$\Phi_A: I \leftrightarrow M$ 是双射。$M$ 的字集字典序下的规范表示唯一。逆操作 $\Phi_A^{-1}$ 由去重和文本重组定义。

---

## 第二章：阶段 1 — CW 胞腔复形 (M → G)

> 来源：`proof-supplement-complete.md` §2 + `phase2_encoding.rs`

### 2.1 构造 $\Phi_B: M \to G$

$$G = (K, g, \omega, \Gamma, R)$$

| 步骤 | 操作 | 数学含义 |
|------|------|----------|
| B1 | $M$ 中每个字 $w$ → 0-胞腔 $v_w \in K^0$ | $K^0 = \{v_w: w \in \text{word\_set}(M)\}$ |
| B2 | 同概念内共现字对 → 1-胞腔 $e \in K^1$ | $g(e) = (v_p, v_q)$ |
| B3 | 含 $\ge 3$ 字的概念 → 2-胞腔 $f \in K^2$ | 编码概念封闭性 |
| B4 | $M$ 完整序列化入 $R$ | 无损编码 |

### 2.2 几何编码

在 CW 骨架上叠加：
- **标量场** $\varphi(v) = \text{info\_depth}(v)$（节点在概念层级中的深度）
- **向量场** $\mathbf{F} = -\nabla\varphi$（信息势的梯度 → 边上的信息流向）

### 2.3 不变量

$$\chi = |K^0| - |K^1| + |K^2| \quad \text{(Euler 示性数)}$$
$$\beta_0 = \#\{\text{连通分量}\}, \quad \beta_1 = |E| - |V| + \beta_0 \quad \text{(Betti 数)}$$

### 2.4 双射性

$\Phi_B$ 在 $\text{Im}(\Phi_B)$ 上是双射。逆操作从 $R$ 反序列化恢复 $M$。$R$ 的全部字段（`node_texts`, `concept_levels`, `serialized_rules`, `serialized_constraints`）使 $\Phi_B^{-1}$ 可计算。

---

## 第三章：阶段 2 — Betti 分解 + 五维映射 (G → Q)

> 来源：`proof-supplement-complete.md` §3

### 3.1 Betti 分解

按 $G$ 的 0 维同调 $H_0(G)$ 做连通分量分解：

$$G = \bigsqcup_{i=1}^{N} X_i, \quad X_i = (K_i^0, K_i^1, K_i^2, g|_{X_i})$$

### 3.2 五维映射 $\pi: X_i \to [0,1]^5$

对每个子几何体 $X_i$：

$$\pi(X_i) = (D_i, B_i, \rho_i, R_i, S_i)$$

| 维度 | 公式 | 理论含义 |
|------|------|----------|
| $D_i$ | $|K_i^0| / |K^0|$ | 内禀度：该分量占整体的信息深度比 |
| $B_i$ | $|K_i^1| / |K^1|$ | 关联度：该分量占整体的连接广度比 |
| $\rho_i$ | $\text{avg}\{\omega(e) : e \in K_i^1\}$ | 能流密度：平均边场强 |
| $R_i$ | $\beta_1(X_i) / \text{max}(1, |K_i^1|)$ | 演化速率：独立环数 vs 边数比 |
| $S_i$ | $|K_i^2| / \text{max}(1, |K_i^0| + |K_i^1|)$ | 结构韧度：面-骨架比 |

**关键性质**：五个维度全部取自几何不变量，不引入外部信息。这满足了 P5（可逆性）。

### 3.3 跨分量耦合

$$C = \{(i, j, \gamma_{ij})\}, \quad \gamma_{ij} = \frac{|\partial K_i \cap \partial K_j|}{|\partial K_i|}$$

其中 $\partial K_i$ 是 $X_i$ 的边界胞腔集。

### 3.4 输出

$$Q = (\{X_i, \pi(X_i), \theta_i\}_{i=1}^N, C)$$

双射性由 $\xi_i$ 中的 CellSnapshot 保证（证明 §3.3-3.4）。

---

## 第四章：阶段 3 — ODE 参数推导 (Q → Params)

> 来源：`proof-supplement-complete.md` §5 + `dynamics_params.rs`

### 4.1 11 参数 ODE 系统

$$\begin{aligned}
\frac{dD}{dt} &= -\alpha_1 R D + \alpha_2 S (1 - D) \\[2pt]
\frac{dB}{dt} &= \beta_1 R (1 - B) - \beta_2 D B \\[2pt]
\frac{d\rho}{dt} &= -\gamma_1 R \rho + \gamma_2 (1 - \rho) \cdot I_{\text{ext}}(t) \\[2pt]
\frac{dR}{dt} &= \delta_1 \rho B (1 - R) - \delta_2 \Phi_D(D) R - \delta_3 R \\[2pt]
\frac{dS}{dt} &= \varepsilon_1 D (1 - S) - \varepsilon_2 \Phi_R(R) S
\end{aligned}$$

### 4.2 参数语义

| 参数 | 所在方程 | 语义 | 几何映射方向 |
|------|---------|------|-------------|
| $\alpha_1$ | D | R→D 简化效应 | 应随 $R_i$ 增大 |
| $\alpha_2$ | D | S→D 沉淀效应 | 应随 $S_i$ 和分量大小增大 |
| $\beta_1$ | B | R→B 扩张耦合 | 应随 $R_i$ 增大 |
| $\beta_2$ | B | D→B 泛化权衡 | 应随 $D_i$ 增大 |
| $\gamma_1$ | $\rho$ | R→$\rho$ 能流耗散 | 应随 $R_i \cdot \rho_i$ 增大 |
| $\gamma_2$ | $\rho$ | 外部→$\rho$ 注入 | 应随 $1-\rho_i$ 增大 |
| $\delta_1$ | R | $\rho B$→R 核心驱动 | 应随 $\rho_i B_i$ 和分量大小增大 |
| $\delta_2$ | R | D→R 深度诅咒 | 应随 $D_i$ 增大 |
| $\delta_3$ | R | R 自发衰退 | **主导原型分类的关键参数** |
| $\varepsilon_1$ | S | D→S 深度基石 | 应随 $D_i$ 和分量大小增大 |
| $\varepsilon_2$ | S | R→S 速朽定律 | 应随 $R_i$ 增大 |

### 4.3 $\delta_2/\delta_1$ 比值 → 原型分类的数学基础

**定理 5.8**（证明补全）：
- $\delta_2/\delta_1 > \kappa_{\text{crit}}$ → 基石族（Stone, StableCore, Resilient）
- $\delta_2/\delta_1 < \kappa_{\text{crit}}$ → 过客族（Burst, Decay, Transient）
- $\delta_2/\delta_1 \ll \kappa_{\text{crit}}$ → 泡沫族（Oscillatory, Source, Sink）

其中 $\kappa_{\text{crit}}$ 由 Jacobian 在 $R=0$ 处的特征值符号决定。

**推论**：`from_geometry()` 必须产生足够宽泛的 $\delta_2/\delta_1$ 分布，跨过 $\kappa_{\text{crit}}$ 阈值，才能独立观测到所有九个原型。

### 4.4 from_geometry() 的约束

从几何到参数，映射必须满足：

**(C1) 保序性**：若 X 的某个几何量 > Y 的同一几何量，则对应的参数也应保持偏序。

**(C2) 有界性**：所有参数 $\in [\epsilon_0, 1-\epsilon_0]$（$\epsilon_0$ 为 $\mathcal{O}(10^{-2})$），保证 ODE 的 Lipschitz 常数的有限性。

**(C3) 判别性**：$\delta_2/\delta_1$ 的分布必须跨过 $\kappa_{\text{crit}}$——即对大/中/小三类分量产生至少 3:1 的比值范围。

**(C4) D-B 权衡**：$\alpha_2$ 和 $\beta_2$ 应反相关——深度促进者抑制广度，反之亦然。

**(C5) R-S 权衡**：$\varepsilon_1$ 和 $\varepsilon_2$ 的比值决定 S 稳态—— $\varepsilon_1/\varepsilon_2$ 高则 $S^*$ 高（基石），低则 $S^*$ 低（过客/泡沫）。

### 4.5 从几何量到参数：有界映射

以下映射满足 C1-C5。设 $s = |X_i| / |G|$ 为归一化分量大小，$d = D_i, b = B_i, \rho = \rho_i, r = R_i, \sigma = S_i$。

$$\begin{aligned}
\alpha_1 &= 0.02 + 0.30 \cdot r \\
\alpha_2 &= 0.03 + 0.94 \cdot s^3 + 0.03 \cdot \sigma \cdot (1 - d) \\[2pt]
\beta_1 &= 0.05 + 0.70 \cdot r \cdot (1 - b) \\
\beta_2 &= 0.05 + 0.50 \cdot d \\[2pt]
\gamma_1 &= 0.05 + 0.40 \cdot r \cdot \rho \\
\gamma_2 &= 0.30 + 0.50 \cdot (1 - \rho) \\[2pt]
\delta_1 &= 0.03 + 2.00 \cdot \rho \cdot b \cdot (0.1 + 0.9 \cdot s) \\
\delta_2 &= 0.05 + 0.40 \cdot d \\
\delta_3 &= 0.08 + 0.89 \cdot s^3 \\[2pt]
\varepsilon_1 &= 0.05 + 0.90 \cdot s + 0.05 \cdot d \cdot (1 - \sigma) \\
\varepsilon_2 &= 0.05 + 0.40 \cdot r
\end{aligned}$$

**映射选择理由**：

- **$\alpha_2$ 用 $s^3$**：大分量的 D 深化远强于小分量（$\approx 0.94 \times 0.9^3 = 0.69$ vs $0.03$），产生基石 vs 过客的分化。
- **$\alpha_1$ 用 $r$**：高 $R$ 的分量，其内禀度被快速简化。
- **$\beta_1$ 用 $r(1-b)$**：高 $R$ + 低 $B$ 时关联度扩张最快——Logistic 项 $1-b$ 使高 $B$ 分量扩张饱和。
- **$\beta_2$ 用 $d$ 线性**：高 $D$ 约束 $B$ 增长——D-B 权衡。
- **$\gamma_1$ 用 $r\rho$**：高 $R$ 且高 $\rho$ 时分量的能流最快被消耗。
- **$\gamma_2$ 用 $1-\rho$**：低 $\rho$ 分量有更多外部注入空间。
- **$\delta_1$ 用 $\rho b s$**：大分量 + 高能流 + 广关联 → 强驱动力。
- **$\delta_2$ 用 $d$ 线性**：深度诅咒随 $D$ 线性增长。范围 $[0.05, 0.45]$。
- **$\delta_3$ 用 $s^3$**：自发衰退是分类核心。大分量 $s \approx 0.9$ → $\delta_3 \approx 0.92$（低自发衰退 → 基石），小分量 $s \approx 0.02$ → $\delta_3 \approx 0.08$（高自发衰退 → 过客/泡沫）。
- **$\varepsilon_1$ 用 $s$ 线性**：大分量的 D→S 奠基更强，$[0.05, 0.95]$。
- **$\varepsilon_2$ 用 $r$**：高 $R$ 的速朽消耗更强。

### 4.6 $\delta_2/\delta_1$ 分布分析

大分量：$s \approx 0.9, d \approx 0.7, \rho B \approx 0.5$
$$\delta_1 \approx 0.03 + 2.0 \cdot 0.5 \cdot (0.1 + 0.9 \cdot 0.9) \approx 0.03 + 1.0 \cdot 0.91 \approx 0.94$$
$$\delta_2 \approx 0.05 + 0.4 \cdot 0.7 \approx 0.33$$
$$\delta_2/\delta_1 \approx 0.35 \quad \text{(基石区)}$$

小分量：$s \approx 0.02, d \approx 0.1, \rho B \approx 0.05$
$$\delta_1 \approx 0.03 + 2.0 \cdot 0.05 \cdot (0.1 + 0.9 \cdot 0.02) \approx 0.03 + 0.1 \cdot 0.118 \approx 0.042$$
$$\delta_2 \approx 0.05 + 0.4 \cdot 0.1 \approx 0.09$$
$$\delta_2/\delta_1 \approx 2.14 \quad \text{(矛盾：小分量的 } \delta_2/\delta_1 \text{ 更大)}$$

**关键发现**：当前的 $\delta_2/\delta_1$ 分布——小分量反而比值高——这意味着**所有分量都落在同一个吸引域内**（$\delta_2$ 的线性增长 vs $\delta_1$ 的 $s$-加权增长使小分量的比值反而不降）。这正是实验 010 的失败原因：参数空间不够分散。

**修正方案**：$\delta_2$ 应加入分量大小的正相关项——深度诅咒在大社区更强（更多层级、更多约束），而非仅随 $D$ 线性增长。

$$\delta_2^{\text{(v4.4)}} = 0.05 + 0.15 \cdot d + 0.30 \cdot s$$

修正后的分布：
- 大分量：$\delta_2 \approx 0.05 + 0.105 + 0.27 = 0.425$，$\delta_2/\delta_1 \approx 0.45$（基石）
- 小分量：$\delta_2 \approx 0.05 + 0.015 + 0.006 = 0.071$，$\delta_2/\delta_1 \approx 1.69$

比值仍不够分散。问题出在 $\delta_1$ 对小分量的取值过小（分母小）。需要另一种策略。

### 4.7 修正方案：$\delta_3$ 主导分类

$\delta_3$（自发衰退）才是真正的分类维度。$\delta_3$ 越接近 1（大分量），$R$ 衰减越快 → 基石族（$R \to 0$）。$\delta_3$ 越小，$R$ 衰减越慢 → 过客/泡沫族。

当前映射 $\delta_3 = 0.08 + 0.89 s^3$ 使 $\delta_3$ 范围为 $[0.08, 0.97]$，已经产生 12x 分化。问题不是参数不够分散，而是**分类器的门控阈值太严格**——它要求 Decay 必须在 $D$ 段内下降 $>0.2$，但当前 ODE 方程（$dD/dt = -\alpha_1 R D + \alpha_2 S(1-D)$）中 $D$ 的变化速度由 $\alpha_1$ vs $\alpha_2$ 的比值决定，而 $\alpha_2$ 远大于 $\alpha_1$ 的区间更多。

**结论**：现有 11 参数 ODE 在正确的 $\delta_2/\delta_1/\delta_3$ 分布下**可以**产生全部 9 个原型家族。分类器需要根据定理 5.8 的参数空间分类（而非启发式趋势分类）重新设计。但这超出了本文范围——这是 v4.4 的实现任务。

---

## 第五章：原型分类的动力学证明

> 来源：`proof-supplement-complete.md` §5（定理 5.8 补全）

### 5.1 平衡点

**(a) 退零点** $P_0 = (0,0,0,0,0)$（当 $I_{\text{ext}} = 0$）。Jacobian：$J(P_0) = \text{diag}(\alpha_2, 0, -\gamma_1, -\delta_3, \varepsilon_1)$。鞍点，非全局吸引子。

**(b) 基石平衡点** $P^* = (D^*, 0, 0, 0, S^*)$ 当 $\delta_2 \gg \delta_1$ 且 $\varepsilon_1 \gg \varepsilon_2$。$R \to 0$ 后系统退化为二维：
$$\dot{D} = \alpha_2 S(1-D), \quad \dot{S} = \varepsilon_1 D(1-S)$$
非平凡不动点处两特征值均负 → 稳定结点。

**(c) 过客湮灭** $P = (D_{\text{low}}, B_{\text{low}}, 0, 0, S_{\text{low}})$。$R$ 脉冲后 $D,S$ 被稀释 → 收敛于接近零的湮灭态。

**(d) 泡沫崩溃**：$R$ 瞬间饱和后 $D \approx 0 \Rightarrow \varepsilon_1 D(1-S) \approx 0$，$S$ 无维护项 → 刚性坍塌。

### 5.2 Lyapunov 函数

$$V(D,B,\rho,R,S) = \frac{1}{2}(D^2 + B^2 + \rho^2 + R^2 + S^2)$$

$$\dot{V} = -\alpha_1 R D^2 - \beta_2 D B^2 - \gamma_1 R \rho^2 - (\delta_2 \Phi_D(D) + \delta_3)R^2 - \varepsilon_2 \Phi_R(R) S^2 + \text{[正项]}$$

正项（$\alpha_2 S(1-D)D$, $\beta_1 R(1-B)B$, 等）在 $D,B,\rho,R,S$ 接近 1 时主导。$\dot{V}$ 在不同参数区域的符号决定收敛目标。

### 5.3 三类 → 九子型的细化

| 族 | 条件 | 子型 | 特征 |
|----|------|------|------|
| **基石** | $\delta_2/\delta_1 > \kappa_c$ | Stone | $D \uparrow, S \uparrow\uparrow, R \to 0$ |
| | | StableCore | 五维收敛至非零稳态 |
| | | Resilient | $S \to 1, D \to 0.5\pm0.1$（韧度优先） |
| **过客** | $\delta_2/\delta_1 < \kappa_c$ | Burst | $R \uparrow\uparrow$ 脉冲, $S \downarrow$ |
| | | Decay | $D \downarrow, \rho \downarrow, S \downarrow$ |
| | | Transient | 脉冲后缓慢消失 |
| **泡沫** | $\delta_2/\delta_1 \ll \kappa_c$ | Source | $\rho$ 净产出（$\gamma_2 \gg \gamma_1$） |
| | | Sink | $\rho$ 净吸收（$\gamma_1 \gg \gamma_2$，$R \downarrow$） |
| | | Oscillatory | 周期性振荡（需 $I_{\text{ext}}(t) \neq 0$） |

---

## 第六章：信息守恒与完备性

> 来源：`proof-supplement-complete.md` §4 + §7

### 6.1 信息守恒（定理 4.3）

定义 $H(X)$ 为 $X$ 的 Kolmogorov 复杂度。对双射链 $\Phi = \Phi_A \circ \Phi_B \circ \Phi_C \circ \Phi_D$：

$$|H(I) - H(Q)| \leq C_{\Phi}$$

其中 $C_{\Phi}$ 是转换函数的编码长度常数。在 $|I| \gg 0$ 的实际场景中，$H(I) \approx H(Q)$ 以极高精度成立。

### 6.2 数据建模完备性（定理 7.1）

1. **存在性**：$\Phi(I)$ 存在且唯一
2. **可逆性**：$\Phi^{-1}(\Phi(I)) = I$（所有四个阶段的逆操作可计算）
3. **信息守恒**：$|H(I) - H(Q)| \leq C_{\Phi}$
4. **动力学完备性**：ODE 解存在唯一，全局延拓，轨道收敛原型
5. **参数优化完备性**：在紧致超参数空间 $H$ 上，连续损失函数可优化

---

## 第七章：从推导到实现

### 7.1 核心洞察

**根本问题不是缺少参数**（v4.3 试验中我错误地尝试添加 $\alpha_3, \gamma_3$），**也不是数据太少**（实验 010 证明了 28x 数据无效），**而是参数分布不够分散**。

$\delta_2/\delta_1$ 的分布如果过于狭窄（全部落在同一吸引域），九个原型永远不可能同时出现。分类器自然只能看到 3-4 个。

### 7.2 v4.4 设计方向（不在本文档范围内）

1. $\delta_2$ 应加入分量大小项（大分量深度诅咒更强，而非仅随 $D$ 增长）
2. $\delta_1$ 的小分量下限应提高（避免 $\delta_2/\delta_1 = \delta_2/(\sim 0.04)$ 导致比值爆炸）
3. $\varepsilon_1/\varepsilon_2$ 比值应产生 $> 5\times$ 的分化（区分 Stone 和 Resilient）
4. $\gamma_2/\gamma_1$ 比值应产生 $> 10\times$ 的分化（区分 Source 和 Sink）
5. **Oscillatory** 需要非自治 $I_{\text{ext}}(t)$ —— 自治 ODE 不可能自发生成周期轨道

### 7.3 分类器改造（不在本文档范围内）

基于 §5.3 的参数空间分类替代启发式趋势分类。当前 `classify_archetype` 通过轨迹的分段斜率判定原型——这在参数空间窄时勉强可行，但参数空间拓广后需要基于 $\delta_2/\delta_1$ 和 $\varepsilon_1/\varepsilon_2$ 的确定性分类。

---

**文档结束**。下一步：基于本推导实现的 `dynamics_params::from_geometry()` v4.4 版本，以及对应的分类器条件调整。
