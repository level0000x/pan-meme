# 通用桥：离散组合 ⇄ 连续动力学的连接组件

**日期**：2026-07-08  
**状态**：设计文档 — 待实现

---

## 〇、问题分析

### 问题 A：离散→连续丢失组合爆炸信息

$n$ 个顶点的图有 $2^{\binom{n}{2}}$ 种可能的边配置。CW 复形 G 有 $|K^0|$ 个顶点（通常 50-300），它所在的离散集合 $\mathcal{G}$ 是**组合爆炸**的。但 $\pi: G \to [0,1]^5$ 将其压缩为 5 个实数——$\mathbb{R}^5$ 的拓扑维度是 5，而 $\mathcal{G}$ 的信息量是 $O(n^2)$ 比特。

**桥要做的**：不丢失离散结构 → 保留信息到一个"足够的"中间表示 → 从那里平滑过渡到连续参数。

### 问题 B：拓扑不变量进微分方程：量纲灾难

Betti 数 $\beta_0, \beta_1$ 是**纯整数**——无量纲、无尺度。ODE 参数如 $\delta_1$ 需要在 $[0.03, 0.95]$ 且有特定分布——必须跨过 $\kappa_{\text{crit}}$ 才能产生三种原型。

从整数到 $[0,1]$ 中的实数——映射是**任意的**。没有"自然单位"将拓扑连接到动力学。

**桥要做的**：提供一个自然尺度 $t$（或等价地，一个自然参数化），使拓扑量在其上映射到动力学参数时有唯一确定的标度。

---

## 一、核心思路：热核谱桥

> 灵感来源：图谱理论（spectral graph theory）+ 热核方法（heat kernel methods）+ Morse 理论中的 Witten 形变

**核心算子**：CW 复形 G 上的**图拉普拉斯**

$$L = D - A$$

其中 $D = \text{diag}(\deg(v_1), \ldots, \deg(v_n))$，$A$ 为邻接矩阵。$L$ 是对称半正定矩阵，特征值分解：

$$L = \sum_{k=0}^{n-1} \lambda_k \cdot \mathbf{u}_k \mathbf{u}_k^\top$$

$$0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{n-1}$$

### 为什么是拉普拉斯谱

| 性质 | 桥接的对应 |
|------|-----------|
| $\lambda_0 = 0$ | 连通性 → $\beta_0$（Betti 不变量） |
| $\lambda_1$(谱隙/Fiedler 值) | 最弱连接强度 → "结构韧度"的自然度量 |
| $\lambda_{n-1}$(谱半径) | 最大连接密度 → 关联度上限 |
| $\text{Tr}(L) = \sum \lambda_k = 2|E|$ | 总边数 → 组合量的连续编码 |
| $|\{\lambda_k \approx 0\}|$ | 近零特征值数量 → 近连通分量数 → 模因数 |
| 特征向量 $\mathbf{u}_1$(Fiedler 向量) | 图的"自然切割" → 社区边界 → Louvain 替代方案 |

**谱保留了离散结构的全部信息**（除 cospectral 同构外），且是**规范表示**（不依赖顶点排序）。

### 热核：从离散到连续的自然插值

热核定义为：

$$H_t = e^{-tL} = \sum_{k=0}^{n-1} e^{-t\lambda_k} \cdot \mathbf{u}_k \mathbf{u}_k^\top$$

其中 $t \in (0, \infty)$ 是**时间参数**——桥的自然尺度。

**关键观察**：$t$ 作为尺度参数，其不同取值暴露了图的不同结构层次：

| $t$ 范围 | 暴露的结构 | 对应的动力学参数族 |
|----------|-----------|-------------------|
| $t \to 0$ | 局部组合结构：度分布、密度 | D(beta), B(beta) |
| $t \approx 1/\lambda_1$ | 中观：社区结构、弱连接 | ρ, R 参数 |
| $t \approx 1/\lambda_{n-1}$ | 全局：完全混合 | S 稳态 |
| $t \to \infty$ | 拓扑不变量：$\beta_0$, $\chi$ | δ₃, δ₂/δ₁ |

**热迹（heat trace）**：$\Theta(t) = \text{Tr}(H_t) = \sum_{k=0}^{n-1} e^{-t \lambda_k}$

$\Theta(t)$ 在 $t \to 0$ 时捕获局部组合量，在 $t \to \infty$ 时退化为 $\beta_0$。它是从离散到连续的光滑插值。

---

## 二、桥的架构：三个阶段

```
G (CW复形, 离散)
    │
    ▼ [阶段 0: 谱提取]
Spectrum(G) = {λ₀, λ₁, ..., λ_{n-1}} × [n × n 特征向量矩阵]
    │
    ▼ [阶段 1: 多尺度热迹分析]
Θ(t) = Tr(e^{-tL})   t ∈ (0, ∞)
ζ_G(s) = Σ λ_k^{-s}  谱 zeta 函数
    │
    ▼ [阶段 2: 标度参数提取]
ScaleMap: (Θ, ζ) → {α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂, δ₃, ε₁, ε₂}
    │
    ▼
Params ∈ R¹¹ → ODE → 原型
```

### 阶段 0：谱提取（确定性压缩）

**输入**：CW 复形 G，加权邻接矩阵

**输出**：特征值谱 $\{\lambda_k\}_{k=0}^{n-1}$ + 选定的特征向量片段

**信息保留**：
- 从 $2^{\binom{n}{2}}$ 种配置 → $n$ 个实数（谱）。压缩比 $O(2^{n^2}/n)$。
- 等价类：cospectral 图映射到相同谱。但"cospectral ≠ 同构"是轻微的、已知的歧义。对泛模因建模而言，两个非平凡的 cospectral 非-同构图如果在谱层面相同，它们的宏观动力学行为应当相同（这正是我们要的）。

**代码替换**：当前 `compute_invariants()` 只计算 $\beta_0, \beta_1, \chi$（O(n)）。替换为 `compute_spectrum()`（O(n³) 但 n ≈ 100-300 可接受）。

### 阶段 1：多尺度分析

**输入**：$\{\lambda_k\}$

**核心函数**：

**(a) 热迹** $\Theta(t) = \sum_{k=0}^{n-1} e^{-t\lambda_k}$

$$\text{局部度: } \lim_{t \to 0} \Theta(t) = n$$
$$\text{全局连通性: } \lim_{t \to \infty} \Theta(t) = \beta_0$$

**(b) 谱 zeta 函数** $\zeta_G(s) = \sum_{\lambda_k > 0} \lambda_k^{-s}$

$$\text{拓扑信息: } \lim_{s \to 0} \zeta_G(s) = n - \beta_0$$

**(c) 特征标度** $t^*$：满足 $\Theta(t^*) = \beta_0 + 1$ 的最小 $t$

这个 $t^*$ 是图的"混合时间"——信息在图中均匀化所需的时间。它是**天然的单位尺度**，直接连接离散和连续。

### 阶段 2：标度参数映射（解决量纲灾难）

**核心思想**：ODE 参数的"正确"取值由图的**谱标度**决定，不由手工系数决定。

定义归一化时间 $\tau = t / t^*$。

**参数映射规则**（全部由谱导出，零手工系数）：

#### α 族（D 方程）

$$\alpha_1 = \frac{\lambda_1}{2|E|} \quad \text{（谱隙/总边 = 每边的信息稀释率）}$$
$$\alpha_2 = \frac{\Theta(0.5 \cdot t^*)}{n} \quad \text{（半混合时间的保留比例 → 沉淀效率）}$$

**物理含义**：$\alpha_1$ 是弱连接相对于总连接的比例——弱连接越少，深度稀释越慢。$\alpha_2$ 是一半信息扩散完成时仍保留在局域的结构比例。

#### β 族（B 方程）

$$\beta_1 = \frac{\lambda_{n-1} - \lambda_1}{\lambda_{n-1}} \quad \text{（谱展宽 = 连接不均匀度 → 扩张潜力）}$$
$$\beta_2 = \frac{\sum_{k=0}^{n-1} \lambda_k^2}{(\sum \lambda_k)^2} \quad \text{（第二矩/第一矩² = 度分布的集中度 → 泛化壁垒）}$$

#### γ 族（ρ 方程）

$$\gamma_1 = 1 - \frac{\Theta(2 \cdot t^*)}{\Theta(t^*)} \quad \text{（从混合时间到双倍混合时间的迹衰减 = 能流耗散率）}$$
$$\gamma_2 = \frac{\Theta(t^*/2)}{\Theta(t^*)} - 1 \quad \text{（从半混合到混合的迹增长 = 外部注入敏感度）}$$

#### δ 族（R 方程）—— 最关键

$$\delta_1 = \frac{\Theta(t^*/2) - \Theta(t^*)}{n} \quad \text{（半混合到混合的信息流动量 = 核心驱动力）}$$
$$\delta_2 = \frac{\lambda_{n-1}}{\sum_{k>0} e^{-\lambda_k t^*}} \quad \text{（谱半径/持久热迹 = 深度诅咒强度）}$$
$$\delta_3 = 1 - \frac{\Theta(3 \cdot t^*)}{\Theta(2 \cdot t^*)} \quad \text{（长尾衰减率 = 自发衰退率）}$$

**关键**：$\delta_3$ 现在由热迹的长尾行为**唯一**确定——不依赖 $s^3$ 或任何手工映射。

#### ε 族（S 方程）

$$\varepsilon_1 = \frac{\lambda_1}{\lambda_{n-1}} \quad \text{（谱隙/谱半径 = 韧度-深度耦合强度）}$$
$$\varepsilon_2 = 1 - \frac{\Theta(2 \cdot t^*)}{n} \quad \text{（长时迹/顶点数 = 结构消解率）}$$

**零手工系数：** 所有参数由 $\{\lambda_k\}$ 和 $t^*$ 解析确定。唯一的自由量是热核的**评估点**（$t^*/2, t^*, 2t^*, 3t^*$）——这些点由图的固有标度决定，不引入经验参数。

---

## 三、谱桥如何修复已知断裂

| 断裂 | 旧方案的失败 | 谱桥的修复 |
|------|-------------|-----------|
| **B6: from_geometry 手工映射** | 手工系数 + $\delta_2/\delta_1$ 反设计 | 谱导出的 $\delta$ 族满足 $\delta_2/\delta_1$ 自然随图密度单调分化 |
| **B5: 5D 映射手工权重** | $0.5, 0.6, 0.4, 0.3, 0.7$ 经验系数 | π 映射的五维可由谱矩直接计算（$D \sim \lambda_1$, $B \sim \lambda_{n-1}$, $\rho \sim -\Theta'(0)$） |
| **B4: Louvain 替代 Betti** | $\gamma$ 分辨率参数无理论指导 | 谱聚类的特征向量间隙 $\lambda_{k+1} - \lambda_k$ 给出"自然社区数"——不用 γ |
| **B1: 词序丢失** | Tokenizer 的有损映射不可逆 | 若 FCA 输入采用 n-gram 邻接图（而非词袋），加密度的二部图转变为加权时序图——谱可捕获时序 |

### 自然社区数（替代 Louvain 的 γ）

图拉普拉斯的**特征值间隙** $\lambda_{k+1} - \lambda_k$ 最大化时,对应的 $k$ 是图的"自然簇数"（谱聚类理论的经典结论——Gap Statistic）。不需要超参 γ。

$$k^* = \arg\max_k (\lambda_{k+1} - \lambda_k)$$

对字-词二部图，$k^*$ 给出"不需要人工设定的"社区数。

### 五维谱矩映射（替代手工 5D 映射）

$$\begin{aligned}
D_i^{\text{(spec)}} &= 1 - \exp(-\lambda_1^{(i)} \cdot |X_i| / |G|) \quad \text{（谱隙加权深度）} \\
B_i^{\text{(spec)}} &= 1 - \exp(-\lambda_{n-1}^{(i)} / \lambda_{n-1}^{(G)}) \quad \text{（谱半径比 → 广度）} \\
\rho_i^{\text{(spec)}} &= \left.\frac{d\Theta_i}{dt}\right|_{t=0^+} \Big/ \left.\frac{d\Theta_G}{dt}\right|_{t=0^+} \quad \text{（热迹初始斜率比 → 能流密度）} \\
R_i^{\text{(spec)}} &= \frac{t^*_G}{t^*_i} \quad \text{（全局/局域混合时间比 → 演化速率）} \\
S_i^{\text{(spec)}} &= \frac{1}{1 + \lambda_1^{(i)}} \quad \text{（谱隙降低 → 韧度增强）}
\end{aligned}$$

---

## 四、量纲一致性验证

原方案：Betti 数 $\beta_1 \in \mathbb{Z}_{\ge 0}$ → $\delta_3 \in [0.08, 0.97]$ → ODE → $\dot{R} \propto -\delta_3 R$

问题：整数到实数的映射没有"单位"——为什么 Betti=3 和 Betti=5 的 $\delta_3$ 差异应该是 X？

谱桥：

$$\delta_3 = 1 - \frac{\Theta(3t^*)}{\Theta(2t^*)}$$

- $\Theta(t)$ 的单位是"顶点数"（无量纲计数，但带尺度 $t$）
- $\delta_3$ 是比例比——本身无量纲，但由 $t^*$ 的尺度自然确定
- 不同图的 $\delta_3$ 差异由各自的混合时间 $t^*$ 自然调节——一个大图的 $t^*$ 与小图的 $t^*$ 不同，$\delta_3$ 自然不同

**不需要手工指定"大 vs 小"**——尺度由图的拉普拉斯谱的内在性质决定。

---

## 五、可行性分析

### 计算复杂度

| 操作 | 复杂度 | n=200 估计 |
|------|--------|------------|
| 拉普拉斯特征分解 | $O(n^3)$ | ~8M 操作，< 0.1s（LAPACK） |
| 热迹求值（10 个 t 点） | $O(n)$ | 可忽略 |
| 谱 zeta 求值 | $O(n)$ | 可忽略 |

总开销 < 0.5s/图。当前实验 008 处理 30 个图，< 15s 额外开销。完全可接受。

### 依赖

需要添加 `nalgebra` 或 `faer`（Rust 线性代数库）做特征值分解。当前 updown_rs 是零外部依赖的——但添加一个线性代数库用于谱分解是合理的（LAPACK 级别的依赖，不是业务逻辑依赖）。

替代方案：对于稀疏图（字-词二部图通常是稀疏的），使用 Lanczos 迭代只求前 k 个特征值，$O(k \cdot |E|) \ll O(n^3)$。

---

## 六、与当前代码的接口

新增模块 `updown_rs/src/theory/spectral_bridge.rs`：

```rust
/// 拉普拉斯谱 — 离散-连续桥的核心数据结构
pub struct LaplacianSpectrum {
    pub eigenvalues: Vec<f64>,       // λ₀, λ₁, ..., λ_{n-1} (升序)
    pub n_vertices: usize,
    pub total_edges: usize,
}

impl LaplacianSpectrum {
    /// 从 CW 复形的邻接矩阵计算
    pub fn from_cw_complex(g: &CWComplex) -> Self { ... }

    /// 热迹 Θ(t)
    pub fn heat_trace(&self, t: f64) -> f64 {
        self.eigenvalues.iter().map(|&l| (-t * l).exp()).sum()
    }

    /// 混合时间 t*: 最小 t 使 Θ(t) = β₀ + 1
    pub fn mixing_time(&self) -> f64 { ... }

    /// 自然社区数 (最大特征值间隙法)
    pub fn natural_community_count(&self) -> usize { ... }
}
```

新增方法 `DynamicsParams::from_spectrum(spectrum: &LaplacianSpectrum) -> Self`。

---

## 七、总结

| 问题 | 当前状态 | 桥方案 |
|------|---------|--------|
| 离散→连续信息丢失 | $2^{O(n^2)}$ → $\mathbb{R}^5$，丢失 > 99.9% | 谱作为规范压缩，保 $n$ 维 → $\mathbb{R}^{11}$ 有理论保证 |
| 量纲灾难 | Betti 整数 → 无标度实数映射 | $t^*$ 作为天然标度，所有参数由 $\Theta(t)$ 的比值定义（单位消去） |
| 手工系数 | 0.5, 0.94, 0.89, ... 全部任意的 | **零手工系数** — 全部分母/分子使单位消去 |
| Louvain 的 γ | 自由超参 | $k^* = \arg\max_k \Delta\lambda_k$ 天然确定 |
| 5D 映射 | 5 维加权平均 | 谱矩导出 — 每个维度有明确的谱解释 |

**实现路线**：
1. 新增 `nalgebra` 依赖 → 拉普拉斯特征值分解
2. 实现 `LaplacianSpectrum` 与热迹/混合时间计算
3. 重写 `from_geometry()` → `from_spectrum()`
4. 重写 5D 映射 → 谱矩版本
5. 运行实验 008+009+010 验证
