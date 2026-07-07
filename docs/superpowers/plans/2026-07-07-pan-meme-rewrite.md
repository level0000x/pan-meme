# Pan-Meme 完整重写计划 v4

> **阅读覆盖：** 泛模因理论.md（2850行全七章+附录A-E）、formal-concept-analysis-proof.md（247行，FCA证明全篇）、proof-supplement-complete.md（349行，Φ_B/Φ_C双射+信息守恒+ODE推导+Θ紧致化全篇）、ROSE-PAN-MEME-迁移规划.md、PB_ARCHITECTURE.md、TASKS.md（关键部分）。

**Goal:** 从数学构造出发，完整重写泛模因理论四阶段管线，使代码的每一步都忠实对应论文中的数学定义。

**Architecture:** 保持四阶段管线不变（浮现→编码→分解→固化），每阶段用论文的数学构造替换当前的手工近似。Phase 1 用 FCA ↑↓ 伽罗瓦连接 + Warshall 传递闭包 + 推理与补全 + 规则约束推导，输出 M=(S,F,C)；Phase 2 用概念格导出 CW 复形，包含 K/g/ω/Γ/R 五元组；Phase 3 用 Louvain 社区检测 + 扩展维度 ξ（偏差于论文 β₀ 分解）；Phase 4 保持 SHA-256 凭证（含完整 meme_state）；Phase 5 ODE 支持跳变点检测 + Π 投影映射 + 脉冲 I_ext。

**Tech Stack:** Rust 2021 edition，零外部依赖（sha2 可选），所有数学构造用纯 Rust 实现。

**设计原则：**
1. 每个函数注释标注对应论文章节号（如 `// §3.2.1 定义 3.1`）
2. 每个结构体标注对应数学对象（如 `/// 对应论文中的 Ψ = (V, E, w)`）
3. 每个关键步骤有对应的可逆操作（`decode_*` 函数 + 可逆性元数据 R）
4. 所有浮点操作有 NaN/∞ 防护
5. 每个模块有独立单元测试 + 完整端到端集成测试

---

## 代码组织

```
updown_rs/src/
├── lib.rs
├── main.rs
├── theory/                     # 核心数学构造（13 模块）
│   ├── mod.rs
│   ├── fca.rs                  # 形式概念分析：↑↓ 伽罗瓦连接（§D.1-D.3, 定义 3.1-3.2）
│   ├── cw_complex.rs           # CW 胞腔复形（§3.3.1, 定义 2.4）
│   ├── vector_field.rs         # 离散梯度场 F = -∇φ（§3.3.3）
│   ├── louvain.rs              # Louvain 社区检测（§3.2.4）
│   ├── five_dim.rs             # 五维状态 D/B/ρ/R/S（§3.4.2, §4.1）
│   ├── extended_dimension.rs   # 扩展维度 ξ（§6.4.3, 前提 5）
│   ├── function_families.rs    # 五族函数（附录 B, §4.3.8）
│   ├── dynamics_params.rs      # 11 参数闭式解（§3.4.3, 附录 A.3）
│   ├── coupling.rs             # 耦合矩阵 C（§6.4.4）
│   ├── ode.rs                  # 5D ODE + RKF45 + 原型分类（§4.3-4.5, 定理 5.8）
│   ├── jump_handler.rs         # 跳变点处理 + Π 投影映射（定理 6）
│   └── optimizer.rs            # 假设0 全局优化 H=T×F×Θ×N（§4.3.8, §D.5）
├── pipeline/                   # 管线阶段（5 模块）
│   ├── mod.rs
│   ├── phase1_emergence.rs     # Phase 1: 浮现（§3.2, M1）
│   ├── phase2_encoding.rs      # Phase 2: 编码（§3.3, M1）
│   ├── phase3_decomposition.rs # Phase 3: 分解（§3.4, M2）
│   ├── phase4_binding.rs       # Phase 4: 固化（§3.5）
│   └── phase5_evolution.rs     # Phase 5: 演化（§4.3-4.4, M2）
└── io/                         # 输入输出（3 模块）
    ├── mod.rs
    ├── tokenizer.rs            # 自然语言 → n-gram 词表
    ├── tsv.rs                  # WikiLine TSV 输出
    └── reversibility.rs        # Φ⁻¹∘Φ 往返验证（推论 5.1）
```

---

## 论文-代码 映射表

### 数学定义 → Rust 结构体

| 论文符号 | 定义 | Rust 结构体 | 文件 |
|---------|------|-----------|------|
| Ψ = (V, E, w) | 关系网络 | `FormalContext` | `theory/fca.rs` |
| K = (K⁰, K¹, K²) | CW 胞腔复形 | `CWComplex` | `theory/cw_complex.rs` |
| φ: K⁰ → ℝ | 标量场 | `ScalarField` | `theory/vector_field.rs` |
| F = -∇φ | 向量场 | `VectorField` | `theory/vector_field.rs` |
| M = (D,B,ρ,R,S) | 五维状态 | `FiveDimState` | `theory/five_dim.rs` |
| ξ_i | 扩展维度 | `ExtendedDimension` | `theory/extended_dimension.rs` |
| C_{ij} | 耦合矩阵 | `build_coupling_matrix` | `theory/coupling.rs` |
| Θ = {α₁,…,ε₂} | 11 参数 | `DynamicsParams` | `theory/dynamics_params.rs` |
| Φ_D, Φ_R | 五族函数 | `FunctionFamily` + `FamilyParams` | `theory/function_families.rs` |
| Π | 投影映射 | `JumpHandler` | `theory/jump_handler.rs` |
| H = T×F×Θ×N | 假设 0 空间 | `OptimizerConfig` + `optimize` | `theory/optimizer.rs` |

### 论文定理 → 代码验证

| 定理 | 内容 | 验证方法 |
|------|------|---------|
| 定理 4.1-4.7 | ↑↓ 有限收敛 | `CycleEngine::cycle_all` 收敛检测 |
| 定理 5 | 四阶段双射 | `io/reversibility.rs` 往返验证 |
| 推论 5.1 | Φ⁻¹(Φ(I)) = I | 词/字完全匹配检查 |
| 定理 6 | 分段解存在唯一性 | `jump_handler.rs` 跳变检测 |
| 定理 7 | Ω 不变性 | 每步裁剪到 [0,1]⁴×[0,∞) |
| 定理 8 | 5 种平衡点 | `ode.rs` 平衡点分类 |
| 定理 5.8 | 原型分类 | `ode.rs` 九类原型判定 |
| 定理 9 | 信息守恒 | 往返 + 双射验证 |
| 定理 6.2 | H 紧致性 | 有限网格搜索保证 |

---

## 五族函数（附录 B）

> 论文 §4.3.8 明确列出五族：幂函数、指数函数、Sigmoid、对数函数、分段线性。

| 函数族 | 解析形式 | k 范围 | 适用场景 |
|--------|---------|--------|---------|
| Power | $x^k$ | k ∈ [0, 2] | 抑制/消耗呈幂律增长 |
| Exponential | $e^{kx} - 1$ | k ∈ [0, 2] | 在阈值附近急剧上升 |
| Sigmoid | $\frac{1}{1+e^{-k(x-x_0)}} - \frac{1}{2}$ | k ∈ [0.5, 2.5], x₀ ∈ [0, 1] | 存在明显临界点 |
| Logarithm | $\ln(1 + kx)$ | k ∈ [0, 2] | 效应随自变量增长递减 |
| PiecewiseLinear | 数据驱动的分段线性插值 | b₁,b₂ ∈ [0,1], b₁<b₂ | 复杂关系，无单一函数可描述 |

**注：** 参数范围与 proof-supplement-complete §6.2 的 Θ 紧致化定义对齐。论文正文 §4.3.8 的 k 范围（如 Sigmoid k∈[5,20]）是使用建议区间，supplement 的紧致化范围（[0.5, 2.5]）是数学证明中保证 Lipschitz 连续性的理论区间。代码使用 supplement 的紧致化范围。

---

## 五维 ODE 方程系统（§4.3.1）

$$\begin{aligned}
\frac{dD}{dt} &= -\alpha_1 R D + \alpha_2 S(1-D) \\
\frac{dB}{dt} &= \beta_1 R(1-B) - \beta_2 D B \\
\frac{d\rho}{dt} &= -\gamma_1 R \rho + \gamma_2 (1-\rho) \cdot I_{\text{ext}}(t) \\
\frac{dR}{dt} &= \delta_1 \rho B (1-R) - \delta_2 \Phi_D(D) R - \delta_3 R \\
\frac{dS}{dt} &= \epsilon_1 D (1-S) - \epsilon_2 \Phi_R(R) S
\end{aligned}$$

其中 $I_{\text{ext}}(t)$ 是脉冲或衰减函数（有界，默认 $I_{\text{ext}}(t) = e^{-t}$），$\Phi_D$ 和 $\Phi_R$ 由假设 0 全局优化从五族函数中选出。

---

## 11 参数汇总（§4.3.7）

| 参数 | 名称 | 含义 | 所在方程 | 来源 |
|------|------|------|---------|------|
| α₁ | 简化效应 | R 对 D 的稀释速率 | dD/dt | 边密度 2|E|/(|V|(|V|-1)) |
| α₂ | 沉淀效应 | S 对 D 的深化速率 | dD/dt | 层级深度 depth/d_max |
| β₁ | 扩张耦合 | R 对 B 的扩张效率 | dB/dt | 外部连接数（论文 §3.4.3） |
| β₂ | 泛化权衡 | D 对 B 的抑制强度 | dB/dt | 内部复杂度/外部连接数（论文 §3.4.3） |
| γ₁ | 能流耗散 | R 对 ρ 的消耗速率 | dρ/dt | 场散度 mean(|∇φ|) |
| γ₂ | 外部赋能 | I_ext 对 ρ 的注入效率 | dρ/dt | 边界流入通量（论文 §3.4.3） |
| δ₁ | 核心驱动力 | ρ·B 对 R 的促进效率 | dR/dt | 曲率 × 连接密度 |
| δ₂ | 深度诅咒 | Φ_D(D) 对 R 的抑制强度 | dR/dt | 层级深度 × 内部复杂度 |
| δ₃ | 自发衰退 | R 的自然衰减率 | dR/dt | 拓扑不变量数量倒数 1/(1+|Γ|) |
| ε₁ | 深度基石 | D 对 S 的奠基效率 | dS/dt | 层级深度 × 稳定性指标 |
| ε₂ | 速朽定律 | Φ_R(R) 对 S 的消耗强度 | dS/dt | 曲率变化率 × 场散度 |

**注：** 以上为论文 §3.4.3 给出的几何推导方向。plan 中部分参数的具体公式为简化近似（如 β₁ 用 0.5 默认值），后续可迭代精确化为论文的几何推导。

---

## 原型分类（定理 5.8 + 定理 8）

### 5 种平衡点（定理 8）

| 平衡点类型 | 状态 | 稳定性 |
|-----------|------|--------|
| 退化零点 | (0,0,0,0,0) | 鞍点 |
| 湮灭型 | (0, b*, 0, 0, 0) | 中性稳定 |
| 惰性型 | (1, 0, ρ, 0, 1) | 中性稳定 |
| 广度-韧度型 | (0, b, ρ, 0, 0) | 中性稳定 |
| 比例平衡点 | (d*, b*, ρ*, r*, s*) | 依参数 |

### 3 族 × 3 子类 = 9 类原型（定理 5.8）

| 原型族 | 子类 | 条件 | 终态 |
|--------|------|------|------|
| **基石型** | Stone | D↑, S↑ | 高 D, 高 S 稳态 |
| | StableCore | 全维度稳态 | 均衡 |
| | Resilient | S↑↑ 低活动 | 惰性型平衡点 |
| **过客型** | Burst | R↑↑, S↓ | 脉冲后衰减 |
| | Decay | 全维度下跌 | 湮灭型平衡点 |
| | Transient | 脉冲态后消失 | 退化零点 |
| **泡沫型** | Oscillatory | 周期振荡 | 极限环 |
| | Source | ρ 净产出 | 广度-韧度型 |
| | Sink | ρ 净吸收 | 退化零点 |

---

## 扩展维度 ξ（§6.4.3, 前提 5 + proof-supplement-complete 定义 3.3）

> "核心五维驱动动力学，扩展维度编码微观涨落保证双射性"

扩展维度 ξ 保证 Φ⁻¹∘Φ 严格双射。当 n 个模因合并/分裂时（定理 6 跳变），ξ 承载以下信息：

| 字段 | 类型 | 含义 | 来源 |
|------|------|------|------|
| `cell_snapshots_v` | Vec<CellSnapshot> | 每个 0-胞腔（顶点）的快照 | supplement 定义 3.3 V_i |
| `cell_snapshots_e` | Vec<CellSnapshot> | 每条 1-胞腔（边）的快照 | supplement 定义 3.3 E_i |
| `cell_snapshots_f` | Vec<CellSnapshot> | 每个 2-胞腔（面）的快照 | supplement 定义 3.3 F_i |
| `boundary_links` | Vec<(usize, usize, f64)> | 跨子几何体的边界链接 | supplement 定义 3.3 B_i |
| `parent_meme_id` | Option<usize> | 分裂来源（新模因从父模因的 ξ 解码初始化） | 定理 6 |
| `micro_fluctuation` | Vec<f64> | 微观涨落模式（5D 残差向量） | 前提 5 |

**注：** ξ 结构与 supplement 定义 3.3 对齐：`(V_i, E_i, F_i, B_i)` 四元组。`parent_meme_id` 和 `micro_fluctuation` 是在此基础上的扩展，用于跳变处理（定理 6）。

---

## 跳变点处理（定理 6）

> "模因数量 n(t) = β₀(K(t)) 在时间轴上只在有限个跳变点处变化"

### 跳变规则

1. **模因追踪**：新几何中存在与旧几何结构连续的子复形 → 对应模因状态承接跳变：$\vec{M}_j(t_i^+) = \Pi(\vec{M}_j(t_i^-), K_j(t_i^+))$

2. **新模因初始化**：n_i > n_{i-1} → 新增模因从父模因的 ξ 解码初始化

3. **模因消失**：n_i < n_{i-1} → 消失模因状态被吸收进存续模因的 ξ 中

### 投影映射 Π 的实现

```rust
/// 投影映射 Π: 从旧几何到新几何的可逆投影
/// 对应定理 6 的模因追踪规则
fn project_state(
    old_state: &FiveDimState,
    old_vertices: &[usize],
    new_vertices: &[usize],
) -> FiveDimState {
    let overlap = old_vertices.iter().filter(|v| new_vertices.contains(v)).count();
    let retention = overlap as f64 / old_vertices.len().max(1) as f64;
    // 状态按重叠比例衰减
    FiveDimState {
        intrinsic_degree: old_state.intrinsic_degree * retention,
        binding_degree: old_state.binding_degree * retention,
        energy_density: old_state.energy_density * retention,
        evolution_rate: old_state.evolution_rate * retention,
        structural_robustness: old_state.structural_robustness * retention,
    }
}
```

---

## Phase 1 数据流（M1：浮现）— 对应论文 §6.2.2 完整流程

```
原始信息 I
    ↓ [tokenize: n-gram 分词 — 提取器接口契约 formal-concept-analysis-proof 推论 7.2]
Token 序列 + 字-词包含关系
    ↓ [FCA: 形式上下文 Ψ = (W∪C, E)]
关系网络 Ψ
    ↓ [Warshall 传递闭包 — O(|V|³)]
完整关系网 Ψ*（§3.2.2 传递性推理闭合）
    ↓ [↑↓ 循环 → 概念组合 — formal-concept-analysis-proof 定理 4.1-4.7]
多层级结构骨架 + 收敛轮数 = 信息深度
    ↓ [彻底不完整判定 — §3.2.5]
    ↓ [推理与补全：传递性/对称性/共现/模式补全/结构相似性 — §3.2.3]
完整关系集合 R
    ↓ [概念-要素循环组合 — §3.2.4]
多层级结构域 S = (V, E, w)
    ↓ [规则推导 F: 时序/因果/传播/结构衍生 — §3.2.6]
    ↓ [约束推导 C: 结构不变量/边界条件/层级一致性/组合路径完整性 — §3.2.6]
    ↓ [自洽性验证 — §3.2.7]
Phase 1 输出: M = (S, F, C)
```

**关键新增：**
1. **Warshall 传递闭包**：论文 M1 数据流明确要求"传递性推理闭合"。Warshall 算法对关系矩阵 R（|V|×|V| 布尔矩阵），对每个 k，对每对 (i,j)，R[i][j] = R[i][j] ∨ (R[i][k] ∧ R[k][j])。完成后 R 是传递闭包。
2. **推理与补全**：基于初始关系 R₀ 自动推理缺失关系，5 种推理规则（传递性/对称性/共现/模式补全/结构相似性），结果直接补全形成关系闭合闭包。
3. **规则与约束推导**：从关系网络提取动态规则 F（5 类规则，带支撑集）和静态约束 C（4 类约束，带作用域）。这是论文 M = (S, F, C) 三元组的完整输出。
4. **提取器接口契约**：tokenizer 实现 formal-concept-analysis-proof 推论 7.2 的接口 `A_F: RawData → (Nodes, Edges^containment)`，支持多模态扩展（文本/图像/音频/代码/地理/数学证明）。

---

## 可证伪判据（§7.3 M4）

> "若在至少 3 个不同领域的数据集上，经假设 0 全局优化选出的 (Φ_D*, Φ_R*) 呈现为不同函数族，且差异无法由参数差异解释，则核心假设被证伪。"

这是理论的核心验证标准。在代码中体现为：多次运行（不同数据集）后，`optimize()` 返回的 `best.family_d` 和 `best.family_r` 必须收敛到同一族。

---

## 任务分解

### Task 1: 项目骨架搭建

**创建 21 个文件：**

| 目录 | 文件 |
|------|------|
| `theory/` | `mod.rs`, `fca.rs`, `cw_complex.rs`, `vector_field.rs`, `louvain.rs`, `five_dim.rs`, `extended_dimension.rs`, `function_families.rs`, `dynamics_params.rs`, `coupling.rs`, `ode.rs`, `jump_handler.rs`, `optimizer.rs` |
| `pipeline/` | `mod.rs`, `phase1_emergence.rs`, `phase2_encoding.rs`, `phase3_decomposition.rs`, `phase4_binding.rs`, `phase5_evolution.rs` |
| `io/` | `mod.rs`, `tokenizer.rs`, `tsv.rs`, `reversibility.rs` |

- [ ] **Step 1: 创建目录结构**

```powershell
Remove-Item -Recurse -Force updown_rs/src/emergence, updown_rs/src/encoding, updown_rs/src/sealing, updown_rs/src/infra -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path updown_rs/src/theory, updown_rs/src/pipeline, updown_rs/src/io
```

- [ ] **Step 2: 创建所有模块文件骨架，填充完整代码**

每个文件包含：模块级文档注释（论文章节引用）、结构体/函数定义、`#[cfg(test)] mod tests` 单元测试。

- [ ] **Step 3: 更新 lib.rs**

```rust
pub mod theory;
pub mod pipeline;
pub mod io;
```

- [ ] **Step 4: 编译验证**

```bash
cd updown_rs && cargo check --tests 2>&1
```

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: 项目骨架 v3 — theory(13模块)/pipeline(5)/io(3)

theory/: FCA, CW复形, 梯度场, Louvain, 五维+扩展维度ξ,
          五族函数(Power/Exp/Sigmoid/Log/Piecewise),
          11参数, 耦合矩阵, ODE+原型分类, 跳变处理, 假设0优化器
pipeline/: 五阶段管线
io/: tokenizer, TSV, 往返验证"
```

---

### Task 2: theory/fca.rs — 形式概念分析

**对应论文：** §D.1-D.3, 定义 3.1-3.2, 定理 4.1-4.7

**核心结构体：**
- `FormalContext`：Ψ = (W, C, E)，字-词二分图
- `CycleEngine`：↑↓ 循环引擎
- `ElementCycle`：单元素循环结果

**核心函数：**
- `FormalContext::from_words(words)`：C 从 W 自暴露
- `CycleEngine::cycle_char(ctx, char_idx)`：对单字执行 ↑↓ 循环
- `CycleEngine::cycle_all(ctx)`：对所有字并行 ↑↓ 循环
- `warshall_closure(adj_matrix)`：Warshall 传递闭包（O(|V|³)）
- `verify_reversibility(ctx, cycle_results)`：验证 ↓(↑(c)) = c 和 ↑(↓(w)) = w（formal-concept-analysis-proof 公理 3）

**关键细节：**
- 收敛检测：两轮间 word_closure 和 char_closure 都不再增长
- 传递闭包必须在 ↑↓ 循环之前执行（M1 数据流要求）

---

### Task 3: theory/cw_complex.rs + vector_field.rs — CW 复形 + 梯度场

**对应论文：** §3.3.1-3.3.5, 定义 2.4

**核心结构体：**
- `Cell`：单胞腔（id, dim, boundary）
- `CWComplex`：胞腔复形（cells, v0_map, e1_map, Euler, Betti）
- `ScalarField`：φ: K⁰ → ℝ
- `VectorField`：F = -∇φ

**核心函数：**
- `CWComplex::add_vertex / add_edge`
- `compute_gradient(complex, scalar, metric)`：离散梯度

---

### Task 4: theory/louvain.rs — Louvain 社区检测

**对应论文：** §3.2.4

**核心结构体：** `LouvainLayer`（node_to_community, n_communities, modularity_Q）

**核心函数：** `run_louvain(adj: &[Vec<(usize, f64)>]) -> Option<LouvainLayer>`

---

### Task 5: theory/five_dim.rs + extended_dimension.rs — 五维状态 + 扩展维度

**对应论文：** §3.4.2, §4.1, §6.4.3, 前提 5

**核心结构体：**
- `FiveDimState`：D, B, ρ, R, S + is_valid()
- `ExtendedDimension`：cell_snapshots_v, cell_snapshots_e, cell_snapshots_f, boundary_links, parent_meme_id, micro_fluctuation（按 supplement 定义 3.3）

---

### Task 6: theory/function_families.rs — 五族函数

**对应论文：** 附录 B, §4.3.8

**核心结构体：**
- `FunctionFamily` 枚举：Power / Exponential / Sigmoid / Logarithm / PiecewiseLinear
- `FamilyParams`：family, k, x0

**核心函数：** `FamilyParams::evaluate(x) -> f64`

**论文函数族（非 v2 的 Logistic/Threshold）：**

| Family | 公式 | k 范围 | 来源 |
|--------|------|--------|------|
| Power | x^k | [0, 2] | supplement §6.2 |
| Exponential | e^{kx} - 1 | [0, 2] | supplement §6.2 |
| Sigmoid | 1/(1+e^{-k(x-x₀)}) - 1/2 | k∈[0.5, 2.5], x₀∈[0, 1] | supplement §6.2 |
| Logarithm | ln(1+kx) | [0, 2] | supplement §6.2 |
| PiecewiseLinear | 分段线性 | b₁,b₂∈[0,1], b₁<b₂ | supplement §6.2 |

**注：** 参数范围与 proof-supplement-complete §6.2 的 Θ 紧致化定义对齐。论文正文 §4.3.8 的 k 范围（如 Sigmoid k∈[5,20]）是使用建议区间，supplement 的紧致化范围（[0.5, 2.5]）是数学证明中保证 Lipschitz 连续性和紧致性的理论区间。代码实现使用 supplement 的紧致化范围以保证 ODE 右侧函数的 Lipschitz 性质（proof-supplement-complete 引理 5.2）。

---

### Task 7: theory/dynamics_params.rs + coupling.rs — 参数 + 耦合

**对应论文：** §3.4.3, §4.3.7, §6.4.4

**核心结构体：**
- `DynamicsParams`：11 参数（α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂, δ₃, ε₁, ε₂）
- `DynamicsParams::from_geometry(nv, ne, depth, max_depth, state, grad_mean, grad_std)`
  - 参数来源按论文 §3.4.3 的几何推导方向（详见 11 参数汇总表）

**耦合矩阵：**
- `compute_coupling(state_i, state_j, shared, total_i, total_j) -> f64`
- `build_coupling_matrix(memes) -> Vec<Vec<f64>>`

---

### Task 8: theory/ode.rs — 5D ODE + RKF45 + 原型分类

**对应论文：** §4.3-4.5, 定理 5.8, 定理 8

**核心结构体：**
- `OdeConfig`：h0, h_min, h_max, rtol, atol, max_steps(20000), t_max(5.0)
- `StateSnapshot`：t, d, b, rho, r, s
- `TerminationReason` 枚举：Converged / MaxSteps / MaxTime / InvalidInitialState / JumpDetected
- `Archetype` 枚举：Stone / StableCore / Resilient / Burst / Decay / Transient / Oscillatory / Source / Sink / Undetermined

**核心函数：**
- `ode_rhs(state, params, i_ext, phi_d, phi_r) -> [f64; 5]`
- `rkf45_step(state, t, h, params, k_d, k_r) -> ([f64; 5], f64)`
- `integrate(init, params, config, k_d, k_r) -> (Vec<StateSnapshot>, TerminationReason)`
- `classify(trajectory, reason) -> Archetype`（9 类）
- `classify_equilibrium(state) -> EquilibriumType`（5 种平衡点）

**关键细节：**
- I_ext(t) 默认 $e^{-t}$（脉冲衰减），非硬编码 0
- 收敛检测：连续 5 步总变化 < 1e-3
- 每步裁剪到 Ω = [0,1]⁴×[0,∞)（定理 7 不变性）
- 跳变检测：监控 β₀(K(t)) 变化

---

### Task 9: theory/jump_handler.rs — 跳变点处理

**对应论文：** 定理 6

**核心结构体：** `JumpHandler`

**核心函数：**
- `project_state(old_state, old_vertices, new_vertices) -> FiveDimState`：Π 投影映射
- `init_new_meme(parent_xi: &ExtendedDimension) -> FiveDimState`：从 ξ 解码新模因
- `absorb_meme(target_xi: &mut ExtendedDimension, vanished_state: &FiveDimState)`：消失模因吸收进 ξ
- `detect_jump(betti_0_old, betti_0_new) -> JumpType`：检测跳变类型

---

### Task 10: theory/optimizer.rs — 假设 0 全局优化

**对应论文：** §4.3.8, §D.5, 定理 6.2

**核心结构体：**
- `OptimizerConfig`：t_values, lambda, n_step
- `OptimalHypothesis`：t, family_d, family_r, k_d, k_r, loss

**核心函数：**
- `optimize(n_nodes, reconstruction_error_fn, config) -> OptimalHypothesis`

**搜索空间：** |T| × |F_D| × |Θ_D| × |F_R| × |Θ_R| = 7 × 5 × 5 × 5 × 5 = 4,375 个假设（网格搜索）

---

### Task 11: pipeline/phase1_emergence.rs — Phase 1 浮现

**对应论文：** §3.2, §6.2.2 M1, formal-concept-analysis-proof 定理 4.1-7.4

**数据流：**
```
words → FormalContext::from_words (C 从 W 自暴露)
     → Warshall 传递闭包
     → CycleEngine::cycle_all (↑↓ 循环，收敛检测)
     → 信息深度 = 收敛轮数
     → 彻底不完整判定（§3.2.5）
     → 推理与补全：5 种推理规则（§3.2.3）
     → 概念-要素循环组合 → 按深度分层（§3.2.4）
     → 规则推导 F：5 类规则 + 支撑集（§3.2.6）
     → 约束推导 C：4 类约束 + 作用域（§3.2.6）
     → 自洽性验证（§3.2.7）
     → PhaseOneOutput { s: S, f: F, c: C }
```

**核心结构体：**
- `PhaseOneOutput`：s (结构域 S), f (规则域 F), c (约束域 C), convergence_rounds, depth, reversibility_record

**核心函数：** `run_phase_one(words, max_rounds) -> PhaseOneOutput`

**关键细节：**
- 收敛检测：两轮间 word_closure 和 char_closure 都不再增长
- 传递闭包必须在 ↑↓ 循环之前执行（M1 数据流要求）
- 彻底不完整判定：循环无法推进 / 关系网络存在结构性空洞 → 生成补全请求
- 每条规则记录支撑集 supp ⊆ V∪E，每条约束记录作用域 dom ⊆ V∪E

---

### Task 12: pipeline/phase2_encoding.rs — Phase 2 编码

**对应论文：** §3.3, §6.3 M1, proof-supplement-complete 定义 2.4 + B1-B4 构造

**数据流：**
```
PhaseOneOutput → 词→0-胞腔 K⁰（supplement B1）
              → 概念内共现词对→1-胞腔 K¹（supplement B2: 同概念内的两个字建立边）
              → 概念(≥3字)→2-胞腔 K²（supplement B3: 概念封闭性编码）
              → 度量编码 g: 1-胞腔长度 = 连接强度（§3.3.3）
              → 标量场 φ(v) = 信息深度/max_depth（§3.3.3）
              → 向量场 F = -∇φ
              → 场编码 ω = Σ f·χ_supp（§3.3.4: F 中规则→几何场）
              → 不变量编码 Γ（§3.3.5: C 中约束→几何不变量）
              → 可逆性记录 R（§3.3.6 + supplement B4: 含 node_texts, node_is_word,
                 word_count, containment_depth, node_levels, concept_levels,
                 concept_termination_reasons, serialized_rules, serialized_constraints）
              → PhaseTwoOutput { complex, metric, fields, invariants, reversibility }
```

**核心结构体：**
- `PhaseTwoOutput`：complex (CWComplex), metric (g), fields (ω), invariants (Γ), reversibility (R)

**核心函数：** `run_phase_two(phase1: &PhaseOneOutput) -> PhaseTwoOutput`

**关键细节：**
- 1-胞腔的边按 supplement B2 构造：对同概念内的每对字建立边（而非简单共享字）
- 2-胞腔按 supplement B3 构造：含 ≥3 字且非最顶层的概念生成面
- 可逆性记录 R 按 supplement B4 的完整字段结构序列化

---

### Task 13: pipeline/phase3_decomposition.rs — Phase 3 分解

**对应论文：** §3.4, §6.4 M2, proof-supplement-complete 定义 3.1-3.4 + 定理 3.5-3.6

> **注：与论文 §3.4.1 的已知偏差** — 论文定义 $n = \beta_0(K)$（连通分量数），但实践中字词关系图往往形成单一巨连通分量（β₀=1），丧失模因分解意义。故采用 Louvain 社区检测作为替代分解策略，模块度 Q 作为分解质量的替代指标。辅助策略（曲率分割、层级边界）保留作为 fallback。此偏差由 proof-supplement-complete 定义 3.1 的 Betti 分解框架提供理论基准——当 Louvain 退化为连通分量时，两种方法等价。

**数据流：**
```
PhaseTwoOutput → 构建 Louvain 邻接表（边权=通量 |Γ¹(e)|）
              → Louvain 社区检测 → 模块度 Q 最大化
              → 每社区: 五维状态 m_i（§3.4.2 + supplement 定义 3.2）
              → 每社区: 扩展维度 ξ_i（supplement 定义 3.3: V_i/E_i/F_i 胞腔快照 + 边界链接 B_i）
              → 每社区: 11 参数 Θ_i 闭式推导（§3.4.3 + supplement 定义 3.4）
              → 耦合矩阵 C（§3.4.4, §6.4.4）
              → PhaseThreeOutput { memes: Vec<MemeState>, coupling: C }
```

**核心结构体：**
- `MemeState`：five_dim (FiveDimState), extended (ExtendedDimension), params (DynamicsParams)
- `PhaseThreeOutput`：memes, coupling, n_memes

**核心函数：** `run_phase_three(phase2: &PhaseTwoOutput) -> PhaseThreeOutput`

**关键细节：**
- 扩展维度 ξ 按 supplement 定义 3.3：包含 V_i (顶点快照), E_i (边快照), F_i (面快照), B_i (跨子几何体边界链接)
- 五维映射按 supplement 定义 3.2：D_i = |K⁰_i|/|K⁰|, B_i = |K¹_i|/|K¹|, ρ_i = avg(Γ¹(e)), R_i 由 b₁(X_i) 和边密度决定, S_i 由 2-胞腔数与维数比决定

---

### Task 14: pipeline/phase4_binding.rs — Phase 4 固化

**对应论文：** §3.5, §6.4.5

**核心结构体：**
```rust
struct Credential {
    header: CredentialHeader,     // version, timestamp, hash_algorithm
    data_hash: String,            // H(原始信息) — SHA-256
    meme_state: PhaseThreeOutput, // 完整的复合模因状态 Q（§3.5.2）
    metadata: CredentialMetadata, // original_size, meme_count, original_type
}
```

**核心函数：** `run_phase_four(phase3: &PhaseThreeOutput, original_input: &str) -> Credential`

**三种操作（§3.5.3）：**
- `create_credential(input, meme_state) -> Credential`：原始信息 + 复合模因状态 → 凭证
- `verify_credential(credential, candidate_input) -> bool`：凭证 + 待验证信息 → 哈希比对
- `restore_info(credential) -> (String, PhaseThreeOutput)`：凭证 → 还原原始信息 + 模因状态（通过逆映射链）

---

### Task 15: pipeline/phase5_evolution.rs — Phase 5 演化

**对应论文：** §4.3-4.4, M2

**数据流：**
```
PhaseThreeOutput → 假设0优化选 (Φ_D*, Φ_R*, k_d*, k_r*)
                → 对每个模因: RKF45 积分
                → 监控 β₀ 变化 → 跳变检测
                → 跳变时: Π 映射 + ξ 解码/吸收
                → 原型分类（9 类）
```

**核心函数：** `run_phase_five(phase3, config) -> OdeOutput`

---

### Task 16: io/ — 输入输出

- **tokenizer.rs**：`extract_ngrams(text) -> Vec<String>`（1/2/3-gram 中文分词）
  - **提取器接口契约**：实现 formal-concept-analysis-proof 推论 7.2 的接口 `A_F: RawData → (Nodes, Edges^containment)`
  - 架构支持多模态扩展（文本/图像/音频/代码/地理/数学证明），所有提取器输出统一 WikiLine TSV 格式写回 Ψ
- **tsv.rs**：`TsvWriter`（WikiLine TSV 输出，每行 `(entity, predicate, target)` 三元组）
- **reversibility.rs**：`verify_roundtrip(original_words, p1, p2, p3) -> ReversibilityReport`
  - 验证 Φ⁻¹∘Φ = I（推论 5.1）
  - 验证 ↓(↑(c)) = c 和 ↑(↓(w)) = w（formal-concept-analysis-proof 公理 3 互逆性）
  - 包含 Shannon 熵跨阶段比较（近似 Kolmogorov 复杂度，proof-supplement-complete 推论 4.4）

---

### Task 17: main.rs + 集成测试

**main.rs**（~100 行）：CLI 参数解析 + 五阶段管线编排。

**集成测试** `tests/full_pipeline.rs`：
- 完整五阶段端到端测试
- NaN 消解验证（全部 is_finite）
- 往返验证（Φ⁻¹∘Φ 词/字完全匹配）
- ODE 收敛验证（轨道无 InvalidInitialState）
- 可证伪判据验证（多次运行选出的 Φ* 在同一族）

---

### Task 18: 删除旧代码 + 最终验证

删除旧模块：`emergence/`, `encoding/`, `sealing/`, `infra/`, `tests/pipeline_integration.rs`

```bash
cd updown_rs && cargo test -- --nocapture
```

---

## 执行检查清单

- [ ] Task 1: 项目骨架搭建
- [ ] Task 2: theory/fca.rs — FCA ↑↓ 循环
- [ ] Task 3: theory/cw_complex.rs + vector_field.rs — CW 复形 + 梯度场
- [ ] Task 4: theory/louvain.rs — Louvain 社区检测
- [ ] Task 5: theory/five_dim.rs + extended_dimension.rs — 五维 + ξ
- [ ] Task 6: theory/function_families.rs — 五族函数
- [ ] Task 7: theory/dynamics_params.rs + coupling.rs — 参数 + 耦合
- [ ] Task 8: theory/ode.rs — 5D ODE + 原型分类
- [ ] Task 9: theory/jump_handler.rs — 跳变处理
- [ ] Task 10: theory/optimizer.rs — 假设 0 优化
- [ ] Task 11: pipeline/phase1_emergence.rs — Phase 1
- [ ] Task 12: pipeline/phase2_encoding.rs — Phase 2
- [ ] Task 13: pipeline/phase3_decomposition.rs — Phase 3
- [ ] Task 14: pipeline/phase4_binding.rs — Phase 4
- [ ] Task 15: pipeline/phase5_evolution.rs — Phase 5
- [ ] Task 16: io/ — 输入输出
- [ ] Task 17: main.rs + 集成测试
- [ ] Task 18: 删除旧代码 + 最终验证

---

## 关键设计决策记录

1. **五族函数按论文原文**：Power / Exponential / Sigmoid(减1/2) / Logarithm / PiecewiseLinear。不包含 Logistic 和 Threshold（那是 v2 的错误）。

2. **I_ext(t) 默认 $e^{-t}$**：论文 §4.3.4 描述为"脉冲或衰减函数"。默认指数衰减模拟初始能流注入后自然消散。

3. **ODE 跳变检测**：监控 β₀(K(t)) 变化。当拓扑改变时停止积分，执行 Π 映射，然后从新初始条件继续。

4. **Warshall 传递闭包**：论文 M1 数据流明确要求。在 Phase 1 的 ↑↓ 循环之前执行。

5. **扩展维度 ξ**：保证 Φ⁻¹∘Φ 严格双射。不是可选优化——是定理 6 跳变映射和定理 5 双射性的必需组件。结构按 supplement 定义 3.3 的 `(V_i, E_i, F_i, B_i)` 四元组。

6. **可证伪判据**：集成测试中验证多次运行选出的 Φ* 在同一函数族。这是理论的核心验证标准（§7.3 M4）。

7. **Phase 1 输出完整 M=(S,F,C)**：遵循论文 §3.2.6-3.2.7 和 supplement 定义 2.1。S 为结构域，F 为规则域（5 类规则 + 支撑集），C 为约束域（4 类约束 + 作用域）。

8. **Phase 2 输出完整 G=(K,g,ω,Γ,R)**：遵循论文 §3.3.2-3.3.6 和 supplement 定义 2.4 + 构造 B1-B4。1-胞腔按 supplement B2 的概念内共现建边，非简单共享字。

9. **Louvain 偏差于论文 β₀**：论文 §3.4.1 定义 n=β₀(K)，但实践中字词关系图常形成单一巨连通分量。Louvain 社区检测是合理的工程替代，当 Louvain 退化为连通分量时与论文方法等价。

10. **参数范围使用 supplement 紧致化**：Θ 紧致化范围（supplement §6.2）保证 ODE 右侧函数的 Lipschitz 性质，而非论文正文 §4.3.8 的使用建议区间。

11. **信息守恒用 Kolmogorov 复杂度定义**：遵循 supplement 第四章，Shannon 熵作为 Kolmogorov 复杂度的近似。$|H(X) - H(Y)| \leq C_{\Phi}$ 是精确的不等式，而非严格等号。

12. **提取器接口契约**：tokenizer 实现 formal-concept-analysis-proof 推论 7.2 的接口，支持多模态扩展（文本/图像/音频/代码/地理/数学证明）。

13. **代码量估计**：旧代码 7233 行 → 新代码 ~4000 行（增加 F/C 推导 + g/ω/Γ/R 编码 + 完整推理 + 提取器契约，但结构更清晰）。