# 泛模因理论（Pan-Meme Theory）— N-迭代定理验证引擎

> 📚 **文献综述**：[docs/related-work.md](docs/related-work.md) — 七领域完整引用  
> 📄 **理论**：[泛模因理论-完整知识库](docs/泛模因理论-完整知识库.md) — 11 章正文 + 定理索引 + 实验附录 §11.8  
> 📋 **证明补充**：[双射·信息守恒·原型收敛·紧致优化](docs/proof-supplement-complete.md)  
> 📊 **实验**：[实验索引](experiments/README.md) — 11 个实验（000-011），Theorem 11.1 在三重参数扫描下 48/48 = 100% 确认  
> 🦀 **活跃开发**：`n_iter_rs` — Chain C N-迭代管线（35 tests, 零警告）
> 
> 目前基本还是在实验阶段数学基础有bug
> 
> **核心命题**：在足够丰富的跨领域数据输入下，形式概念分析（FCA）概念格 + 离散 N 算子（Definition 6.12）构成的信息-结构动力学系统必然满足偏序单调性定理（Theorem 11.1/11.3）。该命题已在 45 篇 Wikipedia 文章 × 三重参数扫描下以 48/48 = 100% 严格确认。

---

## 链 A-B-C 关系

本项目的理论演进分为三条链，从 ODE 原型到离散 N-算子定理验证：

| 链 | 动力学 | 参数来源 | 引擎 | 用途 |
|----|--------|----------|------|------|
| **A** | 连续 ODE | 5 种函数族 | Python `pan_meme` | 理论原型 |
| **B** | 谱参数 + ODE | 热迹/谱 | Rust `updown_rs` | 五阶段全流程 · 实验 000-010 |
| **C** | 离散 N 算子 · A/(A+B+ε) | 统一参数 (all=1.0) | Rust `n_iter_rs` | **定理验证核心 · 当前活跃** |

**关键分叉**：链 B 的 ODE 系统依赖手工选择的函数族和谱参数提取，收敛分类依赖于连续动力学的数值求解。链 C 以离散 N 算子（定义 6.12）替代 ODE——不动点 M\* = N(M\*) 由 A/(A+B+ε) 结构自然确定，11 个耦合系数统一为 1.0，参数域被定理 6.8 唯一确定。

---

## Chain C 管线（n_iter_rs · 当前核心）

```
Wikipedia 文章 (45 篇, 8 类别)
        │
        ▼
  ┌─────────────┐
  │  FCA 概念格   │  article-level word FCA
  │  NextClosure │  → 3-4 形式概念
  └──────┬──────┘
         │ Hasse 边 (general→specific)
         ▼
  ┌─────────────┐
  │  N 迭代      │  5D N 算子 · A/(A+B+ε)
  │  统一参数    │  B↑ 信息流 (specific→general)
  └──────┬──────┘
         │ M* = N(M*)
         ▼
  ┌─────────────────────────────┐
  │  定理验证                    │
  │  Thm 11.1 τ⁻¹ 单调性  100%  │
  │  Thm 11.3 D 单调性    100%  │
  │  Thm 6.17 轨迹守恒    0.00e0 │
  │  推论 6.13A D* 恒等式 确认   │
  └─────────────────────────────┘
```

### N 算子（Definition 6.12）

链 C 的核心是离散 N 算子——5 维状态向量 M = (D, B, ρ, R, S) 经 A/(A+B+ε) 形式的一次迭代：

$$N_D = \frac{\alpha_1 R + \varepsilon}{\alpha_1 R + \beta_1(B + B_{\uparrow}) + \varepsilon}, \quad
N_B = \frac{\gamma_1(R + B_{\uparrow}) + \varepsilon}{\gamma_1(R + B_{\uparrow}) + \delta_1 D + \varepsilon}$$

$$N_\rho = \frac{\zeta_1(D + \rho_{\uparrow}) + \varepsilon}{\zeta_1(D + \rho_{\uparrow}) + \eta_1 R + \varepsilon}, \quad
N_R = \frac{\theta_1(\rho + \rho_{\uparrow} + B_{\uparrow}) + \varepsilon}{\theta_1(\rho + \rho_{\uparrow} + B_{\uparrow}) + \kappa_1 D + \kappa_2 S + \varepsilon}$$

$$N_S = \frac{\lambda_1 D + \varepsilon}{\lambda_1 D + \mu_1 R + \varepsilon}$$

**关键性质**：
- **A/(A+B+ε) 结构**：每个分量天然归一化到 (0,1)，无需外部 sigmoid
- **离散第一性**：每次迭代 = 完整 P₇ 周期（复制+变异+选择），非 ODE 离散化
- **11 耦合系数**：由谱理论（定理 6.8）唯一确定，默认统一为 1.0
- **B↑/ρ↑ 格耦合**：子概念的外延广度和谱能量向上注入父概念（P₈）

### 定理验证总和（2026-07-11）

| 定理 | 验证范围 | 结果 |
|------|----------|------|
| **Theorem 11.1** τ⁻¹ 单调性 | 8 组类别 + β₁×13 + δ₁×13 | **48/48 = 100%** |
| **Theorem 11.3** FCA D₀=\|A\|/\|B\| 单调性 | 8 组类别 | **22/22 = 100%** |
| **Empirical** N-iter D\* 单调性 | 8 组类别 | **22/22 = 100%** |
| **Theorem 6.17** 轨迹守恒 | 全实验 | max dev = **0.00e0** |
| **推论 6.13A** D\* 不动点恒等式 | 机器精度 | **0.00 偏差** |

### M\* 不动点数据（4-概念格，统一参数）

| 概念 | Hasse高 | D\*(自足率) | B\* | τ⁻¹ | ρ(J_N) | B↑(注入) |
|------|---------|------------|------|------|--------|----------|
| C0 根 | h=0 | 0.3161 | 0.8342 | 1.0036 | 0.3666 | 0.8240 |
| C1 | h=1 | 0.3243 | 0.8240 | 0.9958 | 0.3694 | 0.7585 |
| C2 | h=2 | 0.3664 | 0.7585 | 0.9571 | 0.3840 | 0.4511 |
| C3 叶 | h=3 | 0.4511 | 0.4511 | 0.5998 | 0.5489 | 0.0000 |

**五维状态沿 Hasse 边（general→specific）的方向规律**：

| 分量 | 方向 | 物理含义 |
|------|------|----------|
| D\* | ↑ 递增 | 自给自足率：越具体越自足 |
| B\* | ↓ 递减 | 外延广度：越具体覆盖越少 |
| τ⁻¹ | ↓ 递减 | 收敛速度：general 因 B↑ 注入而最快 |
| ρ(J_N) | ↑ 递增 | 谱半径单调性：ρ_gen ≤ ρ_spec 在所有参数下成立 |

### 三大核心发现

1. **D\* 不动点恒等式**（推论 6.13A）：D\* = (R\*+ε)/(R\*+B\*+B↑+ε) — 不动点 D\* 的解析身份是"自给自足率"，以机器精度成立

2. **B↑ 注入 → τ⁻¹ 单调性**：隔离实验中关掉格耦合（B↑=0, ρ↑=0）后，4 概念收敛到**完全相同的 M\***。τ⁻¹ 梯度 100% 来自格耦合的 B↑ 注入。无耦合则无差异。

3. **Jacobian 主导通路转移**：5×5 Jacobian 热图显示—高 B↑（根概念）→ ∂N_S/∂D 主导（结构→抗断裂）；低 B↑（叶概念）→ ∂N_ρ/∂R 主导（速率→潜力）。因果通路沿 Hasse 高度连续变换。

---

## 理论基础

### 八个核心命题

| 命题 | 核心主张 |
|------|----------|
| **P₁ 信息本体论** | 信息是现实的基本构成维度，与物质-能量并列；信息不可还原为物质或能量 |
| **P₂ 泛模因实在论** | 一切可复制、可演化、可保持同一性的信息-结构模式皆为泛模因 |
| **P₃ 结构现实主义** | 真正持久存在的不是实体，而是实体之间的稳定关系模式（结构） |
| **P₄ 认知-建模统一性** | 认知本身就是建模活动；理论将隐式认知转化为可显式执行的方法论 |
| **P₅ 可逆性认识论** | 建模的每一步都有明确定义的逆操作——Φ⁻¹(Φ(I)) ≡ I |
| **P₆ 计算作为模因属性** | 图灵完备性是泛模因系统的逻辑延伸，非外部引入 |
| **P₇ 演化普遍性** | 复制-变异-选择三要素在任何泛模因系统中必然引发演化行为 |
| **P₈ 层级涌现** | 低层泛模因通过组合形成高层泛模因；力的分配：向上因果 B↑,ρ↑→推父；向下因果 D→限子，S→稳子 |

> **公理退化分析（2026-07-11 ■）**：经系统性审查，8 条公理可缩减至 **5 条原始公理**（详见 [§12.1A](docs/泛模因理论-完整知识库.md)）：
> - **P₄** → 哲学预设（路径 3 零使用，可从 P₃+P₅ 推导）
> - **P₆** → 定理 1.6（从 P₂ 直接推断：T 满足 replicable ∧ evolvable ∧ identity_preserving）
> - **P₇** → 合并入 P₂'（与 P₂ 的定义重叠，P₂ 的"evolvable"操作化为 S_{t+1}=S_e∘V∘R）
> - 重构后：**P₁**（本体论）、**P₂'**（P₂+P₇ 合并）、**P₃**（结构）、**P₄'**（原 P₅ 可逆性）、**P₅'**（原 P₈ 层级）

### 链 C 的核心定理

| 定理 | 内容 | 验证状态 |
|------|------|----------|
| **Def 6.12** | N 算子：5 维 A/(A+B+ε) | 实现 |
| **Thm 6.15** | Jacobian 5×5 稀疏矩阵（13 非零项 · 对角线全零） | **48/48 实验确认** |
| **推论 6.13A** | D\* = (R\*+ε)/(R\*+B\*+B↑+ε) | **机器精度确认** |
| **Thm 6.17** | N 算子轨迹严格确定论（非混沌） | max dev = 0.00e0 |
| **Thm 11.1** | τ⁻¹ 沿 Hasse 边单调递减（τ_gen ≥ τ_spec） | **48/48 = 100%** |
| **Thm 11.3** | D₀ = \|A\|/\|B\| 沿 Hasse 边单调递增 | **22/22 = 100%** |
| **Thm 6.8** | 11 耦合系数由谱理论唯一确定 | 默认统一=1.0 |

### 链 A/B 的历史定理（Python 原型 + updown_rs）

| 定理 | 内容 |
|------|------|
| 定理 1-5 | 四阶段双射复合 · Φ⁻¹(Φ(I)) = I（完全可逆） |
| 定理 6-8 | 5D ODE 系统解的存在唯一性 · Lipschitz 连续 · 收敛性 |
| 定理 9 | 信息守恒：H(I) = H(Ψ) = H(M) = H(G) = H(Q) |
| 定理 10 | 九类演化原型的分类与收敛判定 |
| 假设 0 | H = T × F × Θ × N 全局优化 |

---

## 项目结构

```
模因/
├── README.md                                # 本文件
├── .gitignore
│
├── docs/
│   ├── 泛模因理论-完整知识库.md                #   ~2000 行，11 章 + 定理索引 + 实验附录 §11.8
│   ├── formal-concept-analysis-proof.md       #   形式概念分析证明
│   ├── proof-supplement-complete.md           #   双射·信息守恒·原型收敛
│   ├── related-work.md                        #   七领域文献综述
│   └── v4.2-changelog.md                      #   Chain B 工程升级
│
├── experiments/                               #   实验（000-011）
│   ├── README.md                              #     实验索引 + Chain C 定理验证汇总
│   └── 011-fca-n-iteration/                   #     Chain C FCA + N-迭代定理验证
│       ├── param_scan_beta1.tsv               #       β₁ 扫描结果
│       ├── param_scan_delta1.tsv              #       δ₁ 扫描结果
│       └── isolation_experiment.tsv           #       隔离对照实验
│
├── n_iter_rs/                                 # Rust N-迭代引擎（Chain C · 核心）
│   ├── Cargo.toml                             #   依赖：nalgebra 0.33
│   └── src/
│       ├── main.rs                            #   实验主入口（FCA + N-迭代 + 参数扫描）
│       ├── lib.rs                             #   14 模块声明
│       ├── n_operator.rs                      #   5D N 算子（Def 6.12）+ Jacobian（Thm 6.15）
│       ├── fca.rs                             #   形式概念分析（NextClosure）+ Hasse 边 + 定理验证
│       ├── five_dim.rs                        #   5D 状态向量 + Ω 域约束
│       ├── spectrum.rs                        #   谱参数提取（Chain B 遗留）
│       ├── tokenizer.rs                       #   中英文 n-gram 提取
│       ├── coupling.rs                        #   Jaccard × State 耦合矩阵
│       ├── cw_complex.rs                      #   CW 胞腔复形 + Betti 数
│       ├── vector_field.rs                    #   标量场 + 向量场 F = -∇φ
│       ├── louvain.rs                         #   两阶段模块度最大化
│       ├── reversibility.rs                   #   Shannon 熵 + 往返一致性
│       ├── classify.rs                        #   SVD-PCA + k-means + 聚类纯度
│       ├── io.rs                              #   Wikipedia JSON 解析
│       ├── tsv.rs                             #   TSV 输出
│       └── types.rs                           #   5 个语义 newtype
│
├── updown_rs/                                 # Rust ↑↓ 引擎（Chain B · 历史）
│   └── src/                                   #   五阶段 ODE 管线（详见 updown_rs/README.md）
│
├── pan_meme/                                  # Python 原型工具包（Chain A）
│   └── PB_ARCHITECTURE.md                     #   PB 级工业架构白皮书
│
├── scripts/                                   # Python 运行脚本
│   └── ...
│
└── data/                                      # 数据文件
    └── background_tree.json                   #   新华字典背景层级树
```

---

## 快速开始

### 环境要求

- **Rust** Toolchain ≥ 1.75（`n_iter_rs` / `updown_rs`）
- **Python** ≥ 3.10（Chain A 工具包 · 可选）

### Chain C — n_iter_rs（当前活跃）

```bash
cd n_iter_rs
cargo build --release
cargo run --release          # 45 篇 Wikipedia → FCA + N-迭代 + 参数扫描
cargo test                   # 35 个测试，全部通过
```

### Chain B — updown_rs（历史）

```bash
cd updown_rs
cargo build --release
./target/release/updown input.txt
cargo test --lib             # 70 个测试（历史套件）
```

### Chain A — Python 原型（历史）

```bash
python scripts/run_dictionary.py
```

---

## 测试覆盖

### Chain C — n_iter_rs（35 tests · 零警告）

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| n_operator | 4 | N 算子收敛 · Jacobian 谱半径 · 定义域约束 · 轨迹守恒 |
| fca | 2 | FCA 概念格构建 · Theorem 11.3 验证 |
| five_dim | 4 | 状态有效性 · NaN 守卫 · Ω 域约束 · 无效状态拒绝 |
| tokenizer | 3 | 中文 n-gram · 空输入 · 提取器接口 |
| coupling | 3 | 空共享 · 完全共享 · 耦合矩阵 |
| cw_complex | 7 | 顶点/边添加 · 空图 · 简单图 · Betti 不变量 · 隔离概念 |
| vector_field | 3 | 标量场 · 向量场 · NaN 守卫 |
| louvain | 3 | 空图 · 单节点 · 双社区 |
| reversibility | 2 | Shannon 熵 · 往返一致性 |
| classify | 2 | k-means · PCA 维度 |
| spectrum | 2 | Bigram 数据 · 谱参数 |
| tsv | 1 | 三元组输出 |

### Chain B — updown_rs（70 tests · 历史套件）

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| emergence | 18 | 提取器 · ↑↓循环 · 五类推理 · Louvain |
| encoding | 12 | CW复形 · 标量/向量场 · Betti分解 · 五维映射 · 11参数 · 耦合 |
| sealing | 23 | SHA-256 · Merkle树 · RKF45 · 五函数族 · 优化器 |
| pipeline | 5 | Phase 1-5 全管线 · 收敛扫描 · 外部验证 |
| infra | 2 | Tokenizer 注册 · 命名 |

---

## 设计原则

1. **定理驱动开发**：每个算法直接对应理论文档中的定义和定理——N 算子 ⇔ Def 6.12，Jacobian ⇔ Thm 6.15，单调性 ⇔ Thm 11.1/11.3
2. **离散第一性**：N 迭代是离散映射，非 ODE 离散化——每次迭代 = 完整 P₇ 周期
3. **参数节俭**：11 耦合系数统一为 1.0，由定理 6.8（谱理论）唯一确定——零手工调参
4. **可证伪性**：Theorem 11.1 在三重参数扫描下可独立验证——48/48 = 100% 是最强的确认，不是调参调出来的
5. **最小依赖**：n_iter_rs 仅依赖 nalgebra 0.33（5×5 矩阵 Jacobian 计算 + 特征值）

---

## 引用

理论文档：[`docs/泛模因理论-完整知识库.md`](docs/泛模因理论-完整知识库.md) — 11 章 + 定理索引 + 实验附录 §11.8。

所有实验数据：[`experiments/README.md`](experiments/README.md)

学术背景：
- Dawkins, R. (1976). *The Selfish Gene*. Oxford University Press.
- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
- Arthur, W. B. (2009). *The Nature of Technology*. Free Press.
- Ganter, B. & Wille, R. (1999). *Formal Concept Analysis*. Springer.
