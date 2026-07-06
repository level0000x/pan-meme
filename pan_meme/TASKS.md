# pan-meme 实现任务拆分

**版本**: v1.0 | **日期**: 2026-06-29 | **总文件数**: ~85 个  
**原则**: 每完成一个 Phase 即可独立验证，不依赖后续 Phase。

---

## Phase 0: 项目基础设施 (6 files, 优先级 P0)

**目标**: 可 `pip install -e .` + `cargo build`

| # | 文件 | 内容 | 验证 |
|---|------|------|------|
| 0.1 | `pyproject.toml` | setuptools 配置, 依赖: numpy, scipy, networkx, jieba, matplotlib | `pip install -e .` |
| 0.2 | `pan_meme/__init__.py` | 空, 导出版本号 | `import pan_meme` |
| 0.3 | `rust_native/Cargo.toml` | [lib] crate-type=["cdylib"], 依赖 pyo3, nalgebra, petgraph | `cargo build` |
| 0.4 | `rust_native/src/lib.rs` | pyo3 入口, 空模块, `#[pymodule] fn pan_meme_rs` | `import pan_meme_rs` |
| 0.5 | `config/default.json` | 完整配置文件 (从设计文档11节复制) | `json.load()` |
| 0.6 | `config/schema.py` | JSON schema 校验函数 | pytest |

**完成标志**: `python -c "import pan_meme; print(pan_meme.__version__)"` 输出 `0.1.0`

---

## Phase 1: 核心类型系统 (1 file, 优先级 P0)

**数学对应**: 附录D.1-D.2 全部定义

| # | 文件 | 内容 |
|---|------|------|
| 1.1 | `core/types.py` | 设计文档1.4节全部 @dataclass: Token, HierarchyNode, HierarchyTree, RelationNetwork, GapInfo, CompletenessReport, RuleDef, ConstraintDef, MathModel, SimplicialComplex, GeometricObject, MemeState, CompositeMemeState, HierarchicalHash, MerkleTree, Credential, PipelineData, ODEConfig, ODEResult, OptimizationResult |

**验证**: `python -c "from pan_meme.core.types import *; t=Token(modality='text',text='test',span=(0,4),pos='NN'); print(t)"`

---

## Phase 2: 插件系统 (4 files, 优先级 P0)

**数学对应**: 附录D.3 候选函数族

| # | 文件 | 内容 |
|---|------|------|
| 2.1 | `plugins/registry.py` | PluginRegistry: _tokenizers, _functions, _strategies 三个 Dict + register/get 方法 |
| 2.2 | `plugins/functions/base.py` | BaseFunction(ABC): evaluate, derivative, lipschitz_constant, n_params |
| 2.3 | `plugins/functions/{power,exp,sigmoid,log,piecewise}.py` | 5个实现类 (PowerFunction, ExpFunction, SigmoidFunction, LogFunction, PiecewiseLinearFunction), 构造时注册 |
| 2.4 | `plugins/modalities/base.py` | BaseModalityTokenizer(ABC): modality, tokenize, can_handle |

**验证**: `func = PluginRegistry.get_function("power")(k=2.0); assert abs(func.lipschitz_constant - 2.0) < 1e-10`

---

## Phase 3: 多模态 Tokenizer (6 files, 优先级 P0)

**数学对应**: 公理1 — I ∈ U 的原子化

| # | 文件 | 内容 |
|---|------|------|
| 3.1 | `module1_input/tokenizer.py` | Tokenizer 类: _detect_modality → PluginRegistry.get_tokenizer → plugin.tokenize() |
| 3.2 | `plugins/modalities/text_zh.py` | TextZhTokenizer: jieba 分词 + 词性标注 |
| 3.3 | `plugins/modalities/text_en.py` | TextEnTokenizer: spacy tokenize + POS |
| 3.4 | `plugins/modalities/image_rgb.py` | ImageRGBTokenizer: 16×16 patch → Token, payload 含 patch_rgba |
| 3.5 | `plugins/modalities/structured_json.py` | StructuredJsonTokenizer: 递归遍历 dict/list |
| 3.6 | `plugins/modalities/__init__.py` | 注册 4 个 tokenizer 到 PluginRegistry |

**验证**: `tokens = Tokenizer().tokenize("信息是现实的基本维度")` → ≥4 Token, span 不重叠

---

## Phase 4: 模块一 — ↑↓ 循环 + 关系提取 (3 files, 优先级 P0)

**数学对应**: 公理3 (↑↓互逆) + 前提2 (算法A)

| # | 文件 | 内容 |
|---|------|------|
| 4.1 | `module1_input/cycle.py` | CycleEngine: run() 实现 ↑↓ 双循环, 收敛检测, termination_record |
| 4.2 | `module1_input/relation_extractor.py` | RelationExtractor: 对所有节点对计算 w_ij = max(共现, 语义), 按阈值 T 切分 |
| 4.3 | `core/exceptions.py` | ModalityNotSupportedError, PipelineError, ConsistencyViolation |

**验证**: 输入 "水果包括苹果和香蕉" → cycle 运行后 "水果" 的 children 含 "苹果" "香蕉"

---

## Phase 5: 模块一 — 推理 + 完整性 + 概念组合 (3 files, 优先级 P0)

**数学对应**: 前提1 (可识别结构验证) + 前提3 (模式确定性)

| # | 文件 | 内容 |
|---|------|------|
| 5.1 | `module1_input/reasoner.py` | Reasoner: Warshall 传递闭包 + 对称 + 共现 + 模式补全 + 结构相似 |
| 5.2 | `module1_input/completeness.py` | CompletenessChecker: 连通分量 < 80%? 孤立 > 20%? depth=0? |
| 5.3 | `module1_input/concept_composer.py` | ConceptComposer: Louvain 多层社区检测 |

**验证**: 链状 A→B→C 推理后出现 A→C, 权重 ≤ min(w_AB, w_BC)

---

## Phase 6: 模块一 — 规则 + 一致性 + 主入口 (3 files, 优先级 P0)

**数学对应**: 定理1+2 — Φ_B ∘ Φ_A 双射

| # | 文件 | 内容 |
|---|------|------|
| 6.1 | `module1_input/rule_extractor.py` | RuleExtractor: 幂律检测 + 团检测 + 孤立约束 |
| 6.2 | `module1_input/consistency.py` | ConsistencyChecker: support/domain 边界检查 + repair |
| 6.3 | `module1_input/adapter.py` | InputAdapter: 8步管线编排 → MathModel + inverse() |

**验证**: `adapter.adapt("信息是现实的基本维度。结构具有因果效力。")` → MathModel 非空

---

## Phase 7: 模块二 — 几何化 (4 files, 优先级 P0)

**数学对应**: 定理3 — Φ_C: M → G 双射

| # | 文件 | 内容 |
|---|------|------|
| 7.1 | `module2_geo/skeleton.py` | SkeletonEncoder: Ψ → SimplicialComplex (vertex_map + edge_map) |
| 7.2 | `module2_geo/metric_encoder.py` | MetricEncoder: w → g (逐边赋值) |
| 7.3 | `module2_geo/field_encoder.py` | FieldEncoder: F → ω (rule.confidence 在 support 上累加) + decode |
| 7.4 | `module2_geo/invariant_encoder.py` | InvariantEncoder: C → Γ (欧拉示性数 + 约束类型) + decode |

**验证**: encode → decode round-trip: 逐边比对一致

---

## Phase 8: 模块二 + 三 主入口 (3 files, 优先级 P0)

**数学对应**: 定理3 + 定理4

| # | 文件 | 内容 |
|---|------|------|
| 8.1 | `module2_geo/geometrizer.py` | Geometrizer: encode (5步) + decode (逆映射) |
| 8.2 | `module3_meme/geometry_split.py` | GeometrySplit: scipy beta0 连通分量+ merge |
| 8.3 | `module3_meme/mapping_5d.py` | Mapping5D: K_i → (D,B,ρ,R,S,ξ). D=cell×depth, B=edge_density, ρ=mean(|ω|), R=std(∇ω), S=|Γ|/10, ξ=g-median(g) |

**验证**: φ_h^{-1}(φ_h(K)) = K

---

## Phase 9: 模块三 — 参数 + 耦合 + 主入口 (4 files, 优先级 P0)

**数学对应**: 定理4 + 论文4.3.7

| # | 文件 | 内容 |
|---|------|------|
| 9.1 | `module3_meme/param_derive.py` | ParamDerive: 11参数闭式推导 (α₁=边密度, α₂=depth/d_max, ...) |
| 9.2 | `module3_meme/coupling.py` | Coupling: C_ij = 共享边×0.3 + 共享顶点×0.5 + 同层级×0.2 |
| 9.3 | `module3_meme/decomposer.py` | MemeDecomposer: decompose (4步) + reconstruct (合并逆映射) |
| 9.4 | `module1_input/__init__.py` | 空, 从 adapter 导入 |

**验证**: 11参数全 ≥0, α₁≤1, C 对称

---

## Phase 10: 模块四 — 分层哈希 (4 files, 优先级 P0)

**数学对应**: 论文3.5节

| # | 文件 | 内容 |
|---|------|------|
| 10.1 | `module4_bind/hierarchy_hasher.py` | HierarchyHasher: hash_token, hash_hierarchy_tree, hash_relation_network, hash_math_model, hash_meme (5d|xi), hash_composite (Θ|C) |
| 10.2 | `module4_bind/merkle_tree.py` | MerkleTreeBuilder: build (索引叶子→mod1/2/3→root), MerkleVerifier: verify_root/verify_layer/locate_tamper, MerkleUpdater: update_meme/split_meme/merge_memes |
| 10.3 | `module4_bind/credential.py` | CredentialAssembler, HashBasedIndex |
| 10.4 | `module4_bind/binder.py` | CredentialBinder: bind (累积哈希→MerkleTree→凭证), verify_full, verify_layer, verify_component, locate_tamper |

**验证**: 有效凭证 verify_full → True; 篡改任意 meme D 值 → locate_tamper 精确定位

---

## Phase 11: 引擎 — ODE 求解器 (2 files, 优先级 P1)

**数学对应**: 定理6+7 + 论文4.3.1

| # | 文件 | 内容 |
|---|------|------|
| 11.1 | `engines/ode_solver.py` | ODESolver: solve_single (scipy solve_ivp RK45, 5方程RHS, Ω clipping), solve_multi, _classify_convergence |
| 11.2 | `engines/convergence.py` | 定理10收敛分析: 能量函数 V=½∑(D²+B²+ρ²+R²+S²), LaSalle 不变原理, 三类原型判断 |

**验证**: D=B=ρ=R=S=0.5, 所有参数=0.5, Φ(x)=x → Ω 不变性 property test

---

## Phase 12: 引擎 — 全局优化器 (1 file, 优先级 P1)

**数学对应**: 附录D.3 假设0 + 拆分优化

| # | 文件 | 内容 |
|---|------|------|
| 12.1 | `engines/optimizer.py` | GlobalOptimizer: 穷举(T×F×N) + L-BFGS-B(Θ), 损失=fidelity+λ×complexity |

**验证**: 合成案例已知 ground truth → 优化器输出误差 < 1e-6

---

## Phase 13: 管线编排器 (1 file, 优先级 P0)

**数学对应**: 定理5 — Φ = Φ_D ∘ Φ_C ∘ Φ_B ∘ Φ_A

| # | 文件 | 内容 |
|---|------|------|
| 13.1 | `core/pipeline.py` | Pipeline: run_forward (4模块串联+哈希累积), run_inverse (推论5.1), run_with_optimization |

**验证**: hypothesis property test: 随机文本 round-trip 逐 token 还原率 100%

---

## Phase 14: IO + 序列化 (2 files, 优先级 P1)

| # | 文件 | 内容 |
|---|------|------|
| 14.1 | `io/serializer.py` | PipelineData ↔ pickle/json; Credential ↔ json |
| 14.2 | `io/loader.py` | 自动检测文件类型, 分派到对应 tokenizer |

---

## Phase 15: Rust 原生扩展 (4 files, 优先级 P2)

| # | 文件 | 内容 |
|---|------|------|
| 15.1 | `rust_native/src/graph_ops.rs` | petgraph: Warshall 传递闭包, connected_components |
| 15.2 | `rust_native/src/ode_rkf.rs` | 自研 RKF45: 比 scipy 快 3-10x, 可嵌入 Python 管线 |
| 15.3 | `rust_native/src/sparse_linalg.rs` | nalgebra: 稀疏矩阵 CSR 乘法, 拉普拉斯零特征值 |
| 15.4 | `rust_native/Cargo.toml` | [dependencies] pyo3, nalgebra, petgraph |

---

## Phase 16: 测试套件 (每 Phase 完成后补, 持续)

| # | 目录 | 内容 |
|---|------|------|
| 16.1 | `tests/test_module1/` | Tokenizer/Cycle/Relation/Reasoner/Completeness/Composer/Rules/Consistency/Adapter 单元测试 |
| 16.2 | `tests/test_module2/` | Skeleton/Metric/Field/Invariant/Geometrizer 单元测试 |
| 16.3 | `tests/test_module3/` | Split/Mapping5D/ParamDerive/Coupling/Decomposer 单元测试 |
| 16.4 | `tests/test_module4/` | Hasher/Merkle/Credential/Binder 单元测试 |
| 16.5 | `tests/test_engines/` | ODE/Optimizer/Convergence 单元测试 |
| 16.6 | `tests/test_e2e/` | 完整 round-trip + property tests + 多模态测试 |
| 16.7 | `tests/conftest.py` | fixtures: sample_text, sample_json, pipeline_instance, hypothesis strategies |

---

## Phase 17: Lean 4 形式化验证 (持续进行)

| # | 文件 | 内容 |
|---|------|------|
| 17.1 | `formal/lakefile.lean` | Lean 4 项目配置, 复用 Lv-00 的 lake 构建 |
| 17.2 | `formal/lean-toolchain` | leanprover/lean4:v4.14.0 (已从 Lv-00 复制) |
| 17.3 | `formal/Defs.lean` | 形式化定理1-10 的 Lean 定义 |
| 17.4 | `formal/Bijection.lean` | 定理5 双射证明 |
| 17.5 | `formal/Conservation.lean` | 定理9 信息守恒证明 |

---

## 依赖关系图

```
Phase 0 (基础设施)
    ↓
Phase 1 (类型系统)
    ↓
Phase 2 (插件系统) ←────────────────────────────┐
    ↓                                            │
Phase 3 (多模态 Tokenizer)                      │
    ↓                                            │
Phase 4 (↑↓循环 + 关系提取)                      │
    ↓                                            │
Phase 5 (推理 + 完整性 + 概念组合)                │
    ↓                                            │
Phase 6 (模块一主入口)                           │
    ↓                                            │
Phase 7 (模块二编码器)                           │
    ↓                                            │
Phase 8 (模块二三主入口)                         │
    ↓                                            │
Phase 9 (模块三完成)                             │
    ↓                                            │
Phase 10 (模块四分层哈希) ───────────────────────┘
    ↓
Phase 13 (管线编排器) ← Phase 11 (ODE) ← Phase 12 (优化器)
    ↓
Phase 14 (IO) → Phase 15 (Rust) → Phase 16 (测试) → Phase 17 (Lean)
```

## 可并行任务

- Phase 2 + Phase 4 可并行 (都只依赖 Phase 1)
- Phase 11 + Phase 12 可并行 (都只依赖 Phase 1 + 各自无互依赖)
- Phase 14 可与 Phase 13 并行 (一个负责 IO, 一个负责编排)
- Phase 17 可在任意时间开始 (不依赖其余代码, 只依赖 Lean 4 工具链)

## 首日推荐执行顺序

```
1. Phase 0-1: 项目骨架 + 类型 (1h)
2. Phase 2-3: 插件 + Tokenizer (2h)
3. Phase 4-6: 模块一完成 (4h)
4. Phase 7-8: 模块二 + 一半模块三 (3h)
```

首日完成后即可运行 `adapter.adapt("测试文本")` 并得到 MathModel。
