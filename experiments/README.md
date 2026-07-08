# 实验索引

按实验编号排列，每个实验均有独立目录和完整报告。

## 实验列表

| 编号 | 标题 | 状态 | 目的 |
|------|------|------|------|
| 000 | [收敛判据](000-convergence-criteria/) | 基础 | 将"收敛成功"操作化为可测量数字判据，为后续所有实验提供统一验证基准 |
| 001 | [最小可行验证](001-mvp-cross-domain/) | — | 从数据量缩放和跨领域一致性各取一个子任务拼成最小可行验证 |
| 002 | [数据量缩放实验](002-scaling/) | — | 验证 H1：同一领域四个数据量规模下，数据量越大收敛越稳定 |
| 003 | [跨领域一致性实验](003-cross-domain/) | — | 验证 H2：基因/观念/代码/技术四个完全不同领域的演化动力学是否同构 |
| 004 | [可逆性压力测试](004-reversibility-stress/) | — | 验证 H3：用极端输入定位可逆变换的破裂边界 |
| 005 | [消融实验](005-ablation/) | — | 验证四阶段缺一不可——逐个移除阶段检验结果是否退化 |
| 006 | [弱预测验证](006-weak-predictive/) | — | 固定理论默认参数跨四领域运行，检验参数是否从理论中来而非数据过拟合 |
| 007 | [原型多样性验证](007-archetype-diversity/) | ✅ 完成 | 通过 2-Phase Louvain 和轨迹分类器消除原型单一问题 |
| 008 | **[强验证全扫描](008-strong-validation/)** | ✅ PASS | 4 领域 × 3 分辨率，收敛率 100%，原型 3-4 种，满足强验证标准 |

## 结果汇总

### 强验证（实验 008, v4.2）

| 指标 | 阈值 | 实际 | 判定 |
|------|------|------|------|
| 收敛率 | ≥ 90% | 100% | ✓ |
| 原型多样性 | ≥ 3 种/领域 | 3-4 种 | ✓ |
| 低分辨率收敛 | ≥ 90% | 100% | ✓ |

### 原型分布（γ=2.0, t=20）

| 领域 | 社区数 | 原型分布 |
|------|--------|----------|
| biology | 9 | Source, StableCore, Transient |
| code | 6-8 | Source, StableCore, Transient |
| ideas | 7-8 | Source, StableCore, Transient |
| tech | 7-9 | Source, Transient, StableCore |

### 已知缺口

- Decay / Resilient / Oscillatory / Sink 四种原型尚未在实验中观测到，需要更大规模输入数据
- 实验 000-006 的状态需要补录（部分实验撰写于早期版本，需重新跑验）

## 目录结构

```
experiments/
├── README.md                        # 本文件
├── 000-convergence-criteria/        # 收敛判据
├── 001-mvp-cross-domain/            # 最小可行验证
├── 002-scaling/                     # 数据量缩放
├── 003-cross-domain/                # 跨领域一致性
│   └── inputs/                      #   biology/code/ideas/tech 输入文本
├── 004-reversibility-stress/        # 可逆性压力测试
├── 005-ablation/                    # 消融实验
├── 006-weak-predictive/             # 弱预测验证
├── 007-archetype-diversity/         # 原型多样性
└── 008-strong-validation/           # 强验证全扫描
```

## 运行

```bash
# 强验证全扫描（唯一可复现的端到端测试）
cd updown_rs
cargo test --lib experiment_008_strong_validation -- --nocapture
```
