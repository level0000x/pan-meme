# 泛模因理论验证实验

按验证计划 (docs/verification-plan.md) 逐项执行。每个实验单独一个目录，包含输入数据、运行输出和实验报告。

## 实验清单

| 编号 | 实验 | 状态 | 核心问题 |
|------|------|------|---------|
| 000 | 收敛判据 | 完成 | 收敛成功的操作化定义 |
| 001 | 最小可行验证 | 完成 | H₁ 收敛性 + H₂ 跨领域一致性 + H₃ 可逆性 |
| 002 | 数据量缩放 | **完成** | 四规模 100% 收敛，收敛时间恒定 t≈11 |
| 003 | 跨领域一致性 | 待执行 | 四领域参数变异系数 < 10%？ |
| 004 | 可逆性压力测试 | 待执行 | 什么输入会打破可逆性？ |
| 005 | 消融实验 | 待执行 | 每个阶段真的必要？ |

## 目录结构

```
experiments/
├── README.md                    ← 本文件
├── 000-convergence-criteria/    ← 实验零
│   ├── README.md                ← 实验说明
│   └── results.md               ← 结果报告
├── 001-mvp-cross-domain/        ← 实验一
│   ├── README.md                ← 实验说明
│   ├── inputs/                  ← 输入数据
│   ├── runs/                    ← 运行输出 (按参数分组)
│   └── results.md               ← 结果报告
└── ...
```

## 运行实验

```bash
# 生物演化域 (t_max=20.0)
cd updown_rs
cargo run -- experiments/001-mvp-cross-domain/inputs/domain_biology.txt \
    --text --auto-optimize --t-max 20.0 --max-steps 80000 \
    -o experiments/001-mvp-cross-domain/runs/biology_t20

# AI/技术域 (t_max=20.0)
cargo run -- experiments/001-mvp-cross-domain/inputs/domain_ai.txt \
    --text --auto-optimize --t-max 20.0 --max-steps 80000 \
    -o experiments/001-mvp-cross-domain/runs/ai_t20
```

## 验收标准

| 等级 | 条件 | 当前状态 |
|------|------|---------|
| 证伪 | 收敛率 < 50% 或 参数变异系数 > 30% | — |
| 部分验证 | 收敛率 > 50% 但 < 90% | — |
| 验证 | 收敛率 > 90%，变异系数 < 10%，可逆性 100% | ← 001 实验达到 |
| 强验证 | 满足"验证" + 每领域 ≥ 3 个原型 | 待更多数据 |