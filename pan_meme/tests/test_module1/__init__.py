"""模块一（输入适配层）测试套件

覆盖范围:
- Tokenizer:       公理1 — 多模态分词与模态调度
- CycleEngine:     公理3 + 前提0 — ↑↓ 循环层级树构建
- RelationExtractor: 前提2 — 关系网络 Ψ 提取
- Reasoner:        关系闭合 — 传递/对称/共现推理产生 Ψ*
- CompletenessChecker: 前提1 — 结构完整性诊断
- InputAdapter:    定理1+2 — 8步浮现管线 I → Ψ → M
"""
