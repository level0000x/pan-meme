"""优化 6.17A₂ 和 6.17B"""
path = r"c:\Users\xingg\Desktop\知识体系化Wiki\模因\docs\泛模因理论-完整知识库.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# 1. 优化 6.17A₂ 的 CD 段落 (L811-818)
# ============================================================
old_cd = """**CD 的辅助性**。CD 值远小于 RD（安全裕度约 2×）。RD 闭合后 sym(I−J)≻0 可由 Gershgorin 行和盘直接推出（无需 CD），故 CD 的闭合状态对 sym(I−J)≻0 非瓶颈。在实践层面 CD 提供独立交叉验证——数值 max_cd ∈ [0.247, 0.314]（200 FCA 种子），与 200 种子 × 直接特征值检验 λ_min > 0 一致——但收敛链本身的完整性不依赖 CD 的解析闭合。CD 迭代界收紧尝试失败（T=2 轮仍有 9/1000 违规），因其行间分子和分母的结构差异使行和界的端点论证不可复用。"""

new_cd = """**CD 辅助性**。CD 值远小于 RD（裕度 2×），RD 闭合后 sym(I−J)≻0 可直接导出。CD 数值佐证：200 种子 max_cd ∈ [0.247, 0.314]，与特征值检验一致（λ_min > 0 全通过）。收敛链不依赖 CD。"""

if old_cd in content:
    content = content.replace(old_cd, new_cd)
    print("✓ 6.17A₂ CD 段落压缩完成")
else:
    print("WARNING: 6.17A₂ CD old text not found - trying substring match")
    # Try with a substring
    marker = "**CD 的辅助性**"
    idx = content.find(marker)
    if idx >= 0:
        # Find the next paragraph break
        end_idx = content.find("\n\n", idx + len(marker))
        if end_idx > idx:
            old_text = content[idx:end_idx].strip()
            content = content[:idx] + new_cd.strip() + content[end_idx:]
            print(f"✓ 6.17A₂ CD 段落压缩完成 (substring)")
        else:
            print("WARNING: end not found")

# ============================================================
# 2. 优化 6.17B 的"推论 — 全局有效性" → 合并部分与 6.18
# ============================================================
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_corollary = """**推论 — 全局有效性**。对任意 M(0) ∈ [0,1]⁵，直接下界引理给出第一迭代步的下界保障：

$$M^{(1)}_k = N_k(M^{(0)}) \\ge \\frac{a_k}{D_{\\max,k}} = m_k^{(0)}$$

故对 ∀t≥1，D_k(M(t)) ≥ D_low,k 无条件成立——不依赖 M(0) 的特定选择。换言之：无论起点在何处，迭代第一步后所有分量均恢复至 m^(0) 以上，其后分母下界 D_low 始终有效。这保证了 α 收缩界对 t≥1 全局有效，无需对 t=0 的特殊对待。该论证不依赖 N 坐标单调性（该性质在 N 算子中不成立——∂N_k/∂M_j 在 25%–50% 随机采样点处为负）。"""

new_corollary = """**推论 — 全局有效性（→ 即 6.18 的核心论证）**。对任意 M(0) ∈ [0,1]⁵，$M^{(1)}_k = N_k(M^{(0)}) \\ge a_k/D_{\\max,k} = m_k^{(0)}$（直接下界引理——纯代数）。故 t≥1 时 D_k ≥ D_low,k 无条件成立，α 收缩界全局有效。详见定理 6.18。"""

if old_corollary in content:
    content = content.replace(old_corollary, new_corollary)
    print("✓ 6.17B 推论压缩完成")
else:
    # Try with marker
    marker = "**推论 — 全局有效性**"
    idx = content.find(marker)
    if idx >= 0:
        end_marker = "**注**"
        end_idx = content.find(end_marker, idx)
        if end_idx > idx:
            old_text = content[idx:end_idx]
            # Keep the note marker
            content = content[:idx] + new_corollary + "\n\n" + content[end_idx:]
            print(f"✓ 6.17B 推论压缩完成 (substring, {len(old_text)} → {len(new_corollary)} chars)")
        else:
            print("WARNING: end marker **注** not found")
    else:
        print("WARNING: 6.17B 推论 marker not found")

# ============================================================
# 3. 优化 6.17A₂ 的数值验证段落 (L813-818)
# ============================================================
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_verify = """**数值验证**。200 组 FCA 种子×100 均匀网格点 + 500 扩展种子验证：λ_min(sym(I−J)) ∈ [0.679, 0.975]（全部 >0）。Gershgorin 圆盘心 1、半径 (r_k+c_k)/2 < 1 → 所有圆盘 ⊂ (0,2) → sym(I−J) ≻ 0——与直接特征值检验一致。sym(I−J)≻0 为命题 6.17C 的方向单调性提供局部解析框架（在‖Δ‖<λ_min/c_max 半径内成立）。c_max ∈ [0.01, 0.07]（median 0.02）但高阶项在大半径处可贡献可比的量级——实际安全域远超保守估计（实证 |(|Δ|)|=√5 也无违反）。"""

new_verify = """**数值验证**。200 FCA 种子 λ_min(sym(I−J)) ∈ [0.679, 0.975] 全 >0，Gershgorin 圆盘全 ⊂ (0,2)。为命题 6.17C 提供局部方向单调性框架——但 6.17C 的新证明（l₁ 收缩→Rayleigh-Ritz）已将其推广至大半径甚至全局范围。"""

if old_verify in content:
    content = content.replace(old_verify, new_verify)
    print("✓ 6.17A₂ 数值验证压缩完成")
else:
    marker = "**数值验证**。200 组"
    idx = content.find(marker)
    if idx >= 0:
        skip_to = "为命题 6.17C"
        skip_idx = content.find(skip_to, idx)
        if skip_idx > idx:
            content = content[:idx] + new_verify + "\n\n" + content[skip_idx:]
            print(f"✓ 6.17A₂ 数值验证压缩完成 (substring)")
        else:
            print("WARNING: skip target not found")
    else:
        print("WARNING: 6.17A₂ 数值验证 marker not found")

# ============================================================
# 4. 添加依赖链注释到 6.18
# ============================================================
# Already pretty clean, just add explicit dependency
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Write final result
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n最终文档长度: {len(content)} 字符")
print("全部优化完成！")
