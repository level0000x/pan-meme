"""优化文档：压缩 6.17C 冗长数值佐证"""
import re

path = r"c:\Users\xingg\Desktop\知识体系化Wiki\模因\docs\泛模因理论-完整知识库.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 "**数值佐证**（6.17B–6.17C 全面验证）。" 的位置
marker = "**数值佐证**（6.17B–6.17C 全面验证）。"
# 找到 "**凹凸性引理（引理 6.17Cb）■**。" 作为结束标志
end_marker = "**凹凸性引理（引理 6.17Cb）■**。"

idx_start = content.find(marker)
idx_end = content.find(end_marker)

if idx_start < 0:
    print("ERROR: start marker not found")
elif idx_end < 0:
    print("ERROR: end marker not found")
else:
    replacement = """**数值佐证**（紧密）。6.17B 三步不等式全零违规。全轨道 8137 步零真实违反（仅 14 步 ‖Δ‖₁ < 1e-14 处浮点噪声——S4−S6 元组级始终为负）。6.17C 沿 B_sym 主特征向量零违规（min F/‖Δ‖² = 0.860）。D_low 在 950K 次检验零违反（min D/D_low = 1.09）。FCA 域外 500 随机参数 ρ(B_sym) ≤ 0.679——理论稳健。

**r_j 的半解析性**。c_j = Σ_k B_kj ≤ α < 1 解析可证（6.17B）。r_j = Σ_k B_jk ≤ 0.707 ≪ 1 安全裕度 0.293——但解析紧界需 M*（非良基/涌现量），故 ρ(B_sym) 的 AM-GM 界标记半解析。对收敛链无影响（6.18 仅依赖 6.17B）。

**◆ 对"所有 M"的推广**（非收敛必需）。用 D_min 替代 D_low 时 α_min ∈ [0.176, 1.665]——seed 11 超 1（D_min 过保守）。t ≥ 1 版本对收敛链充分。

"""
    new_content = content[:idx_start] + replacement + content[idx_end:]
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✓ 替换完成: 从 {len(content)} 字符 → {len(new_content)} 字符 (减少 {len(content)-len(new_content)} 字符)")

# 2. 也优化 6.17C 核心证明中的 AM-GM 部分
# "由 6.17B，$c_j = \sum..." → 简化为 "由 6.17B 知 c_j < 1（解析）"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_line = "由 6.17B，$c_j = \\sum_k |J_{kj}|\\cdot D^*_k/D_{\\mathrm{low},k} \\le \\alpha < 1$（解析）。行和 $r_j = (D^*_j/D_{\\mathrm{low},j})\\cdot\\sum_k|J_{jk}|$：对 200 种子 $\\max_j r_j = 0.7065$（seed 1），对 FCA 域外 500 组随机参数 $\\max_j r_j = 0.712$。故对任意检验参数：\n\n$$\\rho(B_{\\mathrm{sym}}) \\le \\max_j\\frac{c_j+r_j}{2} \\le \\max\\left(\\frac{0.545+0.707}{2}, \\frac{0.545+0.712}{2}\\right) \\approx 0.629 < 1$$\n\n注意：$c_j$ 界是解析的（6.17B），$r_j$ 界有约 0.3 的安全裕度（max 0.707 vs. 阈值 1.0）——即使最差种子也不会接近 1。$\\rho(B_{\\mathrm{sym}})$ 实际值更小：$\\in [0.149, 0.379]$（max 0.379 at seed 1）——因 $l_1$ 范数界对对称矩阵是保守的。"

new_line = "由 6.17B 知 $c_j < 1$（解析）。行和 $r_j$ 对 200 种子 max = 0.707（安全裕度 0.293），FCA 域外 max = 0.712。AM-GM 界 $\\rho(B_{\\mathrm{sym}}) \\le \\max_j(c_j+r_j)/2 \\le 0.629 < 1$。实际 $\\rho(B_{\\mathrm{sym}}) \\in [0.149,0.379]$（远低于 AM-GM 界——$l_1$ 范数对对称矩阵高度保守）。"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ AM-GM 证明部分压缩完成")
else:
    print("WARNING: AM-GM old_line not found (文件可能已被修改)")

# 3. 优化 6.17Cb 的结尾
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_end = "方向单调性的新证明（l₁ 收缩 → 对称化 $B_{\\mathrm{sym}}$ → Rayleigh-Ritz → AM-GM 界）比先前的局部 Taylor 展开 + 顶点论证更简洁且更强——给出定量下界 $\\min(1-\\rho(B_{\\mathrm{sym}})) = 0.621$（200 种子），仅 $c_j$ 一侧为解析、$r_j$ 一侧以约 0.3 的安全裕度半解析佐证。6.17Cb 保留为独立几何洞察：N 在任意方向的二阶行为由该方向的参数比率固定。∎"
new_end = "证明链：6.17A（精确分解）→ 6.17B（l₁ 收缩+D_low 界）→ B_sym（对称化）→ Rayleigh-Ritz → AM-GM 谱界。仅 c_j 为解析、r_j 半解析（安全裕度 0.293）——对收敛链无影响（6.18 仅依赖 6.17B）。6.17Cb 保留为独立几何洞察：N 的坐标凹凸性由参数比率固定。∎"

if old_end in content:
    content = content.replace(old_end, new_end)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ 6.17C 结尾优化完成")
else:
    print("WARNING: old_end not found")

print("\n全部优化完成！")
