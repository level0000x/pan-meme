# 001 实验轨迹 CSV 文件

本目录下的 trajectory.csv 是完整的 ODE 五维时间序列数据，每行包含 t, D, B, ρ, R, S 六个字段。

重现实验:

```bash
cd updown_rs
cargo run -- ../../experiments/001-mvp-cross-domain/inputs/domain_biology.txt \
    --text --auto-optimize --t-max 20.0 --max-steps 80000 \
    -o ../../experiments/001-mvp-cross-domain/runs/biology_t20
```