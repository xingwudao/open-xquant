# Cyclical EMA Strategy Research

研究目标：在周期性波动的美股标的上，验证 "EMA 趋势跟踪 + LightGBM gating" 是否产生 Sharpe ≥ 1.0 且优于纯 EMA baseline ≥ +0.3 的策略。

## 运行

1. 安装依赖：`uv pip install yfinance lightgbm optuna shap beautifulsoup4 requests pyarrow nbformat`
2. 打开 `research.ipynb`，按 cell 顺序执行
3. 顶部 `REFRESH_DATA = False`（默认复用缓存）；首次或想吃新数据时改 `True` 跑一次

## 输出

- `outputs/candidates.csv` —— 当前 universe + 模型评分 ≥ τ 的候选清单（核心交付物）
- `outputs/model_v0.pkl` —— 训练好的 LightGBM 模型（不入 git）
- `outputs/model_card.md` —— 模型说明 + 性能指标 + 局限
- `outputs/equity_curve_vs_baseline.png` —— vs 纯 EMA baseline 的 equity curve
- `outputs/shap_summary.png` —— top 15 因子的 SHAP 影响图
- `outputs/feature_importance.csv` —— LightGBM 原生 importance
- `outputs/summary_metrics.csv` —— 三层指标汇总

## 设计 + 实施

- 设计文档：`docs/plans/2026-05-07-cyclical-ema-research-design.md`（local-only，不入 git）
- 实施 plan：`docs/plans/2026-05-08-cyclical-ema-research-impl.md`（local-only，不入 git）
