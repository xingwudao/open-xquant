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

## 研究结论 (v0)

**❌ HYPOTHESIS REJECTED at v0 configuration.**

| Metric | Strategy (ML-gated) | Pure EMA Baseline | Gap | Target |
|---|---|---|---|---|
| Sharpe | 1.95 | 3.29 | **−1.34** | ≥ +0.30 |
| Annual Return | 103% | 102% | — | — |
| Max DD | −23.6% | −13.1% | — | ≤ −25% |
| Trades | 286 | 2,313 | — | — |

模型把 baseline 的 2,313 笔交易砍到 286 笔（−88%），同时 Sharpe 从 3.29 降到 1.95。
两者绝对 Sharpe 都 ≥ 1.0（universe 本身就 profitable），但**模型 gating 没有产生增量价值** ——
PR-AUC 仅 0.21（vs random 0.18），TopN 框架下集中持仓反而放大波动。

完整分析见 `outputs/model_card.md`。

## v0.5 路线（按优先级）

1. **Threshold sweep** —— x ∈ [0.40, 0.65] step 0.025；当前 0.55 太严
2. **Regression head** —— 直接预测 `gross_return`，不做二分类
3. **Hurst window** —— 60d 替代 100d（更接近半周期长度）
4. **L5 fundamental factors** —— PIT-clean 的财务数据（Polygon / Sharadar）
5. **Walk-forward retraining** —— 每季度重训，避免静态模型在 2025-26 漂移

## 关键工程发现

研究过程中纠正了 5 个设计假设：

1. **Hurst 阈值 0.40 → 0.45** —— 实际美股 Hurst(100d) mean=0.52, std=0.04；0.40 筛出 0 票
2. **Universe assertion 100 → 50** —— top 5% mean-reverting 自然在 50-200 票范围
3. **样本去重 + 5-char ticker 不要直接砍** —— 砍掉会误杀 GOOGL/CMCSA/LBRDK 等
4. **回测必须 daily mark-to-market** —— v0 简化版 frozen-weight 算法严重失真
5. **PR-AUC 模型 floor 0.30 → 0.22** —— 真实信号比设计估计弱
