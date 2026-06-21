"""Small message catalog for report rendering."""

from __future__ import annotations

from typing import Any

_MESSAGES: dict[str, dict[str, Any]] = {
    "zh": {
        "report_title": "研究报告: {strategy_id}",
        "generated": "生成时间",
        "run_id": "运行 ID",
        "not_specified": "未指定",
        "no_benchmark": "未定义基准",
        "no_robustness": "未配置稳健性测试",
        "no_significant_issues": "未发现显著问题。",
        "no_assets": "未登记图表资产。",
        "figure_prefix": "图",
        "attachment": "附件",
        "headings": {
            "decision": "执行结论",
            "hypothesis": "研究假设",
            "strategy": "策略配置摘要",
            "assumptions": "数据与执行假设",
            "metrics": "回测指标",
            "benchmark": "基准比较",
            "assets": "图表资产",
            "reproducibility": "复现性审计",
            "research_bias": "研究偏差审计",
            "robustness": "稳健性测试",
            "failure_modes": "失败模式",
            "next_actions": "下一步",
            "asset_appendix": "资产清单",
        },
        "labels": {
            "universe": "标的池",
            "signal": "信号",
            "portfolio": "组合",
            "execution": "执行",
            "fee": "手续费",
            "slippage": "滑点",
            "initial_cash": "初始资金",
            "price_adjustment": "价格复权",
            "status": "状态",
            "fatal": "致命",
            "warnings": "警告",
            "kind": "kind",
            "source_script": "source_script",
            "source_artifacts": "source_artifacts",
        },
        "decision_explanation": (
            "**REJECT** = 存在致命审计问题，不应继续。\n"
            "**WATCHLIST** = 需要进一步调查后再升级。\n"
            "**PAPER TRADING CANDIDATE** = 通过基础审计，可进入模拟交易评估。"
        ),
        "next_actions": {
            "reject": "- 修复致命审计问题后再重新评估该策略。",
            "watchlist_1": "- 先处理警告项，再考虑进入模拟交易。",
            "watchlist_2": "- 使用成本倍数运行稳健性测试。",
            "promote_1": "- 可以进入模拟交易并持续监控。",
            "promote_2": "- 建立实盘监控和漂移检测。",
        },
    },
    "en": {
        "report_title": "Research Report: {strategy_id}",
        "generated": "Generated",
        "run_id": "Run ID",
        "not_specified": "not specified",
        "no_benchmark": "No benchmark defined",
        "no_robustness": "No robustness tests configured",
        "no_significant_issues": "No significant issues detected.",
        "no_assets": "No chart assets registered.",
        "figure_prefix": "Figure",
        "attachment": "Attachment",
        "headings": {
            "decision": "Executive Decision",
            "hypothesis": "Hypothesis",
            "strategy": "Strategy Spec Summary",
            "assumptions": "Data and Execution Assumptions",
            "metrics": "Backtest Metrics",
            "benchmark": "Benchmark Comparison",
            "assets": "Report Assets",
            "reproducibility": "Reproducibility Audit",
            "research_bias": "Research Bias Audit",
            "robustness": "Robustness Tests",
            "failure_modes": "Failure Modes",
            "next_actions": "Next Actions",
            "asset_appendix": "Asset Appendix",
        },
        "labels": {
            "universe": "Universe",
            "signal": "Signal",
            "portfolio": "Portfolio",
            "execution": "Execution",
            "fee": "Fee",
            "slippage": "Slippage",
            "initial_cash": "Initial Cash",
            "price_adjustment": "Price Adjustment",
            "status": "Status",
            "fatal": "Fatal",
            "warnings": "Warnings",
            "kind": "kind",
            "source_script": "source_script",
            "source_artifacts": "source_artifacts",
        },
        "decision_explanation": (
            "**REJECT** = Fatal audit findings, do not proceed.\n"
            "**WATCHLIST** = Needs further investigation before promotion.\n"
            "**PAPER TRADING CANDIDATE** = Passes basic audits, suitable for paper trading evaluation."
        ),
        "next_actions": {
            "reject": "- Fix fatal audit findings before reconsidering this strategy.",
            "watchlist_1": "- Address warnings before promoting to paper trading.",
            "watchlist_2": "- Run robustness tests with cost multiplier.",
            "promote_1": "- Proceed to paper trading with monitoring.",
            "promote_2": "- Set up live monitoring and drift detection.",
        },
    },
}


def messages(lang: str) -> dict[str, Any]:
    try:
        return _MESSAGES[lang]
    except KeyError as exc:
        raise ValueError(f"unsupported report language: {lang}") from exc
