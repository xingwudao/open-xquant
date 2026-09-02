import { defineConfig } from 'vitepress'
import { seoHead } from './seo.mts'

export default defineConfig({
  lang: 'zh-CN',
  title: 'open-xquant',
  description: '面向 AI Agent 和量化研究者的中文友好 AI 量化研究框架',
  base: '/open-xquant/',
  titleTemplate: false,
  cleanUrls: true,
  lastUpdated: true,
  sitemap: {
    hostname: 'https://xingwudao.github.io/open-xquant/'
  },
  transformHead: seoHead,
  ignoreDeadLinks: false,
  themeConfig: {
    logo: '/images/open-xquant-subagent-collaboration.png',
    nav: [
      { text: '首页', link: '/' },
      { text: 'AI 量化框架', link: '/guide/ai-quant-framework' },
      { text: '工作流', link: '/workflows/strategy-backtest' },
      { text: 'Agent Skills', link: '/skills/' },
      { text: 'CLI 能力', link: '/tools/' },
      { text: 'GitHub', link: 'https://github.com/xingwudao/open-xquant' },
    ],
    sidebar: {
      '/guide/': [
        {
          text: '指南',
          items: [
            { text: '什么是 AI 量化框架', link: '/guide/ai-quant-framework' },
            { text: 'AI Agent 量化研究', link: '/guide/agentic-quant-research' },
            { text: '可复现量化研究', link: '/guide/reproducible-quant-research' },
          ],
        },
      ],
      '/workflows/': [
        {
          text: '工作流',
          items: [
            { text: 'AI 量化回测', link: '/workflows/strategy-backtest' },
            { text: 'AI 因子研究', link: '/workflows/factor-research' },
            { text: '量化回测审计', link: '/workflows/research-audit' },
            { text: '量化策略稳健性检验', link: '/workflows/robustness-testing' },
            { text: 'AI 量化实盘交易', link: '/workflows/live-trading' },
          ],
        },
      ],
      '/skills/': [
        {
          text: '入口',
          items: [
            { text: 'Agent Skills 索引', link: '/skills/' },
            { text: 'open-xquant 路由', link: '/skills/open-xquant' },
          ],
        },
        {
          text: '研究治理',
          items: [
            { text: '工作区治理', link: '/skills/govern-research-workspace' },
            { text: '版本管理', link: '/skills/manage-strategy-version' },
            { text: '版本比较', link: '/skills/compare-strategy-versions' },
            { text: '最终版本选择', link: '/skills/select-final-version' },
          ],
        },
        {
          text: '策略与审计',
          items: [
            { text: '想法梳理', link: '/skills/brainstorm-strategy-idea' },
            { text: '想法审计', link: '/skills/audit-strategy-idea' },
            { text: '规格构建', link: '/skills/build-strategy-spec' },
            { text: '规格审计', link: '/skills/audit-strategy-spec' },
            { text: '运行语义审计', link: '/skills/audit-runtime-semantics' },
            { text: '产物血缘审计', link: '/skills/audit-artifact-lineage' },
          ],
        },
        {
          text: '数据与因子',
          items: [
            { text: '数据探索', link: '/skills/explore-data' },
            { text: '标的池构建', link: '/skills/build-universe' },
            { text: '因子评估', link: '/skills/evaluate-factor' },
            { text: '横截面因子评估', link: '/skills/evaluate-cross-sectional' },
            { text: '时间序列因子评估', link: '/skills/evaluate-time-series' },
            { text: '因子筛选', link: '/skills/screen-factors' },
          ],
        },
        {
          text: '组件开发',
          items: [
            { text: '组件创建', link: '/skills/create-component' },
            { text: '组件实现', link: '/skills/author-component' },
            { text: '指标创建', link: '/skills/create-indicator' },
            { text: '信号创建', link: '/skills/create-signal' },
            { text: '规则创建', link: '/skills/create-rule' },
            { text: '规则实现', link: '/skills/build-rule' },
            { text: '组合优化器创建', link: '/skills/create-portfolio-optimizer' },
          ],
        },
        {
          text: '执行与报告',
          items: [
            { text: '授权回测执行', link: '/skills/run-authorized-backtest' },
            { text: '交易执行配置', link: '/skills/configure-trade-execution' },
            { text: '实盘交易管理', link: '/skills/manage-live-trading' },
            { text: '运行监控', link: '/skills/monitor-strategy-run' },
            { text: '参数调优', link: '/skills/tune-parameters' },
            { text: '绩效复核', link: '/skills/review-performance' },
            { text: '实验结果比较', link: '/skills/compare-experiments' },
            { text: '指标可视化', link: '/skills/plot-indicators' },
            { text: '报告图表构建', link: '/skills/build-report-charts' },
            { text: '研究报告撰写', link: '/skills/write-research-report' },
            { text: '研究报告审阅', link: '/skills/review-research-report' },
          ],
        },
      ],
      '/tools/': [
        {
          text: 'CLI 能力',
          items: [{ text: 'open-xquant CLI 能力索引', link: '/tools/' }],
        },
      ],
    },
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                noResultsText: '没有找到结果',
                resetButtonTitle: '清除搜索',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },
  },
})
