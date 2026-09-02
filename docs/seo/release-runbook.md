# SEO Release And Indexing Runbook

This runbook is for post-merge production release only. Do not run these
external gates from an unmerged feature branch.

## Release Gates

1. Confirm the SEO branch is merged to main.
2. Enable GitHub Pages with GitHub Actions as the source.
3. Watch the Docs and GitHub Pages workflow until both jobs pass.
4. Smoke-test homepage, three core pages, robots.txt and sitemap.xml.
5. Update GitHub description, homepage and ten approved topics.
6. Read GitHub metadata back and compare exact values.
7. Verify Google Search Console ownership and submit sitemap.xml.
8. Record production date and begin weekly observation.

## Gate 1: Confirm Main

Record the merged main commit SHA before any production work.

- Merged main commit SHA:
- Release date:
- Release owner:

## Gate 2: Enable Pages Source

Enable GitHub Pages with GitHub Actions as the source for
`xingwudao/open-xquant`.

When CLI access is not available, the repository owner must complete this in
the authenticated GitHub web interface before continuing.

## Gate 3: Watch Pages Workflow

Run:

```bash
seo_run_id="$(gh run list --workflow docs-pages.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$seo_run_id" --exit-status
```

Capture:

- Successful Pages workflow run URL:

Stop if the Docs or GitHub Pages workflow fails.

## Gate 4: Production Smoke Tests

Run:

```bash
curl -fsS https://xingwudao.github.io/open-xquant/
curl -fsS https://xingwudao.github.io/open-xquant/guide/ai-quant-framework
curl -fsS https://xingwudao.github.io/open-xquant/workflows/strategy-backtest
curl -fsS https://xingwudao.github.io/open-xquant/skills/
curl -fsS https://xingwudao.github.io/open-xquant/robots.txt
curl -fsS https://xingwudao.github.io/open-xquant/sitemap.xml
```

Capture:

- Production deployment URL: `https://xingwudao.github.io/open-xquant/`

Stop if any smoke test fails. Homepage metadata must not be updated before all
production smoke tests pass.

## Gate 5: Update GitHub Metadata

Only run this gate after Gate 4 passes.

Run:

```bash
gh repo edit xingwudao/open-xquant \
  --description "AI 量化研究框架：AI Agent 驱动策略回测、因子研究、稳健性检验、审计报告与实盘交易 | Agentic Quant Research Kernel" \
  --homepage "https://xingwudao.github.io/open-xquant/" \
  --add-topic ai-quant \
  --add-topic quantitative-finance \
  --add-topic quant-research \
  --add-topic ai-agents \
  --add-topic agentic-ai \
  --add-topic backtesting \
  --add-topic factor-research \
  --add-topic algorithmic-trading \
  --add-topic trading-strategy \
  --add-topic python
```

The approved topics are exactly:

- `ai-quant`
- `quantitative-finance`
- `quant-research`
- `ai-agents`
- `agentic-ai`
- `backtesting`
- `factor-research`
- `algorithmic-trading`
- `trading-strategy`
- `python`

Do not add, remove, rename, or reorder the production update without a new
approved contract.

## Gate 6: Read Metadata Back

Run:

```bash
gh repo view xingwudao/open-xquant \
  --json description,homepageUrl,repositoryTopics,url
```

Capture:

- GitHub metadata readback:

Compare the readback against these exact values:

- Description:
  `AI 量化研究框架：AI Agent 驱动策略回测、因子研究、稳健性检验、审计报告与实盘交易 | Agentic Quant Research Kernel`
- Homepage: `https://xingwudao.github.io/open-xquant/`
- Topics:
  `ai-quant`, `quantitative-finance`, `quant-research`, `ai-agents`,
  `agentic-ai`, `backtesting`, `factor-research`, `algorithmic-trading`,
  `trading-strategy`, `python`

If metadata readback fails or differs from the approved contract, stop at this
gate and leave the previous production metadata unchanged where possible.

## Gate 7: Search Console

Verify Google Search Console ownership and submit `sitemap.xml` for:

`https://xingwudao.github.io/open-xquant/sitemap.xml`

When CLI access is not available, the repository owner must complete ownership
verification and sitemap submission in the authenticated Google Search Console
web interface.

Capture:

- Search Console verification status:
- Sitemap submission date:

## Gate 8: Observation

Record the production date and begin weekly observation.

- Production date:
- Weekly observation owner:

Track Search Console impressions, clicks, CTR, average position and indexed
pages weekly. Ranking improvement is an observed outcome over 6 to 8 weeks,
not a build or release claim.
