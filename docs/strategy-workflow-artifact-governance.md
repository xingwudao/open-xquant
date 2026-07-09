# Strategy Workflow Artifact Governance

本文档定义 OpenXQuant multi-Agent 研究工作流的目录治理、角色协作、
skill 边界和新增治理能力。

目标不是让目录更好看，而是让每个研究结论都能回答四个问题：

- 这个结论来自哪个策略版本？
- 这个版本的用户确认、审计、编译、回测和报告是否完整？
- 如果用户反复修改策略，旧版本和新版本的证据链如何同时保留？
- 如果多个版本都跑完了，谁基于什么证据把哪个版本选为最终版本？

## Visual Overview

全局有向图：

![Strategy workflow artifact governance](images/strategy-workflow-artifact-governance.png)

图源文件：

- `docs/images/strategy-workflow-artifact-governance.dot`

## Core Decision

研究目录采用三层模型：

```text
strategy family
  -> strategy version
    -> run attempt
```

含义如下：

- strategy family 是一个研究主题，例如 `global10`。
- strategy version 是一次完整策略语义的不可变候选版本，例如 `v001`。
- run attempt 是同一个版本下的一次执行、鲁棒性、压力测试或复跑。

核心规则：

```text
策略语义变化 => 新 version
同一 SPEC 的执行变化或复跑 => 新 run
跨 run 或跨 version 对比 => 写入 comparisons/
最终版本选择 => 写入 final/
```

根目录不能再作为各阶段产物的杂物区。根目录只承接索引、注册表和跨版本
治理产物。

## Root Layout

建议的策略家族根目录如下：

```text
global10/
  .open-xquant/
    workspace.yaml

  workflow_manifest.json
  current.json
  lineage.json
  experiments.jsonl

  conversations/
    conv_20260707_154103/
      transcript.md
      confirmations.jsonl
      conversation_hash.txt

  components/
    bundles/
      <bundle_id>/
        component_manifest.json
        component_catalog.json
        custom_components/

  versions/
    v001/
    v002/

  comparisons/
    comparisons.jsonl
    cmp_v001_runA_vs_v002_runB/

  final/
    current_final.json
    selection_20260707_180000/
```

Root 文件职责：

- `workflow_manifest.json` 记录 layout schema、runner、workspace、治理策略。
- `current.json` 指向当前活跃 version、phase、run，不保存研究证据本身。
- `lineage.json` 记录 version 父子关系和创建原因。
- `experiments.jsonl` 记录所有完成或失败但值得登记的 run，防止选择性记忆。
- `conversations/` 保存可审计的原始对话和用户确认。
- `components/` 保存可复用的 workspace-local component bundle。
- `comparisons/` 保存跨 run 或跨 version 的对比证据。
- `final/` 保存最终版本选择记录。

## Version Layout

每个 `versions/vNNN/` 是一个策略语义版本。它内部按阶段分区：

```text
versions/v001/
  version_manifest.json
  phase_state.json

  01_brainstorm/
    strategy_idea_brief.json

  02_idea_audit/
    strategy_idea_audit.json

  03_component_authoring/
    component_request.json
    component_manifest.json
    component_catalog.json
    result.json

  04_spec_build/
    strategy_spec.yaml
    component_catalog.json
    spec_build_notes.md
    spec_mapping_notes.md
    spec_mapping_contract.json
    builder_phase_result.json

  05_data_inspection/
    data_inspection_result.json
    data_availability_report.md

  06_spec_audit/
    spec_audit.json
    spec_confirmation_table.md

  07_compile_preview/
    compiled_plan.json
    strategy.py
    spec_hash.txt

  08_runtime_audit/
    runtime_audit.json
    backtest_authorization.json

  09_backtests/
    run_20260707_161018/
      strategy_spec.yaml
      spec_audit.json
      runtime_audit.json
      compiled_plan.json
      strategy.py
      metrics.json
      equity_curve.csv
      trades.csv
      positions.csv
      orders.csv
      target_weights.csv
      execution_assumptions.json
      data_manifest.json
      artifact_hashes.json
      research_bias_audit.json
      reproducibility_audit.json
      robustness.json
      run_log.jsonl

  10_reports/
    run_20260707_161018/
      report_assets/
      research_report.md
      research_report.html
      writer_result.json
      report_review.json
```

重要约束：

- 每个 worker 只能写自己拥有的 phase 目录。
- 后续 phase 只能引用前序 phase 的 immutable artifact。
- 已确认的 SPEC 不能被原地修改。
- 如果用户改变策略语义，创建新 version，而不是覆盖旧 version。
- 如果 builder 修复的是翻译错误，并且用户策略语义未变，可以留在当前
  version 内重建 `04_spec_build/`，但要记录 attempt。
- 如果 spec audit 已经 confirmed，任何策略语义变化都必须新建 version。

## Run Layout

`09_backtests/` 下每个 run 都是不可变执行证据包。

同一 version 下可以有多种 run：

- `primary`：该 version 的主回测。
- `rerun`：同一 SPEC、同一设置下的复现运行。
- `cost_stress`：成本压力测试。
- `robustness_variant`：鲁棒性扰动或样本扰动。
- `extended_data`：同一 SPEC 在更新数据上的再评估。

run 不应该写在根目录 `runs/` 里。兼容旧 CLI 时，可以由 coordinator 将
CLI 输出移动或指定到：

```text
versions/<version_id>/09_backtests/<run_id>/
```

如果 CLI 仍要求 `runs/<run_id>/`，则 `runs/` 只能作为兼容 staging 区。
正式登记前必须归档到 version 目录，并更新 manifest。

## Hash Policy

所有跨 artifact 引用必须记录 hash 类型。

允许的 hash 类型：

- `spec_canonical_hash`：由 `StrategySpec.compute_hash()` 计算。
- `raw_file_sha256`：原始文件字节 hash。
- `normalized_json_sha256`：规范化 JSON hash。
- `conversation_sha256`：原始对话正文 hash。
- `component_bundle_sha256`：component bundle 内容 hash。

禁止事项：

- 禁止只写 `sha256:...` 而不说明 hash 类型。
- 禁止在不同 artifact 中混用 canonical spec hash 和 raw YAML hash。
- 禁止下游 artifact 引用一个后来被覆盖的 root-level 文件。
- 禁止写 `sha256:placeholder`、空字符串 hash 或不可复算 hash。

标准引用格式：

```json
{
  "artifact": "strategy_spec",
  "path": "versions/v001/04_spec_build/strategy_spec.yaml",
  "hash": "sha256:...",
  "hash_type": "spec_canonical_hash",
  "created_by_role": "oxq-strategy-builder-worker",
  "created_at": "2026-07-07T08:00:00Z"
}
```

## Conversation And Confirmation Governance

用户确认不能只存在于 Agent 回复里。`oxq-coordinator` 负责落盘和维护
conversation artifacts；worker 可以读取它们，但不拥有确认事件写入。

必须落盘：

```text
conversations/<conversation_id>/
  transcript.md
  confirmations.jsonl
  conversation_hash.txt
```

`confirmations.jsonl` 每一行记录一个确认事件：

```json
{
  "event_id": "spec-confirmation-1",
  "timestamp": "2026-07-07T08:00:00Z",
  "phase": "spec_confirmation",
  "field_scope": "full_spec_table",
  "user_text": "确认",
  "artifact_path": "versions/v001/06_spec_audit/spec_confirmation_table.md",
  "artifact_hash": "sha256:...",
  "spec_audit_path": "versions/v001/06_spec_audit/spec_audit.json",
  "spec_audit_hash": "sha256:..."
}
```

默认值治理规则：

- Agent 可以提出 candidate value。
- candidate value 不等于 confirmed value。
- 用户必须明确确认 candidate value 或给出替代值。
- audit worker 必须检查默认值是否被确认。
- 未确认默认值阻塞进入 SPEC 或 runtime。

SPEC 审计必须分成两个状态：

- `audit_conclusion: all_pass`：审计角色认为无剩余 blocker。
- `user_confirmation_status: confirmed`：用户确认完整 SPEC 表。

只有两者同时成立，才能进入 compile preview。
`user_confirmation_status: confirmed` 必须引用
`conversations/<conversation_id>/confirmations.jsonl` 中对应事件的路径和 hash；
没有 durable confirmation event 时，即使审计内容 all pass，也仍然停留在
confirmation gate。

## Existing Roles

现有角色继续保留，但写入范围改为 version phase 目录。

`oxq-coordinator`
- 使用 `open-xquant`。
- 负责路由、phase 状态、用户确认、conversation artifact、worker handoff。
- 不写 SPEC、audit、runtime、run、report。
- 新增职责：调用 version governance 和 final governance。

`oxq-strategy-brainstorm-worker`
- 使用 `brainstorm-strategy-idea`。
- 写 `01_brainstorm/strategy_idea_brief.json`。
- 主动解释阶段、追问用户、拉回流程。
- 不写 SPEC。

`oxq-strategy-idea-auditor-worker`
- 使用 `audit-strategy-idea`。
- 写 `02_idea_audit/strategy_idea_audit.json`。
- 审计 brainstorm 是否按阶段收集、解释、确认。
- 不通过时回到 brainstorm。

`oxq-strategy-builder-worker`
- 使用 `build-strategy-spec`。
- 写 `04_spec_build/strategy_spec.yaml` 等。
- 只从 passing idea audit 构建 SPEC。
- 在 `strategy_spec.yaml` 和 `builder_phase_result.json` 记录同一个
  `required_oxq_version`。
- 写 `spec_mapping_notes.md` 和可机审的 `spec_mapping_contract.json`，
  明确外部来源字段到 SPEC、Studio、report 或 unsupported 的边界。
- 不做用户意图追问，不做 provenance audit。

`oxq-component-author-worker`
- 使用 `author-component`，必要时用 `create-*`。
- 写 `03_component_authoring/` 或 root `components/bundles/`。
- 用于 Indicator、Signal、PortfolioOptimizer。
- 对横截面组合权重逻辑，优先尝试 PortfolioOptimizer，而不是 Indicator。
- 对横截面 reusable factor，例如 RPS，可使用实现
  `compute_cross_section` 的 cross-sectional Indicator。

`oxq-data-inspection-worker`
- 使用 `explore-data`。
- 写 `05_data_inspection/`。
- 检查数据目录、覆盖期、warmup、provider 和质量。
- 不下载未经授权的数据。

`oxq-spec-auditor-worker`
- 使用 `audit-strategy-spec`。
- 写 `06_spec_audit/`。
- 先写 `spec_audit.json`。
- 仅当 `audit_conclusion: all_pass` 且等待用户确认或已确认时，写
  `spec_confirmation_table.md`。
- `audit_conclusion: blocked` 时不得写占位 confirmation table。
- `audit_conclusion: all_pass` 但 `user_confirmation_status: pending` 时仍然
  block。
- 用户确认后更新同一个 `spec_audit.json` 为
  `user_confirmation_status: confirmed`。
- 审计 `required_oxq_version` 缺失、空值或和 builder/runner 元数据不一致。
- 不编译、不回测。

`oxq-runtime-auditor-worker`
- 使用 `audit-runtime-semantics`。
- 写 `07_compile_preview/` 和 `08_runtime_audit/`。
- 只有 confirmed spec audit 后才能运行。
- 必须把完整 `strategy.py` 源码展示给用户。

`oxq-runner-worker`
- 使用 `run-authorized-backtest`。
- 写 `09_backtests/<run_id>/` 和 runner result。
- 只读已授权 artifact，不修复上游 artifact。

`oxq-monitor-worker`
- 使用 `monitor-strategy-run`。
- 写 run-local post-run audit artifact，并 append `experiments.jsonl`。
- 负责 reproducibility、research bias、robustness 和 experiment registry。
- 不写报告正文，不选择最终版本。

`oxq-report-writer-worker`
- 使用 `build-report-charts` 和 `write-research-report`。
- 写 `10_reports/<run_id>/` 作为唯一正式 report package。
- 不修改 run artifact。

`oxq-report-reviewer-worker`
- 使用 `review-research-report`。
- 写 `report_review.json`。
- 审核报告语义是否忠于 artifact。

## Proposed New Roles And Skills

需要新增或改造的治理能力如下。

### `govern-research-workspace`

配套角色：`oxq-artifact-governor-worker`

用途：

- 初始化和验证 strategy family layout。
- 读取 `.open-xquant/workspace.yaml`、`workflow_manifest.json`、`current.json`。
- 检查 root 是否有被误写的 phase artifact。
- 检查 phase artifact 是否写在正确目录。
- 检查 hash 类型和引用路径是否一致。
- 仅当 `spec_audit.json` 处于 `audit_conclusion: all_pass`、
  `user_confirmation_status: pending` 或
  `user_confirmation_status: confirmed` 时，检查
  `spec_confirmation_table.md` 是否存在；blocked audit 不要求占位表。
- 检查 root 是否存在 phase artifact 污染，包括
  `strategy_idea_brief.json`、`strategy_idea_audit.json`、
  `data_inspection_result.json`、`data_availability_report.md`、
  `strategy_spec.yaml`、`component_request.json`、`component_manifest.json`、
  `component_catalog.json`、`spec_build_notes.md`、`spec_mapping_notes.md`、
  `spec_mapping_contract.json`、`builder_phase_result.json`、
  `spec_audit.json`、`audit_notes.md`、`spec_confirmation_table.md`、
  `compile_preview/`、`runtime_audit.json`、`compiled_plan.json`、
  `backtest_authorization.json`、`runner_result.json`、`result.json`、
  `research_report.md`、`research_report.html`、`writer_result.json`、
  `report_review.json` 和 `report_assets/`。

输出：

```text
governance/workspace_audit.json
governance/workspace_audit.md
```

### `manage-strategy-version`

配套角色：`oxq-version-manager-worker`

用途：

- 判断用户修改是继续当前 phase，还是新建 version。
- 创建 `versions/vNNN/version_manifest.json`。
- 更新 `lineage.json` 和 `current.json`。
- 冻结已通过 gate 的旧 version。

触发条件：

- 新策略研究开始。
- 用户在 idea audit pass 后改变策略语义。
- 用户在 spec audit confirmed 后改变任何 material field。
- 用户从 comparison 或 report review 结论中选择改策略。

输出：

```text
versions/vNNN/version_manifest.json
versions/vNNN/phase_state.json
lineage.json
current.json
```

### `audit-artifact-lineage`

配套角色：`oxq-lineage-auditor-worker`

用途：

- 审计 version 内 artifact chain 是否完整。
- 审计 run 是否引用同一 confirmed SPEC、runtime audit 和 data manifest。
- 审计 report 是否对应正确 run。
- 审计 final selection 是否只引用 eligible candidate。

输出：

```text
governance/lineage_audit_<timestamp>.json
governance/lineage_audit_<timestamp>.md
```

### `compare-strategy-versions`

配套角色：`oxq-experiment-comparator-worker`

用途：

- 扩展现有 `compare-experiments`。
- 支持 within-version run comparison。
- 支持 cross-version candidate comparison。
- 调用 `oxq backtest compare-runs` 作为 strict comparability gate。
- 对 cross-version 比较，不把 spec hash 不同当作失败，而是作为 spec diff
  证据。

输出：

```text
comparisons/<comparison_id>/
  comparison_manifest.json
  comparability_audit.json
  spec_diff.yaml
  metrics_comparison.json
  comparison_report.md
  figures/
```

### `monitor-strategy-run` role split

配套角色：`oxq-monitor-worker`

当前已有 `monitor-strategy-run` skill，但 role 层通常由 runner 或
monitor/report worker 混合承接。新模型建议拆出独立 monitor worker。

用途：

- 读取完成的 `09_backtests/<run_id>/`。
- 运行或核验 reproducibility audit。
- 运行或核验 research bias audit。
- 运行 robustness。
- append expanded `experiments.jsonl`。
- 只输出 post-run audit 和 registry，不写报告，不做 final selection。

输出：

```text
versions/<version_id>/09_backtests/<run_id>/reproducibility_audit.json
versions/<version_id>/09_backtests/<run_id>/research_bias_audit.json
versions/<version_id>/09_backtests/<run_id>/robustness.json
experiments.jsonl
```

### `select-final-version`

配套角色：`oxq-final-selector-worker`

用途：

- 根据候选 version、primary run、comparison、report review 和用户确认的
  selection policy 选择最终版本。
- 不跑回测，不改 run artifact，不改 report。
- 只写 final governance artifact。

前置条件：

- 每个候选 version 至少有一个 primary run。
- primary run 通过 reproducibility audit。
- research audit 没有 fatal。
- runtime audit pass。
- report review pass 或明确记录为 non-blocking。
- 用户确认 selection policy。

输出：

```text
final/selection_<timestamp>/
  candidate_set.json
  selection_policy.json
  comparison_refs.json
  final_decision.json
  final_decision.md

final/current_final.json
```

`select-final-version` 不能把 winner 当作投资建议。它只能说这个版本是当前
研究证据下的最终研究候选或 paper trading candidate。

## Skill Routing By Situation

新研究开始：

- Coordinator 调用 `manage-strategy-version` 创建 `v001`。
- Coordinator 调用 `brainstorm-strategy-idea`。

用户给出不完整策略想法：

- 使用 `brainstorm-strategy-idea`。
- 继续停留在 earliest incomplete phase。

brainstorm 完成：

- 使用 `audit-strategy-idea`。
- fail 或 block 时回到 brainstorm。

idea audit pass：

- 使用 `build-strategy-spec`。
- 如果需要自定义组件，调用 `author-component`。

SPEC 初稿完成：

- 使用 `explore-data` 检查数据影响。
- 使用 `audit-strategy-spec` 审计 provenance。

SPEC audit all pass 但未确认：

- Coordinator 向用户展示完整 `spec_confirmation_table.md`。
- Workflow 进入 `next_required_phase: user_spec_confirmation`。
- 不允许 runtime audit。

用户确认完整 SPEC 表：

- Coordinator 追加 durable event 到
  `conversations/<conversation_id>/confirmations.jsonl`，记录 `event_id`、
  table path/hash、`spec_audit_path`、pre-confirmation
  `spec_audit_hash`、event line 和 event hash。
- 更新同一个 `spec_audit.json`，设置
  `user_confirmation_status: confirmed`，并写入 `confirmation_event` 引用。
- 使用 `audit-runtime-semantics`。

runtime audit pass：

- Coordinator 写 `backtest_authorization.json`。
- 使用 `run-authorized-backtest`。

run 完成：

- 使用 `monitor-strategy-run`。
- 写或更新 `experiments.jsonl`。
- 需要报告时使用 `write-research-report`。
- 报告完成后使用 `review-research-report`。

用户要求比较两个结果：

- 如果是同 version 复跑，先用 `oxq backtest compare-runs`。
- 如果是跨 version，使用 `compare-strategy-versions`。
- 对比结果写入 `comparisons/`。

用户要求确定最终版本：

- 使用 `select-final-version`。
- 若 selection policy 未确认，先问用户确认。
- 只更新 `final/`，不复制和篡改 run artifact。

用户在任意阶段改策略：

- Coordinator 调用 `manage-strategy-version` 判断：
  - 仍在 brainstorm 未完成：继续当前 version。
  - audit pass 后改变语义：新建 version。
  - confirmed SPEC 后改变 material field：新建 version。
  - 只是复跑或成本压力测试：新建 run。

## Directed Lifecycle

完整流转如下：

```text
User
  -> Coordinator
  -> Version Manager
  -> Brainstorm Worker
  -> Idea Auditor
  -> Strategy Builder
  -> optional Component Author
  -> Data Inspector
  -> Spec Auditor
  -> User SPEC Confirmation
  -> Runtime Auditor
  -> Runner
  -> Monitor
  -> Report Writer
  -> Report Reviewer
  -> Experiment Registry
  -> optional Comparator
  -> optional Final Selector
```

失败流转：

- idea audit block -> brainstorm worker。
- spec audit says idea incomplete -> brainstorm worker。
- spec audit says SPEC mistranslated -> strategy builder worker。
- spec audit all pass but pending confirmation -> user confirmation。
- runtime audit mismatch -> strategy builder worker or framework/component fix。
- runner authorization fail -> runtime audit or spec audit phase。
- report review blocked -> report writer worker。
- final selector lacks eligible candidates -> comparison or report review phase。

## Comparison Governance

比较分两类。

[Within-version comparison]
- Pros:
  - 可以严格比较同一 SPEC 的复跑和压力测试。
  - `oxq backtest compare-runs` 应该 pass。
- Cons:
  - 不能说明不同策略语义谁更好。
- Best for:
  - 复现、成本压力、数据刷新、robustness。
- Risk:
  - 如果 run artifact hash 过期，比较必须失败。

[Cross-version comparison]
- Pros:
  - 可以比较不同策略语义带来的表现差异。
  - spec diff 是核心证据。
- Cons:
  - 不能简单用收益率命名 winner。
  - `spec_hash` 不同是预期，不是错误。
- Best for:
  - v001 与 v002 的策略改动评估。
- Risk:
  - 若成本、数据、验证区间不同，必须明确标为 non-comparable 或
    partially comparable。

跨版本比较报告必须回答：

- 两个版本的策略语义差异是什么？
- 执行、成本、数据、验证窗口是否一致？
- 指标差异可能由哪些 SPEC 差异驱动？
- 哪些结论只能说 association，不能说 causality？
- 是否满足进入 final selection 的候选资格？

## Final Selection Governance

final selection 是治理动作，不是报告写作，也不是 run comparison。

选择前必须有用户确认的 `selection_policy.json`：

```json
{
  "policy_id": "selection_policy_20260707",
  "confirmed_by_user": true,
  "eligible_if": {
    "spec_audit": "confirmed",
    "runtime_audit": "pass",
    "reproducibility_audit": "pass",
    "research_audit_fatal": 0,
    "report_review": "pass"
  },
  "rank_by": [
    "oos_sharpe_ratio",
    "max_drawdown",
    "robustness_status",
    "trade_count"
  ],
  "tie_breakers": [
    "simpler_spec",
    "lower_turnover",
    "lower_cost_sensitivity"
  ]
}
```

Agent 不允许默认 selection policy。即便 Agent 建议 policy，也必须让用户
确认后才能写 final decision。

`final_decision.json` 示例：

```json
{
  "schema_version": 1,
  "status": "selected",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260707_173012",
  "selected_as": "final_research_candidate",
  "selection_policy": "final/selection_20260707_180000/selection_policy.json",
  "comparison_refs": [
    "comparisons/cmp_v001_runA_vs_v002_runB/comparison_report.md"
  ],
  "blocked_candidates": [],
  "created_by_role": "oxq-final-selector-worker"
}
```

`current_final.json` 只做指针：

```json
{
  "selection_id": "selection_20260707_180000",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260707_173012",
  "final_decision_path": "final/selection_20260707_180000/final_decision.json"
}
```

## Workspace YAML Evolution

现有 `.open-xquant/workspace.yaml` 有：

```yaml
paths:
  current_spec: strategy_spec.yaml
  runs_dir: runs
  final_dir: runs/final
  comparisons_dir: comparisons
```

新治理模型建议改为：

```yaml
paths:
  versions_dir: versions
  conversations_dir: conversations
  components_dir: components
  experiment_registry: experiments.jsonl
  comparisons_dir: comparisons
  final_dir: final
  current_manifest: current.json
  lineage_manifest: lineage.json
```

兼容策略：

- `current_spec` 可以保留，但只能指向当前 active version 的 SPEC。
- 禁止长期保留 root-level `strategy_spec.yaml` 作为事实来源。
- `runs_dir` 可以保留给旧 CLI staging，但正式 artifact 要归档到 version。
- `final_dir` 应改为 `final/`，不再使用 `runs/final`。

## Artifact Ownership Rules

每个角色有唯一写入边界：

- Coordinator：handoff、用户确认请求、`confirmations.jsonl`、
  `backtest_authorization.json`。
- Version Manager：`current.json`、`lineage.json`、`version_manifest.json`、
  `phase_state.json`。
- Brainstorm：`01_brainstorm/`。
- Idea Auditor：`02_idea_audit/`。
- Component Author：`03_component_authoring/` 或 `components/bundles/`。
- Builder：`04_spec_build/`。
- Data Inspector：`05_data_inspection/`。
- Spec Auditor：`06_spec_audit/`。
- Runtime Auditor：`07_compile_preview/`、`08_runtime_audit/`。
- Runner：`09_backtests/<run_id>/`。
- Monitor：run-local audit outputs and `experiments.jsonl` append。
- Report Writer：`10_reports/<run_id>/`。
- Report Reviewer：`10_reports/<run_id>/report_review.json`。
- Comparator：`comparisons/<comparison_id>/`。
- Final Selector：`final/selection_<timestamp>/` and `final/current_final.json`。
- Artifact Governor：`governance/*_audit.*`。

任何 worker 如果发现需要写出自己边界之外的 artifact，必须 block 并把任务
交还 coordinator。

## Required Updates To Existing Skills

`open-xquant`
- 增加 version/final governance 路由。
- 当用户修改策略时，先路由到 `manage-strategy-version`。
- 当用户说“最终版”“选一个版本”“对比版本”时，路由到 comparator 或
  final selector。

`brainstorm-strategy-idea`
- 输出写到 version phase。
- 每个 candidate/default 都要求 confirmation evidence。

`audit-strategy-idea`
- 检查 version phase 和 conversation hash。

`build-strategy-spec`
- 禁止写 root `strategy_spec.yaml`。
- 写入 `04_spec_build/strategy_spec.yaml`。
- 所有 result 引用 phase-local path 和 hash type。

`author-component`
- 输出从 root `result.json` 改为 phase-local result。
- 对可复用组件写 root `components/bundles/<bundle_id>/`。

`audit-strategy-spec`
- 写单个 `spec_audit.json`，用 `audit_conclusion` 和
  `user_confirmation_status` 区分 all pass 与 confirmed。
- 仅当审计达到 `audit_conclusion: all_pass`、
  `user_confirmation_status: pending` 或
  `user_confirmation_status: confirmed` 时落盘
  `spec_confirmation_table.md`。
- JSON 中登记的 table path 必须存在；blocked audit 不写 placeholder table。

`audit-runtime-semantics`
- 输出目录固定为 `07_compile_preview/`。
- 禁止把文件名 `compiled_plan.json` 当目录名。

`run-authorized-backtest`
- run 输出固定到 version-local `09_backtests/<run_id>/`。
- run 内必须附带 spec audit、runtime audit、conversation hash 和 artifact hashes。

`monitor-strategy-run`
- 写入 expanded `experiments.jsonl`，记录 `version_id`、`run_role`、`run_path`。

`compare-experiments`
- 升级为可处理 within-version 和 cross-version。
- 输出 `comparison_manifest.json` 和 `comparability_audit.json`。

`write-research-report`
- 报告必须记录 `version_id`、`run_id` 和 source artifact refs。
- 默认写入 `10_reports/<run_id>/`，避免多个 run 的报告互相覆盖。

`review-research-report`
- 检查 report 是否指向正确 version/run。

## Acceptance Criteria

一个研究从开始到最终选择，必须满足：

- `research init` 默认创建 active `v001`，并写入 `current.json`、
  `lineage.json`、`versions/v001/version_manifest.json`、
  `versions/v001/phase_state.json` 和 `01_brainstorm` 到 `10_reports`
  phase 目录。
- `oxq doctor` 对缺失 active version、空 lineage、active version 不在
  lineage、active version 目录缺失，以及 root-level phase artifact
  pollution 发出 governance warning。
- root 没有孤立的 `strategy_idea_brief.json`、`strategy_idea_audit.json`、
  `data_inspection_result.json`、`data_availability_report.md`、
  `strategy_spec.yaml`、`component_request.json`、`component_manifest.json`、
  `component_catalog.json`、`spec_build_notes.md`、`spec_mapping_notes.md`、
  `spec_mapping_contract.json`、`builder_phase_result.json`、
  `spec_audit.json`、`audit_notes.md`、`spec_confirmation_table.md`、
  `compile_preview/`、`runtime_audit.json`、`compiled_plan.json`、
  `backtest_authorization.json`、`runner_result.json`、`result.json`、
  `research_report.md`、`research_report.html`、`writer_result.json`、
  `report_review.json` 或 root-level `report_assets/`。
- 每个 version 都有 `version_manifest.json`。
- 每个 confirmed SPEC 都有 `spec_confirmation_table.md`。
- `spec_audit.json` 内的 `audit_conclusion: all_pass` 和
  `user_confirmation_status: confirmed` 状态可区分。
- compile preview 在 `07_compile_preview/`，不是 `compiled_plan.json/`。
- run artifact 内包含 spec、spec audit、runtime audit 和 hash manifest。
- `experiments.jsonl` 可追溯到 version/run。
- comparison 不写进任一 run 目录。
- final selection 不复制和修改 run artifact，只写引用和决策。
- 所有用户确认都有 conversation artifact。
- 所有跨 artifact 引用都有 path、hash、hash_type。

## Migration For Existing Messy Workspace

对已有 `global10` 这类目录，迁移步骤如下：

1. 运行 `govern-research-workspace` 生成 workspace audit。
2. 根据现有 `strategy_idea_brief.json` 和 `strategy_idea_audit.json` 创建
   `versions/v001/01_brainstorm/` 和 `02_idea_audit/`。
3. 将被审计和回测时使用的 SPEC 归档到 `04_spec_build/`。
4. 将 `compiled_plan.json/` 这种错误目录迁移为 `07_compile_preview/`。
5. 将 run 目录迁移到 `09_backtests/<run_id>/`。
6. 将报告迁移到 `10_reports/<run_id>/`；run-local 报告只能作为明确标记的
   legacy 副本，不能作为正式事实来源。
7. 重建 `experiments.jsonl` 的 version/run 引用。
8. 对缺失的确认表或 hash mismatch 标记为 blocked，不补造证据。

## Non-Goals

本文档不要求：

- 把所有 governance 都下沉进 CLI。
- 让 CLI 负责语义判断。
- 自动替用户选择最终版本。
- 自动补造缺失的用户确认或审计证据。
- 把报告结论当作投资建议。

CLI 负责 deterministic primitives。Agent skill 负责上下文判断、用户确认、
语义审计、版本治理和最终研究候选选择。
