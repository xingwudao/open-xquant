# Strategy Workflow Artifact Governance

## Round 26 Current Contract Registry

Current schemas are closed: decision schema `5` (including `report_revision` and
`selection_request_id`), candidate `3`, policy `3`, comparison `3`, and lineage
schema `3`. Pointer validation accepts decision schema `5` only; schema `4` is
historical and rejected.

Confirmation events require exactly `schema_version`, `event_id`, `phase`,
`timestamp`, `decision`, `selection_request_id`, `policy_hash`, `confirmed_by`,
`producer`, `coordinator`, and `raw_line_hash`; missing, mismatched, stale, or
self-attested provenance is rejected.

Historical refresh is ordered `write -> review -> lineage -> prepare new selection
-> comparison -> resume`. New evidence gets fresh revision IDs, lineage and
candidate hashes, a fresh selection ID, and a fresh comparison ID; the old
selection is never reused; the fresh `comparison_id` is bound to the fresh
selection.

The exact current comparison request is:

```json
{
  "schema_version": 3,
  "mode": "build_selection_comparison",
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selection_policy": {"path": "<final_dir>/selection_20260712_190000/selection_policy.json", "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
  "candidate_set": {"path": "<final_dir>/selection_20260712_190000/candidate_set.json", "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
  "comparison_population": ["v001/runA", "v002/runB"]
}
```

Chart publication starts with an unsealed report revision attempt. A chart retry
uses a fresh `report_revision_id` and seals only after chart completion.
`default_professional_chart_pack` has a versioned canonical requested set;
requested equality must be exact. Omitted charts must be generated or recorded
with a closed skip reason.

本文档定义 OpenXQuant multi-Agent 研究工作流的目录治理、角色协作、
skill 边界和新增治理能力。

目标不是让目录更好看，而是让每个研究结论都能回答四个问题：

- 这个结论来自哪个策略版本？
- 这个版本的用户确认、审计、编译、回测和报告是否完整？
- 如果用户反复修改策略，旧版本和新版本的证据链如何同时保留？
- 如果多个版本都跑完了，谁基于什么证据把哪个版本选为最终版本？

## Version Root Resolution

所有角色和 skill 在解释阶段路径前，必须读取
`.open-xquant/workspace.yaml`。`version_root` 来自
`paths.versions_dir`，仅当该键缺失时默认使用 `versions`；绝对路径、
父目录穿越或解析后逃逸 workspace 的 symlink 必须阻断。随后读取
`<version_root>/<version_id>/version_manifest.json`，并把其中
`phase_paths` 的对应值作为阶段目录的权威来源。例如自定义根
`research_versions` 下，`v003` 的 spec-build phase 应解析为
`research_versions/v003/04_spec_build`，不得写入默认版本树。

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

每个 `<version_root>/<version_id>/` 是一个策略语义版本。它内部按阶段分区：

```text
<version_root>/<version_id>/
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
      candidates/
        report_20260707_170000/
          report_assets/
          research_report.md
          research_report.html
          writer_result.json
          chart_build_result.json
          candidate_manifest.json
      reviews/
        review_20260707_171000/
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
<phase_paths.09_backtests>/<run_id>/
```

如果 CLI 仍要求 `runs/<run_id>/`，则 `runs/` 只能作为兼容 staging 区。
正式登记前必须归档到 version 目录，并更新 manifest。

## Hash Policy

所有跨 artifact 引用都必须遵循其 producer schema 的 path/hash 约束。
Require `hash_type` only on structured references whose schema defines that field。
例如 `spec_confirmation_table` 这类结构化引用必须按 schema 校验
`hash_type`；`spec_hash`、`spec_audit_hash`、`runtime_audit_hash`、
`artifact_hash` 和 `event_hash` 这类标量字段通过必需的 `sha256:<hex>`
值表达算法，不要求相邻的 `hash_type` 字段。

允许的 hash 类型：

- `spec_canonical_hash`：由 `StrategySpec.compute_hash()` 计算。
- `raw_file_sha256`：原始文件字节 hash。
- `normalized_json_sha256`：规范化 JSON hash。
- `conversation_sha256`：原始对话正文 hash。
- `component_bundle_sha256`：component bundle 内容 hash。

禁止事项：

- 禁止结构化引用在其 schema 要求 `hash_type` 时省略该字段。
- 禁止标量 hash 字段省略 schema 要求的算法前缀。
- 禁止在不同 artifact 中混用 canonical spec hash 和 raw YAML hash。
- 禁止下游 artifact 引用一个后来被覆盖的 root-level 文件。
- 禁止写 `sha256:placeholder`、空字符串 hash 或不可复算 hash。

标准引用格式：

```json
{
  "artifact": "strategy_spec",
  "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
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
  "decision": "confirmed",
  "user_text": "确认",
  "artifact_path": "<phase_paths.06_spec_audit>/spec_confirmation_table.md",
  "artifact_hash": "sha256:...",
  "spec_audit_path": "<phase_paths.06_spec_audit>/spec_audit.json",
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
- 必须把完整 `strategy.py` 源码展示给用户；worker 返回
  `strategy_source_code`，Coordinator 也必须把源码内容以 fenced Python
  block 回显给用户，不能只告诉用户文件路径。

`oxq-runner-worker`
- 使用 `run-authorized-backtest`。
- 写 `09_backtests/<run_id>/` 和 runner result。
- 只读已授权 artifact，不修复上游 artifact。

`oxq-monitor-worker`
- 使用 `monitor-strategy-run`。
- 通过 canonical publishers 写 run-local post-run audit artifact，并 append
  configured experiment registry。
- 负责 reproducibility、research bias、robustness 和 experiment registry。
- 不写报告正文，不选择最终版本。

`oxq-report-writer-worker`
- 使用 `build-report-charts` 和 `write-research-report`。
- 写 `10_reports/<run_id>/` 作为唯一正式 report package。
- 不修改 run artifact。

`oxq-report-reviewer-worker`
- 使用 `review-research-report`。
- Current producer 写 `reviews/<review_revision_id>/report_review.json`；direct
  `report_review.json` 仅用于 historical recognition。
- 审核报告语义是否忠于 artifact。

### Historical Schema-1 Handoff Recognition

当 active version 是 `v002`，任何 explicit inactive candidate（例如
`v001/runA`）的 review is missing or stale 时都可走 guarded re-review；这也
包括 stale current-schema `report_review.json`，不得临时切换 active
pointer。Coordinator 必须向 Report Reviewer 传递以下确切 handoff：
Any explicit inactive candidate is eligible only when its review is missing or
stale.
以下 JSON 仅用于识别旧 handoff；任何写入前必须转换成
`candidate_scoped_historical_report_revision`，不能执行 direct-path
overwrite。

```json
{
  "mode": "candidate_scoped_historical_rereview",
  "version_id": "v001",
  "run_id": "runA",
  "current_state_guard": {
    "path": "current.json",
    "active_version": "v002",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "reason": "stale_report_review",
  "requested_by_role": "oxq-coordinator"
}
```

Reviewer 必须先验证 exact field set、`reason` 只能是
`missing_report_review` 或 `stale_report_review`、`current.json` initial
bytes/hash 和 `active_version`。还必须有 direct evidence 证明 exact review
不存在或无法通过 normal current identity/schema/inventory/digest/input/asset
checks；artifact age 和 schema age 都不是 eligibility condition。然后仅通过
`<version_root>/v001/version_manifest.json` 的 manifest-owned phase paths
解析 `v001/runA`。run/report 必须分别是对应 phase 的 direct child，并通过
normal review 的完整 identity、containment、run manifest、report assets 和
pre/post hash checks。Rerun deterministic report QA 后，不得替换 direct
`report_review.json`；必须针对 fresh immutable report revision 原子发布 fresh
schema-version-2 review revision，不得 backfill 旧 payload。Reviewer must not
change `current.json`、phase state、version state 或 active run，并在发布前后
重读 `current.json` 证明 guard bytes 未变化。该模式 does not reactivate
`v001`。

这是 candidate-scoped historical re-review 的兼容路由名称；当前 producer
必须使用 immutable historical report revision workflow。

End-to-end regeneration order 固定为：先完成 candidate-scoped historical
re-review；再 rerun artifact lineage audit for `v001/runA`；再 regenerate every
comparison that cites the old review or lineage evidence；最后 rerun final
selection 到新的 selection directory。Historical re-review 不是 active phase
completion，Version Manager 和 Coordinator 都不得更新 active state。

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
- 创建 `<version_root>/<version_id>/version_manifest.json`。
- 更新 `lineage.json` 和 `current.json`。
- 冻结已通过 gate 的旧 version。

触发条件：

- 新策略研究开始。
- 用户在 idea audit pass 后改变策略语义。
- 用户在 spec audit confirmed 后改变任何 material field。
- 用户从 comparison 或 report review 结论中选择改策略。

输出：

```text
<version_root>/<version_id>/version_manifest.json
<version_root>/<version_id>/phase_state.json
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
<governance_dir>/lineage_audit_<timestamp>.json
<governance_dir>/lineage_audit_<timestamp>.md
```

每个 lineage audit 只审计一个 candidate version/run，并使用 schema version
2。producer、Final Selector 和 normative governance doc 共享以下确切结构：

```json
{
  "schema_version": 2,
  "status": "pass",
  "scope": {
    "version_id": "v002",
    "run_id": "run_20260712_173012"
  },
  "hash_algorithm": "sha256-file-bytes-v1",
  "input_hashes": [
    {
      "path": "<version_root>/v002/version_manifest.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
      "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    {
      "path": "<phase_paths.06_spec_audit>/spec_audit.json",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    {
      "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    {
      "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
      "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/artifact_hashes.json",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/reproducibility_audit.json",
      "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
    },
    {
      "path": "<phase_paths.09_backtests>/run_20260712_173012/research_bias_audit.json",
      "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
    },
    {
      "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "checked_artifacts": [
    "strategy_spec.yaml",
    "runtime_audit.json",
    "reproducibility_audit.json",
    "report_review.json"
  ],
  "blocking_findings": [],
  "warnings": [],
  "next_required_phase": "final_selection"
}
```

These nine paths are the complete mandatory inventory；不得由 producer 根据
“material files”自行增减：

- `<version_root>/<version_id>/version_manifest.json`
- `<phase_paths.04_spec_build>/strategy_spec.yaml`
- `<phase_paths.06_spec_audit>/spec_audit.json`
- `<phase_paths.07_compile_preview>/compiled_plan.json`
- `<phase_paths.08_runtime_audit>/runtime_audit.json`
- `<phase_paths.09_backtests>/<run_id>/artifact_hashes.json`
- `<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json`
- `<phase_paths.09_backtests>/<run_id>/research_bias_audit.json`
- `<phase_paths.10_reports>/<run_id>/report_review.json`

`input_hashes` 必须与上述 independently derived paths exact set equality。
每个 entry 只能包含 `path`/`sha256`。Reject omission、addition、wrong phase、
wrong run、duplicate recorded path、duplicate canonical target、unsafe path、
symlink escape 和 non-exact regular file。发布 audit 前必须对 exact current
bytes 复算完整 SHA-256；`checked_artifacts` 不能放宽 closed set。

`require_current_run_digest()` is not the complete current-evidence gate；无论
helper 当前实现包含哪些 checks，callers must independently validate producer
required inventory 和 transitive bindings。Lineage producer 必须在读取证据前和发布前执行 full
manifest-entry integrity validation：根据 producer schema 派生 required
entries，验证每个 non-metadata entry 的安全 current regular-file target，
并用 producer artifact-specific algorithm 复算。Mutation without a manifest
refresh 必须 block；refresh 后则旧 review/lineage binding 失效。

Report review、lineage、comparison 和 final selection 的每个 pre/post gate
都必须 before evidence consumption and again immediately before publication，
independently invoke `validate_run_artifact_inventory(run_dir)` independent of
the digest-row check。返回的 immutable profile 是 authoritative，并要求
`profile.contract_schema_version == RUN_ARTIFACT_INVENTORY_SCHEMA_VERSION == 1`。
再独立选择 exactly one current digest row，并要求
`digest_row.artifact_inventory == {"schema_version": 1, "profile": profile.name}`。
Profile 只能由 `artifact_hashes.json.schema_version` 派生；runtime-defined
`artifact_hashes_v0_legacy` through `artifact_hashes_v5` 是完整支持集。Omission、
unknown/unbound extension、alias、duplicate、unsafe/stale binding、downgrade 或
profile mismatch 都必须 reject。Digest-row pass 不能替代 executable inventory
call。

Lineage pass 还必须 recursively validate bound `report_review.json` 的 exact
identity、source-run digest、完整 run manifest、exact five `decision_inputs`、
exact four `reviewed_artifacts` 和 exact registered report asset set/hash。
Nested pass status 或 containing file hash 不能替代 transitive validation。

Schema-version-1 lineage audits are historical only。Do not backfill `scope`
or `input_hashes`，不得根据目录名推断或原地修改旧 audit。必须 rerun artifact
lineage audit against exact current manifest-owned inputs，生成 schema version
2。

Incomplete schema-version-2 audit 也不得 append missing hash 或原地修 path；
任何 mandatory input 或 transitive review binding 改变后都必须 regenerate
完整 audit。

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

Historical schema-version-1 compatibility examples follow. They are retained
only so legacy artifacts can be recognized and rejected for final-selection
consumption; they are not a producer target:

```json
{
  "schema_version": 1,
  "comparison_id": "cmp_v001_runA_vs_v002_runB",
  "candidate_identities": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

```json
{
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_evidence": [
    {
      "version_id": "v001",
      "run_id": "runA",
      "selected_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      }
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012",
      "selected_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      }
    }
  ],
  "evidence_hashes": {
    "comparability_audit.json": {
      "path": "<comparisons_dir>/<comparison_id>/comparability_audit.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics_comparison.json": {
      "path": "<comparisons_dir>/<comparison_id>/metrics_comparison.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "spec_diff.yaml": {
      "path": "<comparisons_dir>/<comparison_id>/spec_diff.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "comparison_report.md": {
      "path": "<comparisons_dir>/<comparison_id>/comparison_report.md",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "figures": [
      {
        "path": "<comparisons_dir>/<comparison_id>/figures/metrics_bar.png",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      }
    ]
  }
}
```

Do not backfill these artifacts. Regenerate the comparison from the exact
current candidate set and transitive evidence as schema version 2.


For `build_selection_comparison`，`comparison_manifest.json` 必须是
schema-version-2 comparison manifest，并包含以下 selection binding 和
versioned candidate identity envelope；producer 不能省略或推断这些 fields：

```json
{
  "schema_version": 2,
  "comparison_id": "cmp_v001_runA_vs_v002_runB",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "candidate_identities": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

Manifest 的 exact `selection_id` 和 exact `{path, sha256}` candidate-set
reference 必须等于 accepted request 和 current candidate-set bytes。即使两个
selection 具有相同 identity，也必须 reject cross-selection substitution。
`selection_policy` 必须等于 accepted request 和 candidate set 的 exact
reference；重读 schema-version-2 policy bytes，验证 user confirmation 和
exact selection id，并 reject stale or cross-selection policy。
`candidate_identities` 至少包含两个无重复 entry；每个 entry 只有
`version_id` 和 `run_id`，并与已验证 registry row 及 manifest-resolved
direct run directory 一致。`comparison_id` 必须等于 direct parent directory
name。

同一个 schema-version-2 manifest 还必须包含以下完整 evidence fragment：

```json
{
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_evidence": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    },
    {
      "ordinal": 1,
      "identity": {
        "version_id": "v002",
        "run_id": "run_20260712_173012"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        "scope": {
          "version_id": "v002",
          "run_id": "run_20260712_173012"
        }
      }
    }
  ],
  "evidence_hashes": {
    "comparability_audit.json": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/comparability_audit.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics_comparison.json": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/metrics_comparison.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    },
    "spec_diff.yaml": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/spec_diff.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "comparison_report.md": {
      "path": "<comparisons_dir>/<selection_id>/<comparison_id>/comparison_report.md",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "figures": [
      {
        "path": "<comparisons_dir>/<selection_id>/<comparison_id>/figures/metrics_bar.png",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      }
    ]
  }
}
```

Candidate evidence must equal the exact ordered projection of complete
candidate-set entries selected by `comparison_population`，包括 original
ordinal、identity、`primary_run` 和 complete `lineage_audit`；`selected_run`
alias 无效。Identity projection 必须与 `candidate_identities` exact equality。
Producer 和 consumer 都要 independently recompute 每个 current run digest、
完整 lineage-v2 closed set、transitive report-review binding 和 lineage audit
file hash。`evidence_hashes` 必须含 exact four named output files；即使是
within-version comparison 也必须生成 deterministic `spec_diff.yaml`。
`figures` 必须等于该目录的 exact current regular-file set；目录不存在时才
能为 `[]`。

Identity-only manifest 即使 identity 正确也无效。缺少
`candidate_evidence`、`evidence_hashes`、任一 exact output 或 exact figure
binding，或者包含 omitted/extra/duplicate/stale/unsafe evidence，都必须
reject。不得 backfill legacy manifest；必须 regenerate the comparison from
current candidate evidence。

Schema-version-1 comparison manifests are historical only for final-selection
consumption because they lack the exact selection and candidate-set binding。
不得 infer 或 backfill；必须从 exact current candidate set 和 transitive
evidence regenerate 新的 schema-version-2 comparison manifest。Missing、
stale、path-only 或 mismatched binding 同样无效。非 selection comparison
不得伪造 selection binding，也不能作为 final-selection comparison ref。

### `monitor-strategy-run` role split

配套角色：`oxq-monitor-worker`

当前已有 `monitor-strategy-run` skill，但 role 层通常由 runner 或
monitor/report worker 混合承接。新模型建议拆出独立 monitor worker。

用途：

- 读取完成的 `09_backtests/<run_id>/`。
- 按以下 exact commands 发布 reproducibility、research bias 和 robustness：

```bash
uv run oxq audit reproducibility "$RUN_DIR" --json --publish
uv run oxq audit research "$RUN_DIR" --json --publish
uv run oxq robustness run "$RUN_DIR" --json
```

`RUN_DIR` 是 manifest-resolved direct run directory。`--json` is response
formatting only。`--publish` is the audit publication contract，并原子 publish
和 bind canonical audit artifact。Robustness self-publishes；robustness needs no
redirection or extra publish flag。Shell redirection into governed artifacts is
invalid。

- 每个 canonical publisher 后 require current run digest。
- 最后 append expanded configured experiment registry，并再次 require current
  run digest；随后只读 handoff。
- 需要 regeneration 时按 reproducibility、research、robustness、experiment
  registration 的 dependency order 完整重跑。
- 只输出 post-run audit 和 registry，不写报告，不做 final selection。

输出：

```text
<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json
<phase_paths.09_backtests>/<run_id>/research_bias_audit.json
<phase_paths.09_backtests>/<run_id>/robustness.json
<experiment_registry>
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
- report review has status exactly `pass`。
- report review has verdict exactly `consistent`。
- report review has `blocking_findings` exactly empty。
- report review has `required_report_edits` exactly empty。
- report review has `errors` exactly empty。
- 用户确认 selection policy。

输出：

```text
final/selection_<timestamp>/
  candidate_set.json
  selection_policy.json
  comparison_refs.json
  final_decision.json

final/current_final.json
```

Final selection uses one staged coordinated request. The normal sequence is
`prepare_selection` -> `candidate_set_ready` ->
`build_selection_comparison` -> `comparison_ready` -> `resume_selection`, with
the same `selection_id` and same exact candidate-set reference at every handoff.
The following exact schema-version-2 request envelope is retained for
historical recognition only; current production uses the Round 25
schema-version-3 envelope:

```json
{
  "schema_version": 2,
  "mode": "prepare_selection",
  "selection_id_policy": {
    "source": "generated",
    "selection_id": null
  },
  "selection_policy": {
    "source": "confirmed_payload",
    "payload": {
      "schema_version": 1,
      "confirmed_by_user": true,
      "confirmation": {
        "source_conversation": "conversation://final-selection-policy",
        "confirmed_at": "2026-07-12T18:00:00Z"
      },
      "eligible_if": {
        "spec_audit": "confirmed",
        "runtime_audit": "pass",
        "reproducibility_audit": "pass",
        "research_audit_fatal": 0,
        "report_review": "pass"
      },
      "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
      "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
    },
    "reference": null
  },
  "candidate_population": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    },
    {
      "ordinal": 1,
      "identity": {
        "version_id": "v002",
        "run_id": "run_20260712_173012"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        "scope": {
          "version_id": "v002",
          "run_id": "run_20260712_173012"
        }
      }
    }
  ]
}
```

The envelope and nested objects have exact key sets. There is no implicit
default for `selection_id_policy.source`. `source: generated` requires
`selection_id: null`; `source: provided` requires one valid non-empty id
whose selection directory does not exist and cannot act as a resume alias.

Selection ids have one normative grammar:
`\Aselection_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z`. Generated and provided ids use
the same grammar and form one normal direct-child component. Separators,
backslashes, dot segments, absolute forms, drive-qualified forms, Unicode, and
longer values are invalid. Canonicalize `<final_dir>` without symlink
components before allocation; the candidate's resolved parent must equal the
canonical `<final_dir>` exactly, and there may be no symlink parent. Allocate
with exclusive atomic `mkdir`, never an existence check followed by creation.
A provided collision is rejected. A generated collision retries with a fresh
generated id under the final-selection lock and never opens, resumes, removes, or reuses the existing
directory.

Candidate ordinal, identity, primary run, and lineage audit input have exact
ordered equality with the published candidate set. Validation must not
rediscover, add, omit, replace, deduplicate, or reorder the population.

`selection_policy` 是 closed union。`source: confirmed_payload` 必须携带上面
exact user-confirmed payload 且 `reference: null`；
`source: hash_bound_reference` 必须是 `payload: null` 和 exact current
`{path, sha256}` source reference。Selector must not infer policy fields。
必须 reject unsafe、malformed、unconfirmed、stale 或 cross-selection policy。
Selector 绑定 exact selection id，并 atomically publishes it inside the
generated selection directory as schema-version-2 `selection_policy.json`，再
把 exact reference 写入 schema-version-2 candidate set。
这个 `{path, sha256}` form 是 hash-bound source reference；Selector
atomically publishes it inside the generated selection directory。

唯一 alternative policy input exact shape：

```json
{
  "source": "hash_bound_reference",
  "payload": null,
  "reference": {
    "path": "<conversations_dir>/<conversation_id>/confirmations.jsonl",
    "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
  }
}
```

The selector's preparation result is exactly:

```json
{
  "schema_version": 2,
  "status": "candidate_set_ready",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_refs": [],
  "next_action": "compare_then_resume",
  "blocking_findings": []
}
```

This is a normal nonterminal handoff, not blocked or failed. In this state the
selector must not write `final_decision.json` and must not update
`current_final.json`. Exactly one candidate 使用
`next_action: resume_selection` 和 `comparison_refs: []`；Coordinator must not
invoke the comparator；Final Selector 先 durable persist singleton ledger，再直接
用 empty comparison refs resume。Two or more
candidates 使用 `next_action: compare_then_resume`。Comparator 只接受至少两个
unique candidates。The coordinator sends this exact multi-candidate comparator
request:

```json
{
  "schema_version": 2,
  "mode": "build_selection_comparison",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_population": [
    {
      "version_id": "v001",
      "run_id": "runA"
    },
    {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    }
  ]
}
```

The comparator revalidates the exact candidate set and returns exactly:

```json
{
  "schema_version": 2,
  "status": "comparison_ready",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_ref": {
    "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "blocking_findings": []
}
```

Without a new user request, the coordinator collects ready refs and invokes the
selector with exactly:

```json
{
  "schema_version": 2,
  "mode": "resume_selection",
  "selection_id": "selection_20260712_180000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ]
}
```

Resume must reuse the same immutable candidate set and selection directory; it
must not regenerate either. It must persist and re-read the exact
comparison-reference ledger before ranking or decision publication. Every selector result has exactly
`schema_version`, `status`, `selection_id`, `candidate_set`, `comparison_refs`,
`selection_policy`, `next_action`, and `blocking_findings`. Missing or incomplete comparison
coverage returns `status: blocked` with `next_action: resume_selection` and can
resume the same selection after remediation. Invalid candidate-set or
transitive candidate evidence returns `status: blocked` with
`next_action: restart_selection` and requires a new preparation. In all
nonterminal, blocked, and failed states, the prior `current_final.json` remains
unchanged.


This protocol covers a complete one-candidate direct-resume handoff and
complete two-candidate and three-candidate handoffs from
request through pointer publication. Two candidates use their full ordered
projection. Three candidates may use one full comparison or connected A-B/B-C
projections, while every handoff preserves the request order, selection id,
candidate-set reference, and exact candidate evidence.

### Comparison Reference Ledger Publication

The Final Selector is the sole producer of `comparison_refs.json`. The
coordinator and router never write `comparison_refs.json`; they only validate
and route handoff values, and they must not write `comparison_refs.json` under
any retry path. Selector writes the ledger in the existing selection
directory at `<final_dir>/<selection_id>/comparison_refs.json`. This producer
step must not allocate a new selection id, redirect to another selection, or
let an unlocked coordinator filesystem write stand in for selector
publication.

The two branches are normative:

1. For exactly one candidate, `prepare_selection` must persist the literal
   UTF-8 bytes `[]`, with no BOM, whitespace, or trailing newline, before
   returning `candidate_set_ready`. This occurs after policy and candidate-set
   publication, under `final-selection.lock`, and before direct
   `resume_selection`.
2. For multiple candidates, Coordinator collects exact `comparison_ref`
   objects from validated `comparison_ready` results. It uses comparator
   dispatch order, not completion order, requires the same `selection_id`,
   selection-policy reference, and exact candidate-set reference in every
   result, and passes that exact ordered array without alteration in
   `resume_selection`. Before ranking or writing `final_decision.json`, Final
   Selector validates every reference and complete coverage against that same
   `selection_id` and exact candidate-set reference, then persists the exact
   ordered array under `final-selection.lock`. An incomplete or invalid array
   blocks before ledger creation.

The singleton ledger is exactly:

```json
[]
```

The multi-candidate ledger for the running example is exactly:

```json
[
  {
    "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  }
]
```

Encode a non-empty ledger as compact UTF-8 JSON with no BOM or trailing
newline, preserving array order and each entry's `path`, then `sha256`, key
order. Before reading or publishing, revalidate canonical `<final_dir>`, the
selection directory as its non-symlink direct child, and the target as the one
safe workspace-relative direct regular file named `comparison_refs.json`. If
the target exists, use `lstat` and reject a symlink, non-regular file, different
canonical parent, or aliased target.

While holding the precomputed workspace `final-selection.lock`, create a
restrictively permissioned same-directory temporary regular file with
exclusive no-follow creation, write all canonical bytes, flush and `fsync` the
temporary file, perform atomic `os.replace`, and `fsync` the selection
directory before reporting publication. Clean up only the disposable temporary
file on failure and leave `current_final.json` unchanged.

Retries are immutable and idempotent. If the target is absent, publish it. If
it already contains the exact expected canonical bytes, fully revalidate it,
sync the file and parent directory, and treat the retry as an idempotent no-op.
If it contains a stale or different array, malformed bytes, or a semantically
equal but non-canonical encoding, block or fail that selection; the selector
must not overwrite it and must not allocate a new selection id as part of the
retry.

After either branch publishes or accepts the ledger, `resume_selection` must
re-read its persisted bytes from the validated direct regular file, require
canonical encoding, parse and fully validate the array, and require request
`comparison_refs` to equal the persisted array exactly. Repeat this read and
equality check immediately before decision publication.
`final_decision.comparison_refs` must equal that same persisted array exactly.

On `resume_selection`, acquire or reacquire `final-selection.lock` before this
ledger read and hold the same lock continuously through decision and pointer
publication. Do not release it between multi-candidate ledger publication or
idempotent acceptance, the required re-read, final-decision publication, and
atomic pointer replacement.

Pointer-time validation re-reads `comparison_refs.json` and requires its bytes,
parsed array, and transitive comparison evidence to remain current and equal
the validated decision array. `current_final.json` does not duplicate the
array; its hash-bound decision reference transitively binds the same array only
after this equality check passes. Mismatch blocks pointer replacement and
leaves the prior pointer byte-for-byte unchanged.

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
- Coordinator then routes back to `oxq-spec-auditor-worker`; the
  `audit-strategy-spec` updates `spec_audit.json`，设置
  `user_confirmation_status: confirmed`，并写入 `confirmation_event` 引用。
- 使用 `audit-runtime-semantics`。

runtime audit pass：

- Coordinator 已经向用户回显完整 `strategy.py` 源码，而不仅是
  `<phase_paths.07_compile_preview>/strategy.py` 路径。
- Coordinator 写 `backtest_authorization.json`。
- 使用 `run-authorized-backtest`。

run 完成：

- 使用 `monitor-strategy-run`。
- 写或更新 `experiments.jsonl`。
- monitor pass 后自动使用 `write-research-report`，不得停在 backtest 或
  monitor 结果等待新提示。
- Coordinator 传入 `chart_decision: default_professional_chart_pack`。
- 报告默认必须先用 `build-report-charts` 生成并注册专业图表包，不询问用户
  是否需要图表，也不得因为用户未主动要求图表就省略。
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

workspace classifier 必须与 CLI 完全一致：当且仅当
`workflow.layout == version_governed` 或 `paths.versions_dir` 键存在时，
workspace 才是 version-governed；否则是 legacy。仅存在
`.open-xquant/workspace.yaml` 不会把 legacy workspace 变成 governed。
classifier 先按 key presence 判定 governance，再验证 value：只要
`paths.versions_dir` key 存在，malformed value 就是 invalid governed，绝不
回退成 legacy。必须要求 non-empty safe workspace-relative string，并拒绝
absolute path、`..` traversal 和 canonical symlink escape。

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

## Immutable Selection Comparison Output

For `build_selection_comparison`, the only output root is
`<comparisons_dir>/<selection_id>/<comparison_id>/`, an immutable
selection-scoped directory. A normal manifest path is
`<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json`.
Create the directory exclusively before any output write and reject an existing
output directory. Never overwrite, delete, merge, or repair comparison evidence,
especially evidence reachable from a prior `current_final.json`. Hash exact
final bytes into the schema-version-2 comparison manifest only for historical
recognition; current production uses schema version 3. A remediable retry
uses a fresh `comparison_id` under the same `selection_id` and keeps the same
policy/candidate-set binding; `restart_selection` allocates a new selection and
comparison scope.

## Historical Schema-2 Final Selection Governance

本节保留 schema-version-2 policy/candidate/decision examples 以识别历史
artifact；current producer contract 以 Round 25 addendum 为准。final selection
是治理动作，不是报告写作，也不是 run comparison。

选择前必须有用户确认的 policy payload。Selector 在 generated selection
directory 内拥有并原子发布 selection-bound schema-version-2
`selection_policy.json`：

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "policy_source": {
    "source": "confirmed_payload",
    "reference": null
  },
  "policy_payload": {
    "schema_version": 1,
    "confirmed_by_user": true,
    "confirmation": {
      "source_conversation": "conversation://final-selection-policy",
      "confirmed_at": "2026-07-12T18:00:00Z"
    },
    "eligible_if": {
      "spec_audit": "confirmed",
      "runtime_audit": "pass",
      "reproducibility_audit": "pass",
      "research_audit_fatal": 0,
      "report_review": "pass"
    },
    "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
    "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
  }
}
```

Inline request 的 `policy_payload` 必须与 exact user-confirmed payload object
equality；hash-bound source request 则把 exact `{path, sha256}` 写入
`policy_source.reference`，并要求 payload 与 source bytes 解析结果 equality。
Agent 不允许默认或推断 selection policy。必须 reject stale or
cross-selection policy，且 exact selection id、policy reference must equal 于
candidate set、comparison v2、final v4 的绑定。Selector atomically publishes
it inside the generated selection directory；其他角色不得写该 artifact。

The historical decision uses schema version 3 and the historical pointer uses
schema version 2. The following compatibility examples are
historical compatibility examples only. They must not be emitted, selected, or
backfilled under the current contract:

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_180000",
  "status": "selected",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "selected_as": "final_research_candidate",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selected_run": {
    "path": "<phase_paths.09_backtests>/run_20260712_173012",
    "digest": "sha256:1111111111111111"
  },
  "report_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.md",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.html",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/writer_result.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "report_review": {
    "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "lineage_audit": {
    "path": "<governance_dir>/lineage_audit_20260712_175500.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "scope": {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    },
    "input_hashes": [
      {
        "path": "<version_root>/v002/version_manifest.json",
        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      },
      {
        "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
        "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
      },
      {
        "path": "<phase_paths.06_spec_audit>/spec_audit.json",
        "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
      },
      {
        "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
        "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
      },
      {
        "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/artifact_hashes.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/reproducibility_audit.json",
        "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/research_bias_audit.json",
        "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
      },
      {
        "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      }
    ]
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "blocked_candidates": [],
  "blocking_findings": [],
  "created_by_role": "oxq-final-selector-worker"
}
```

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "final_decision": {
    "path": "<final_dir>/selection_20260712_180000/final_decision.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Historical `final_decision.json` 使用 schema version 4；current production
uses schema version 5。Old selected 示例必须与 Final
Selector contract 保持结构一致：

```json
{
  "schema_version": 4,
  "selection_id": "selection_20260712_180000",
  "status": "selected",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "selected_as": "final_research_candidate",
  "hash_algorithm": "sha256-file-bytes-v1",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "selected_run": {
    "path": "<phase_paths.09_backtests>/run_20260712_173012",
    "digest": "sha256:1111111111111111"
  },
  "report_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.md",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/research_report.html",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/run_20260712_173012/writer_result.json",
      "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "report_review": {
    "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "lineage_audit": {
    "path": "<governance_dir>/lineage_audit_20260712_175500.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "scope": {
      "version_id": "v002",
      "run_id": "run_20260712_173012"
    },
    "input_hashes": [
      {
        "path": "<version_root>/v002/version_manifest.json",
        "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
      },
      {
        "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
        "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
      },
      {
        "path": "<phase_paths.06_spec_audit>/spec_audit.json",
        "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
      },
      {
        "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
        "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
      },
      {
        "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
        "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/artifact_hashes.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/reproducibility_audit.json",
        "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
      },
      {
        "path": "<phase_paths.09_backtests>/run_20260712_173012/research_bias_audit.json",
        "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
      },
      {
        "path": "<phase_paths.10_reports>/run_20260712_173012/report_review.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
      }
    ]
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "comparison_refs": [
    {
      "path": "<comparisons_dir>/selection_20260712_180000/cmp_v001_runA_vs_v002_runB/comparison_manifest.json",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    }
  ],
  "blocked_candidates": [],
  "blocking_findings": [],
  "created_by_role": "oxq-final-selector-worker"
}
```

`selection_id` 和 candidate-set reference 必须与 published candidate set
相等。Selected `{version_id, run_id}` 必须在该 set 中 exactly once；
`selected_run` 必须等于该 entry 的 `primary_run`，`lineage_audit` 必须等于该
entry 指向并完整验证过的 audit。不得从 policy、comparison prose 或 unbound
registry row 推断 selected identity/evidence。

上述 selected payload 只能包含示例中的确切 top-level keys，不得增加
path-only alias。`selected_run.digest` 必须来自 `run_digests.jsonl` 中
exactly one valid matching `run_id` row，并使用当前 `artifact_hashes.json`
按 producer canonical JSON algorithm 复算。Zero 或 multiple matching rows
都必须 block；do not use file order or choose a last matching row。该检查与
`require_current_run_digest()` 的 cardinality/equality semantics 一致，但
selector 仍须 independently derive required entries，不能用 helper result
替代 full manifest-entry integrity validation。
Selector 必须对每个 candidate 的所有 non-metadata manifest entries 用
producer artifact-specific algorithm 复算。Mutation without a manifest refresh
必须 block；refresh 后旧 review、lineage、comparison 和 selection binding
全部失效。
`report_artifacts` 必须复制已接受 `report_review.json` 中的三个 path/hash
pair，并对当前文件逐一复算 full SHA-256；随后再分别 hash 当前
`report_review.json` 和已确认 `selection_policy.json` 的确切 bytes。

Historical `candidate_set.json` uses schema version 2，并且仅用于兼容识别；
current schema version 3 见 Round 25 addendum。其 old exact schema 如下：

```json
{
  "schema_version": 2,
  "selection_id": "selection_20260712_180000",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_180000/selection_policy.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "candidates": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA.json",
        "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    },
    {
      "ordinal": 1,
      "identity": {
        "version_id": "v002",
        "run_id": "run_20260712_173012"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/run_20260712_173012",
        "digest": "sha256:2222222222222222"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v002_run_20260712_173012.json",
        "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        "scope": {
          "version_id": "v002",
          "run_id": "run_20260712_173012"
        }
      }
    }
  ]
}
```

至少一个 candidate；top-level、candidate 和 nested object 都只能包含示例
中的 exact keys。`ordinal` 是 zero-based array position。Candidate identity
必须 unique，并与 Coordinator 明确传入的 ordered selection population exact
ordered equality；registry discovery 只能验证，不能 omit、add、replace、
deduplicate 或 reorder。`primary_run` 必须绑定 manifest-resolved direct run 和
current producer digest；`lineage_audit.scope` 必须与 `identity` 相等，并通过
complete lineage-v2 validation。

Candidate-set policy reference must equal preparation result 和 current policy
artifact；policy 的 exact selection id 必须等于 candidate-set selection id。
Reject stale or cross-selection policy。

完整 payload 和 transitive evidence 通过后，在 ranking 前原子发布并对 exact
bytes 计算 full SHA-256。`final_decision.json` 和 `current_final.json` 必须包含
同一个 exact `{path, sha256}` reference。Revalidate `candidate_set.json`
immediately before final-decision publication，然后 revalidate it again
immediately before current-pointer publication；两次都必须重新检查 current
bytes、schema、selection id、order、identities、primary runs、lineage scopes 和
transitive hashes。Do not mutate or backfill an existing candidate set；missing、
stale、path-only、aliased、reordered 或 post-hash mutation 都必须 block。

`candidate_set.json` 是 lineage audit path 的 selection manifest。每个
candidate 必须有 exactly one lineage audit reference；不得 glob 或按最新
timestamp 推断。该 path 必须指向 canonical `<governance_dir>` 的一个 direct
regular JSON file，不能 symlink escape。Final Selector 必须验证 audit schema
version 2、`status: pass`、empty `blocking_findings`、exact candidate scope，
并 independently derive normative nine-path inventory，require exact set
equality，reject duplicate canonical target，并对每个 `input_hashes` current
file 复算 hash。随后必须执行 complete lineage-v2 validator，recursively
revalidate bound `report_review.json` 的 source run、full run manifest、exact
five decision inputs、exact four reviewed artifacts 和 exact registered report
assets。不得只信 `status: pass` 或 lineage-audit file hash。Zero/multiple
audit、stale input、wrong-candidate scope、missing input 或 path ambiguity 都
必须 reject。
通过后 hash exact current audit bytes，并把 `path`、`sha256`、`scope` 和
`input_hashes` exact copy 到 `final_decision.lineage_audit`。
这是 one unambiguous candidate-set-manifest-owned audit path。

report review 只有在 `status` exactly `pass`、`verdict` exactly
`consistent`，且 `blocking_findings`、`required_report_edits`、`errors`
全部 exactly empty 时才 eligible。五个条件是同一个 cross-field
invariant，不能用“non-blocking”推断替代。

`comparison_refs.json` 必须是上面 Final Selector 发布的 canonical ledger，且
request `comparison_refs`、persisted array 和
`final_decision.comparison_refs` 必须完全相等。非空 array
中的每个 entry 只能包含 `path` 和 `sha256`。`path` 必须是安全的
workspace-relative path，不得包含 `..`，并在 canonical resolution 后指向
`<comparisons_dir>/<selection_id>/<comparison_id>/comparison_manifest.json` 的确切 direct
regular file，不能通过 symlink 逃逸。发布前必须 recompute SHA-256 over
the exact current file bytes，并与完整小写 `sha256:<64 hex>` 值一致。还要
Validate the referenced `comparison_manifest.json` producer schema；
final-selection consumption requires schema version 2。Validate it as a
schema-version-2 comparison manifest，
要求 `comparison_id` 等于 parent directory、exact `selection_id` 等于 resumed
selection，且 exact `{path, sha256}` candidate-set reference 等于 resume
request 与 current candidate set。必须 reject cross-selection substitution。
Candidate evidence must equal the exact ordered projection of complete
candidate-set entries。对 every `candidate_evidence` entry, including
non-selected candidates，必须重新执行 complete lineage-v2 validator 和
transitive report-review/full-manifest checks，并 independently recompute
candidate run digest、lineage audit hash、exact four output hashes 和 exact
figure set。report、figure 或 malformed manifest 均不能作为 comparison ref；
identity-only manifest 也无效。

Schema-version-1 comparison manifests are historical only；不得 infer 或
backfill selection binding，必须从 exact current candidate set 和 transitive
evidence regenerate schema version 2。

Comparison population must be a subset of the hash-bound candidate set，且每个
comparison 的 ordered identities 必须等于 candidate order 中的 projection。
comparison_refs must be non-empty whenever candidate_set has multiple
candidates。Exactly two candidates 时，每个 referenced comparison 必须
exactly equal the two-candidate ordered population。More than two candidates
时，每个 population 至少有两个 unique candidates，the union must exactly equal
the complete candidate-set population，并且 comparison coverage graph must be
connected（candidate 是 node，共同出现在 comparison 中形成 edge）。Omitted
candidate、unrelated replacement、duplicate、order mismatch、disconnected group
或非 current candidate-set population 都必须 reject。Reject an omitted candidate
or unrelated replacement explicitly。只有 single-candidate set
可使用 `comparison_refs: []`。The selected identity must appear exactly once
in the union of validated comparison populations。The selector must not require
the selected identity in every referenced comparison；connected A-B/B-C coverage
可以选择 A、B 或 C。
The old requirement that the selected version_id/run_id identity appears
exactly once in each referenced comparison is invalid。

`current_final.json` 只做 hash-bound 指针：

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_180000",
  "selected_version_id": "v002",
  "selected_run_id": "run_20260712_173012",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_180000/candidate_set.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "final_decision": {
    "path": "<final_dir>/selection_20260712_180000/final_decision.json",
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

### Workspace-Scoped Selection Publication Lock

Use `governing_workspace_root(subject)` with the canonical subject and nearest
ancestor `.open-xquant/workspace.yaml`，then precompute
`final_selection_lock_path(subject)`。Valid non-governed subject => no lock；
malformed or unsafe governed configuration => fail closed。Use
`hold_final_selection_lock(precomputed_path)` for canonical
`<workspace_root>/.open-xquant/locks/final-selection.lock`。Resolve real
workspace root，reject symlink path components，并在 persistent regular lock
file 上使用 exclusive advisory file lock；never unlink the lock file。The
Final Selector owns the lock lifecycle、comparison-reference ledger 和 pointer
publication。All governed
writers of selection-transitive evidence 必须在 replacement 前获取同一个
exclusive lock，包括 policy、candidate set、lineage、run inventory、
comparison manifest/output、comparison-reference ledger、review/report、
decision 和 pointer writers。

Runtime run publisher acquires the final-selection lock centrally；Agent must
not pre-acquire it。Runtime order 是 canonicalize/discover ->
`run_digests.jsonl.lock` -> `final-selection.lock` innermost，并 hold both
through recovery、mutation、publication 和 validation。

Agent-owned chart/report/review publication must use
`publish_report_artifacts(report_dir, artifacts, *, lock_subject=None)` for
assets, scripts, report manifests, Markdown, HTML, and result JSON. The mapping
contains safe relative keys and complete `bytes`; `None` deletes a target. A
callable builder executes under the final-selection lock, performs the baseline
check there, and commits one atomic all-or-rollback batch. Direct path writes,
shell redirection, and report asset CLI publication paths are forbidden. For
exports outside the governed workspace pass `lock_subject=source_run_dir`. If
the writer needs coherent run locking, wrap publication with
`run_digest_transaction(source_run_dir)`; runtime acquires the run lock first
and the final-selection lock second. Never pre-acquire the final lock around
the publisher.

Agent-owned lineage、comparison 和 final-decision publisher 必须先调用
`governing_workspace_root(subject)`、预计算
`final_selection_lock_path(subject)`，再用
`hold_final_selection_lock(precomputed_path)` 作为 last lock acquired。

Selection lock 是 workspace-wide order 中的 last lock acquired。Selector must
release every run and registry lock before `final-selection.lock`，and holder
must not acquire another lock while holding it。Router/Coordinator 不获取 lock，
只负责串行 handoff；workers 遵守该协议。Selector acquires this precomputed
lock for comparison-reference ledger publication and final decision/pointer
publication。Pointer publisher 必须 acquire it
before direct schema-version-4 transitive revalidation，并 hold it continuously
直到 validation、unchanged checks、pointer validation 和 atomic
`current_final.json` replacement 全部完成。

Full validator 必须 snapshot the exact bytes at each dependency read。完成
closed-set validation 后执行 unchanged-byte sweep：逐一重读 policy、candidate
set、`comparison_refs.json`、所有 lineage/run-inventory inputs、所有 comparison manifests/outputs、
reviews/reports 和 published decision，并要求 byte-for-byte equality。这一
sweep 必须检测 mutation after every dependency read。

开始前保留 previous pointer byte-for-byte 供 recovery classification。新
pointer 在同目录 temp regular file 完整构建、校验、flush 和 `fsync`；atomic
replace `current_final.json` 后必须执行 `fsync(<final_dir>)` after atomic
replacement，完成后才可报告成功。Pre-rename failure 保证 prior pointer
unchanged。Post-rename directory-sync failure 表示 publication outcome is
indeterminate；此时 must not claim that the prior pointer is unchanged，必须进入
下面的 locked recovery。最后 release advisory lock，但 never unlink lock file。


先校验完整 in-memory decision 并原子写入。Immediately before
`current_final.json` publication，必须针对 published decision 和 current
workspace evidence 重跑 full schema-version-4 decision validation，而不是只检查
candidate set。Inside the final-selection lock，perform only direct byte
snapshots、direct parsing/hash/path checks、unchanged-byte sweep 和 atomic
pointer replace；must not invoke `validate_run_artifact_inventory`，must not
invoke `require_current_run_digest`，must not invoke run-locking APIs。完整 gate
必须重新读取并验证 selection policy、
`comparison_refs.json`（其 canonical parsed array 必须 exact equal
`final_decision.comparison_refs`）、comparison manifests and every required comparison
output（包括 exact figure inventory）、candidate set、每个 candidate 的 complete
lineage-v2 validator 与 transitive report-review bindings、每个 direct snapshot
current run digest 和 lock 前已完成的 inventory result，以及 selected report review
和 report artifacts，并复算所有 schema、identity、path、set 和 hash invariants。

完整 gate 通过后，re-read the published `final_decision.json` bytes；this must
re-read the decision bytes after that validation，并要求与刚完成 full validation
的 bytes byte-for-byte unchanged，然后才计算
`final_decision.sha256` 并构建 pointer。Policy、comparison manifest、required
comparison output、candidate、lineage、inventory、report 或 decision 在 decision
与 pointer 之间发生 mutation 都必须 block。Pointer 的 candidate-set reference
必须与 decision exact equality；pointer hash-bound decision bytes 和
pointer-time ledger validation 必须 transitively bind the same persisted
comparison-reference array，完整 pointer 验证后才可原子发布。Write
`current_final.json` last；任何 pointer-time schema、path、hash、evidence 或写入
失败都必须 do not update `current_final.json`；leave the prior
`current_final.json` unchanged。

`blocked` 或 `fail` decision 仍使用完全相同的 top-level keys，但 selected
identity、`selected_as`、`selected_run`、`report_review` 和 `lineage_audit`
为 JSON `null`，`report_artifacts` 为 `{}`，且 `blocking_findings` 非空。
有效 schema-version-2 candidate set 已发布时必须保留 hash-bound reference；
只有无法安全发布 candidate set 时才可为 JSON `null`。它们不得写入或更新
`current_final.json`。schema-version-1 through schema-version-3
`final_decision.json`、未绑定 candidate set 的 decision、path-only policy
reference 和 schema-version-1/schema-version-2 pointer 仅作为历史记录；do not backfill
`lineage_audit` in place，必须从 exact current lineage audit、run/report
evidence、comparison manifests 和 confirmed policy rerun final selection
生成 schema-version-5 decision 和 schema-version-4 pointer。regeneration
block/fail 时 existing current pointer
unchanged。

Schema-version-1 prepare requests and candidate sets are historical only；它们
缺少 selection-bound confirmed policy，不得 backfill 或 resume。必须返回
`restart_selection`，使用新的 schema-version-2 prepare request、selection id、
policy v2 和 candidate-set v2 仅可重放 historical validator；current production
必须使用新的 schema-version-3 request、coordinator confirmation event、policy
v3、candidate-set v3 和 comparison v3。
Every current selector result uses schema version 2. `restart_selection`
allocates a new selection id and must not reuse or overwrite the failed
selection directory.
外层 schema-version-4 decision 或 schema-version-3 pointer 若仍绑定 candidate
set v1 或 unbound policy，也属于 historical record；outer version 未变化不能
替代 restart。

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
- 新 workspace 建议使用 `final/`；已有安全配置可以继续使用
  `runs/final`，但必须遵守下述 placeholder-aware ownership resolution。

## Artifact Ownership Rules

每个角色有唯一写入边界：

所有 role 必须先解析 placeholder，再比较 `outputs` 与
`forbidden_outputs`。若同一路径同时匹配两者，declared output 只在该确切
file 或 declared output subtree 内优先；其 parent、siblings 和 output
以外路径仍受更宽的 forbidden rule 约束。例如 Final Selector 配置
`paths.final_dir: runs/final` 时可写 `runs/final/**`，但不能写其他
`runs/**` 路径。

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
- Report Writer：`10_reports/<run_id>/candidates/<report_revision_id>/`。
- Report Reviewer：
  `10_reports/<run_id>/reviews/<review_revision_id>/report_review.json`。
- Comparator：standalone 使用 `comparisons/<comparison_id>/`；selection-bound
  使用 `comparisons/<selection_id>/<comparison_id>/`。
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
  `lineage.json`、`<version_root>/v001/version_manifest.json`、
  `<version_root>/v001/phase_state.json` 和 `01_brainstorm` 到 `10_reports`
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
- 所有跨 artifact 引用都有 producer schema 要求的 path/hash 字段；仅当
  结构化引用的 schema 定义 `hash_type` 时才要求该字段。

## Migration For Existing Messy Workspace

对已有 `global10` 这类目录，迁移步骤如下：

1. 运行 `govern-research-workspace` 生成 workspace audit。
2. 根据现有 `strategy_idea_brief.json` 和 `strategy_idea_audit.json` 创建
   `<phase_paths.01_brainstorm>/` 和 `<phase_paths.02_idea_audit>/`。
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

## Round 25 Normative Addendum

This section supersedes earlier producer schemas for new artifacts. Earlier
examples remain historical recognition fixtures only.

### Immutable Report And Review Revisions

Selectable reports are immutable report revisions under
`<phase_paths.10_reports>/<run_id>/candidates/<report_revision_id>/`; reviews
are immutable review revisions under
`<phase_paths.10_reports>/<run_id>/reviews/<review_revision_id>/`. Create each
revision directory exclusively and publish once. Never overwrite, delete,
rename, merge, repair, or substitute evidence reachable from any prior
selection. Every consumer requires the exact `{path, sha256}` report-revision
reference to `candidate_manifest.json` and exact `{path, sha256}`
review-revision reference to `report_review.json`.

Current review schema version 2 is:

```json
{
  "schema_version": 2,
  "version_id": "v001",
  "run_id": "runA",
  "review_revision_id": "review_20260712_181500",
  "report_revision": {
    "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "hash_algorithm": "sha256-file-bytes-v1",
  "source_run": {
    "path": "<phase_paths.09_backtests>/runA",
    "digest": "sha256:1111111111111111"
  },
  "status": "pass",
  "verdict": "consistent",
  "findings": [],
  "blocking_findings": [],
  "required_report_edits": [],
  "reviewed_artifacts": {
    "research_report.md": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.md",
      "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "research_report.html": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/research_report.html",
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    },
    "writer_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/writer_result.json",
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
    },
    "metrics.json": {
      "path": "<phase_paths.09_backtests>/runA/metrics.json",
      "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444"
    }
  },
  "decision_inputs": {
    "strategy_spec.yaml": {
      "path": "<phase_paths.04_spec_build>/strategy_spec.yaml",
      "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "spec_audit.json": {
      "path": "<phase_paths.06_spec_audit>/spec_audit.json",
      "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
    },
    "compiled_plan.json": {
      "path": "<phase_paths.07_compile_preview>/compiled_plan.json",
      "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
    },
    "runtime_audit.json": {
      "path": "<phase_paths.08_runtime_audit>/runtime_audit.json",
      "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888"
    },
    "chart_build_result.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/chart_build_result.json",
      "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
    },
    "report_assets/manifest.json": {
      "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/report_assets/manifest.json",
      "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
  },
  "errors": []
}
```

### Current Lineage Schema

Current lineage schema version 3 has exactly `schema_version`, `status`,
`scope`, `hash_algorithm`, `report_revision`, `report_review`, `input_hashes`,
`checked_artifacts`, `blocking_findings`, `warnings`, and
`next_required_phase`. The report/review fields equal the candidate-set refs.
The mandatory inventory replaces the old direct review path with the immutable
review-revision path and adds the candidate manifest path; no mutable direct
report path is accepted.

### Chart And Script Integrity

Asset-manifest schema version 2 requires every safe package-relative
`source.script` plus a full lowercase `source.script_sha256`. Recompute the
script SHA-256 at write, review, lineage, comparison, selection, and pointer
gates; script mutation invalidates all downstream evidence.
`chart_build_result.json` is hash-bound to the exact `{path, sha256}` manifest
reference and records requested/applicable/generated/skipped. Enforce its set
invariants and closed skip reason codes: `missing_optional_input`,
`empty_optional_input`, `structurally_insufficient_input`, and
`not_applicable_to_strategy`. Other failures block rather than masquerading as
optional skips.

### Coordinator-Owned Policy Confirmation

The Coordinator is the sole producer of policy events in the conversation's
append-only `confirmations.jsonl`, written under persistent
`confirmations.jsonl.lock`. Hash exact raw JSONL line bytes excluding the
terminating LF. Caller self-attestation is invalid. A current selector must
validate exact path, event id, line number, event hash, `decision: confirmed`,
selection request id, and policy hash; reject fabricated, stale, mismatched,
duplicate, malformed, replaced, or worker-produced evidence.

```json
{
  "schema_version": 3,
  "mode": "prepare_selection",
  "selection_request_id": "selection-request-20260712-1",
  "selection_id_policy": {
    "source": "generated",
    "selection_id": null
  },
  "selection_policy": {
    "payload": {
      "schema_version": 2,
      "eligible_if": {
        "spec_audit": "confirmed",
        "runtime_audit": "pass",
        "reproducibility_audit": "pass",
        "research_audit_fatal": 0,
        "report_review": "pass"
      },
      "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
      "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
    },
    "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "confirmation_event": {
      "path": "<conversations_dir>/<conversation_id>/confirmations.jsonl",
      "event_id": "selection-policy-confirmation-1",
      "line_number": 1,
      "event_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "decision": "confirmed",
      "selection_request_id": "selection-request-20260712-1",
      "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    }
  },
  "candidate_population": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "report_revision": {
        "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
        "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "report_review": {
        "path": "<phase_paths.10_reports>/runA/reviews/review_20260712_181500/report_review.json",
        "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA_r25.json",
        "sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    }
  ]
}
```

The policy payload hash uses `sha256-canonical-json-v1`; no alternate
confirmed-policy artifact is produced. Current schema-version-3
`selection_policy.json` binds the exact journal line:

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "hash_algorithm": "sha256-file-bytes-v1",
  "policy_hash_algorithm": "sha256-canonical-json-v1",
  "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "confirmation_event": {
    "path": "<conversations_dir>/<conversation_id>/confirmations.jsonl",
    "event_id": "selection-policy-confirmation-1",
    "line_number": 1,
    "event_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "decision": "confirmed",
    "selection_request_id": "selection-request-20260712-1",
    "policy_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "policy_payload": {
    "schema_version": 2,
    "eligible_if": {
      "spec_audit": "confirmed",
      "runtime_audit": "pass",
      "reproducibility_audit": "pass",
      "research_audit_fatal": 0,
      "report_review": "pass"
    },
    "rank_by": ["oos_sharpe_ratio", "max_drawdown", "robustness_status", "trade_count"],
    "tie_breakers": ["simpler_spec", "lower_turnover", "lower_cost_sensitivity"]
  }
}
```

Current candidate-set schema version 3 is:

```json
{
  "schema_version": 3,
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "hash_algorithm": "sha256-file-bytes-v1",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "candidates": [
    {
      "ordinal": 0,
      "identity": {
        "version_id": "v001",
        "run_id": "runA"
      },
      "primary_run": {
        "path": "<phase_paths.09_backtests>/runA",
        "digest": "sha256:1111111111111111"
      },
      "report_revision": {
        "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
        "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "report_review": {
        "path": "<phase_paths.10_reports>/runA/reviews/review_20260712_181500/report_review.json",
        "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      },
      "lineage_audit": {
        "path": "<governance_dir>/lineage_audit_v001_runA_r25.json",
        "sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "scope": {
          "version_id": "v001",
          "run_id": "runA"
        }
      }
    }
  ]
}
```

### Closed Comparator Recovery

Current schema-version-3 `comparison_manifest.json` has exactly
`schema_version`, `comparison_id`, `selection_id`, `hash_algorithm`,
`selection_policy`, `candidate_set`, `candidate_evidence`, and
`evidence_hashes`. Candidate evidence is an exact ordered projection of
complete candidate-set entries, including report/review revisions; evidence
hashes is the closed required output inventory.

Current comparator results use schema version 3 and exactly these keys:

```json
{
  "schema_version": 3,
  "status": "comparison_ready",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": {
    "path": "<comparisons_dir>/selection_20260712_190000/cmp_v001_runA_vs_v002_runB_r25/comparison_manifest.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  },
  "next_action": "resume_selection",
  "blocker_codes": [],
  "blocking_findings": []
}
```

```json
{
  "schema_version": 3,
  "status": "blocked",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": null,
  "next_action": "retry_with_fresh_comparison_id",
  "blocker_codes": ["comparison_id_collision"],
  "blocking_findings": ["The requested immutable comparison directory already exists."]
}
```

```json
{
  "schema_version": 3,
  "status": "fail",
  "selection_id": "selection_20260712_190000",
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_ref": null,
  "next_action": "restart_selection",
  "blocker_codes": ["stale_report_revision"],
  "blocking_findings": ["The candidate report revision no longer matches its binding."]
}
```

The closed blocker-code mapping permits `retry_with_fresh_comparison_id` only
for `comparison_id_collision`, `comparison_build_failed`, and
`comparison_publication_failed`. Stale confirmation, policy, candidate,
report/review revision, lineage, or selection binding requires
`restart_selection`. Unknown or mixed blocker codes are a deterministic
protocol violation and fail closed; prose never controls routing.

### Guarded Historical Revision Route

`candidate_scoped_historical_report_revision` targets one explicit inactive
version under an exact current-state guard. It allocates a fresh
`report_revision_id` and fresh `review_revision_id`; every participant must not
reactivate the version and must not overwrite old evidence. Route
`write -> review -> lineage -> comparison -> reselection`: publish a fresh
lineage audit, a fresh `comparison_id`, then `restart_selection` with a new
selection. Prior revision bytes remain reachable.

The exact handoff is:

```json
{
  "schema_version": 1,
  "mode": "candidate_scoped_historical_report_revision",
  "version_id": "v001",
  "run_id": "runA",
  "base_report_revision": {
    "path": "<phase_paths.10_reports>/runA/candidates/report_20260701_120000/candidate_manifest.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "report_revision_id": "report_20260712_181000",
  "review_revision_id": "review_20260712_181500",
  "current_state_guard": {
    "path": "current.json",
    "active_version": "v002",
    "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  },
  "reason": "stale_report_review",
  "requested_by_role": "oxq-coordinator"
}
```

The field set is closed. `base_report_revision` is null only when no prior
candidate exists. Closed reasons are `missing_report_review`,
`stale_report_review`, `report_revision_required`, and
`chart_revision_required`. Revalidate the guard around every publication.

### Journaled Governance Transaction

Every bootstrap/governance publication is one logical atomic/recoverable
transaction journaled at
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`.
Record target baselines, staged hashes, durable backup hashes, replacement
order, commit index, and `prepared -> committing -> committed`. Acquire
`workspace-governance.lock`, then `final-selection.lock` last under the global
lock order. Hold both through recovery, unchanged-byte checks, replacement,
validation, fsync, and commit.

Journal top-level keys are exactly `schema_version`, `transaction_id`,
`operation`, `state`, `commit_index`, `targets`, and `replacement_order`.
Target keys are exactly `path`, `kind`, `baseline`, `staged`, and `backup`;
baseline binds existence/hash and staged/backup bind safe paths plus exact-byte
hashes.

Before the first replacement, discard staging only. After a non-pointer
replacement but before `current.json`, roll back exact prior bytes/absence from
durable backups. After `current.json` replacement, roll forward only if every
target equals exact staged bytes; otherwise roll back the entire set including
the pointer. Retain journal/backups until committed parent fsyncs are durable.

### Pointer Durability And Decision Output

After pointer rename, `fsync(<final_dir>)` after atomic replacement is
mandatory. A post-rename directory-sync failure means publication outcome is
indeterminate and must not claim that the prior pointer is unchanged. Recover
under `final-selection.lock`: exact new pointer bytes are fully revalidated and
synced, exact prior pointer bytes trigger full retry, and any other bytes block
as corruption. Parent fsync is required and never prohibited.

Atomic replacement is not the final fallible publication operation；required
parent fsync follows it。Every pre-rename failure must leave the prior
`current_final.json` byte-for-byte unchanged；post-rename sync failure 使用上述
indeterminate recovery。

`final_decision.json` is the sole canonical decision artifact. Current decision
schema version 5 and pointer schema version 4 transitively bind the confirmation
event, schema-version-3 candidate set, exact report/review revisions, lineage,
and schema-version-3 comparisons. No companion decision output is required.

```json
{
  "schema_version": 5,
  "status": "selected",
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selected_version_id": "v001",
  "selected_run_id": "runA",
  "selected_as": "final_research_candidate",
  "selected_run": {
    "path": "<phase_paths.09_backtests>/runA",
    "digest": "sha256:1111111111111111"
  },
  "report_revision": {
    "path": "<phase_paths.10_reports>/runA/candidates/report_20260712_181000/candidate_manifest.json",
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "report_review": {
    "path": "<phase_paths.10_reports>/runA/reviews/review_20260712_181500/report_review.json",
    "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  },
  "lineage_audit": {
    "path": "<governance_dir>/lineage_audit_v001_runA_r25.json",
    "sha256": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "scope": {
      "version_id": "v001",
      "run_id": "runA"
    }
  },
  "selection_policy": {
    "path": "<final_dir>/selection_20260712_190000/selection_policy.json",
    "sha256": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  },
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "comparison_refs": [],
  "blocking_findings": []
}
```

```json
{
  "schema_version": 4,
  "selection_id": "selection_20260712_190000",
  "selection_request_id": "selection-request-20260712-1",
  "selected_version_id": "v001",
  "selected_run_id": "runA",
  "candidate_set": {
    "path": "<final_dir>/selection_20260712_190000/candidate_set.json",
    "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  },
  "final_decision": {
    "path": "<final_dir>/selection_20260712_190000/final_decision.json",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  }
}
```
