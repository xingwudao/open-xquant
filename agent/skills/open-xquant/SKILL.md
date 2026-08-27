---
name: open-xquant
description: >-
  Use when the user asks for any open-xquant or quantitative research task,
  including strategy design, backtesting, factor evaluation, parameter tuning,
  audit, robustness, reports, chart assets, SDK/component development,
  broker/live trading, workspace setup, Agent installation, or when deciding
  which open-xquant skill applies before CLI, SDK, tools, or file writes.
---

# open-xquant Router

Round 26 routing: decision schema `5`; candidate, policy, comparison, and
lineage schema `3`; schema-4 pointers are rejected. Use the exact closed schema-3
`build_selection_comparison` request and route historical refresh as `write ->
review -> lineage -> prepare new selection -> comparison -> resume` with a fresh
selection and candidate-set hash; the old selection must not be reused. The
request includes `comparison_population`.

This is the mandatory entry skill for open-xquant work. It routes the task to
the most specific open-xquant skill and sets only the minimum runner and
workspace context needed for that handoff.

## Router Contract

- Use this skill first for any open-xquant or quantitative research request.
- This skill routes; it does not replace the leaf skill.
- If a more specific skill applies, load and follow that skill before using
  CLI, SDK, scripts, or file writes.
- Existing artifacts, metrics, loaded context, or a simple-looking task are not
  reasons to skip the matching leaf skill.
- The only CLI commands this router may run before leaf-skill handoff are the
  minimal runner/workspace commands explicitly listed in this skill, such as
  `doctor`, `research init`, `research init --sdk`, and `agent status`.
- Do not run other `oxq` commands, import `oxq`, edit specs, create charts, or
  write report files directly from this router.
- Do not write report files directly. Route final report writing to
  `write-research-report`.

If you catch yourself thinking "I know which command to run directly", stop
and load the leaf skill first.

## Runner And Workspace

Resolve the runner before the leaf skill runs commands:

Do not resolve a runner for `brainstorm-strategy-idea` or `audit-strategy-idea`
unless a later routed skill will actually run `oxq`. These two skills forbid
`oxq` commands, SDK imports, and SPEC writes; reading runner metadata before
them only adds permission surface and can interrupt a pure conversation gate.

1. If the current directory is an open-xquant source worktree, or the user is
   developing the framework itself, use the current worktree runner such as
   `uv run oxq` or `uv run --project . oxq`.
2. Otherwise read `~/.config/open-xquant/agent.yaml`.
3. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner`.
4. If that metadata is missing or fails, read
   `~/.config/open-xquant/agent-install.json` and use
   `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`.
5. Keep the shell in the user's research directory. Do not search unrelated
   home directories for another open-xquant checkout.

If a research directory lacks `.open-xquant/workspace.yaml`, this router may
initialize it with the resolved runner before loading the leaf skill that will
create strategy artifacts. Use `research init --sdk` when the user will write
SDK-based custom research code.

`.open-xquant/workspace.yaml` is configuration only. `current.json` and
`lineage.json` live at the workspace root. Resolve each key independently from
`paths`: use `paths.experiment_registry`, `paths.comparisons_dir`, and
`paths.final_dir`. Use `experiments.jsonl`, `comparisons`, or `final`,
respectively, only when that key is absent; never derive one key from another
configured key. Each configured value must be a safe relative path whose
resolved target stays inside the workspace. Reject absolute paths, traversal
outside the workspace, and symlink escapes whose resolved target leaves the
workspace. Call the resolved paths `<experiment_registry>`, `<comparisons_dir>`,
and `<final_dir>` below. Do not probe `.open-xquant/current.json` or other
hidden-directory manifest paths when checking active version, lineage, or
experiment registry state.

When a role contract contains placeholders, resolve every placeholder before
matching `outputs` and `forbidden_outputs`. If a concrete path matches both,
the declared output wins only for that exact file or declared output subtree.
The broader forbidden rule still applies to the output's parent, siblings, and
every path outside the declared output. For example, `paths.final_dir:
runs/final` permits a final-selector output under `runs/final/**` even though
`runs/**` is forbidden, but it does not permit any other `runs/**` write.

## Workspace Classifier

Use exactly the same classifier as the CLI. A workspace is version-governed if
and only if `workflow.layout == version_governed` or the
`paths.versions_dir` key is present. Otherwise it is legacy. The existence of
`.open-xquant/workspace.yaml` by itself does not make a workspace governed.
Key presence determines governance before value validation: malformed
`paths.versions_dir` is invalid governed, never legacy. Require a non-empty safe
workspace-relative string and block absolute paths, `..` traversal, and
canonical symlink escapes instead of falling back to legacy behavior.

```yaml
workspace_classifier:
  governed_if_any:
    - workflow.layout == version_governed
    - paths.versions_dir is present
  invalid_governed_if: paths.versions_dir is present but not a non-empty safe workspace-relative path
  otherwise: legacy
  workspace_yaml_presence_only: legacy
```

## Task Routing

- Agent install, upgrade, uninstall, cached runner, or target directory
  questions: use the embedded "Install And Upgrade Questions" section below.
  If the source checkout is available, `docs/agent-guide.md` has the longer
  installation guide, but installed Agents must not depend on that file.
- Workspace artifact governance, misplaced root-level phase artifacts, messy
  research directories, or pre-handoff layout checks: use
  `govern-research-workspace`.
- version governance for a new strategy family, a user strategy edit, a
  semantic change, or deciding whether to create a new version or append a run:
  use `manage-strategy-version`.
- Artifact lineage checks across version/run/final references before
  comparison, migration, or final selection: use `audit-artifact-lineage`.
- New strategy idea or incomplete strategy description: use
  `brainstorm-strategy-idea`.
- Audit a completed strategy idea brief or brainstorm process: use
  `audit-strategy-idea`.
- Strategy spec creation from a passing `strategy_idea_audit.json`: use
  `build-strategy-spec` in its normal builder mode. Existing SPEC validation
  only: use the same skill's read-only validation-only mode.
- Multi-Agent workflows use narrow leaf skills only. Do not route worker tasks
  to end-to-end skills; the downstream system owns phase ordering and
  orchestration.
- Spec field provenance, SPEC calibration against an audited idea, or
  pre-backtest assumption confirmation: use `audit-strategy-spec`.
- SPEC-to-runtime compile consistency or `compiled_plan.json` material field
  checks: use `audit-runtime-semantics`.
- Authorized backtest execution from gated artifacts: use
  `run-authorized-backtest`.
- Universe, symbols, index membership, survivorship, or tradable pool design:
  use `build-universe`.
- Data availability, parquet inspection, downloads, or data quality: use
  `explore-data`.
- Costs, slippage, fill price, order timing, lot size, cash return, or broker
  simulation assumptions: use `configure-trade-execution`.
- Stop loss, take profit, drawdown guard, holding limit, rebalance constraint,
  or risk rule: use `build-rule`.
- Post-backtest audit, reproducibility, research bias, robustness, or
  experiment logging: use `monitor-strategy-run`.
- Performance interpretation or "did this strategy work?": use
  `review-performance`.
- Factor IC, Rank IC, ICIR, decay, hit rate, or factor predictiveness: use
  `evaluate-factor`.
- Value, quality, momentum, multi-factor screening, or candidate-list
  generation: use `screen-factors`.
- Parameter optimization, grid search, walk-forward, or overfitting checks:
  use `tune-parameters`.
- Indicator overlays or quick chart inspection: use `plot-indicators`.
- Report chart assets, figure requirements, plotting scripts, image QA, or
  registering generated figures: use `build-report-charts`.
- Always inspect `.open-xquant/workspace.yaml` before choosing a comparison
  skill.
  In a version-governed workspace, route every comparison request through
  `compare-strategy-versions`, including a generic "compare two experiments"
  request. This preserves per-version candidate containment and lineage checks.
- Cross-run experiment comparison, spec diff, metric comparison, or comparison
  report in a legacy, non-governed workspace: use `compare-experiments`.
- Cross-version strategy comparison, within-version reproducibility
  comparison, or comparison in a version-governed workspace: use
  `compare-strategy-versions`.
- Final version selection, final research candidate selection, or requests to
  choose/promote/mark a version as final: use `select-final-version`.
- Final human-readable report writing or editing `research_report.md` /
  `research_report.html`: use `write-research-report`.
- Semantic review of a completed report, decision consistency, audit fidelity,
  robustness interpretation, or chart narrative quality: use
  `review-research-report`.
- New workspace-local Indicator, Signal, or PortfolioOptimizer component: use
  `author-component`. Workspace-local custom Rule requests must block unless
  the user explicitly asks for open-xquant framework development that adds
  audited spec, compile, runtime, and backtest support; in that framework case
  use `create-component`.
- Broker connectivity, paper trading, live trading, account checks, or order
  submission: use `manage-live-trading`.

## Historical Schema-2 Staged Routing

The schema-version-2 request below is retained only for historical recognition;
current routing uses the Round 25 schema-version-3 request near the end of this
skill:

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
        "path": "<candidate_run_dir>/runA",
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
        "path": "<candidate_run_dir>/run_20260712_173012",
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

The request and nested objects have exact key sets. There is no implicit
default for `selection_id_policy.source`. `source: generated` requires
`selection_id: null`; `source: provided` requires one valid non-empty,
not-yet-existing selection id.

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

Candidate ordinals, identities, primary runs,
and lineage audit references have exact ordered equality with the candidate set
the selector publishes. Routing may validate but must not rediscover, add,
omit, replace, deduplicate, or reorder the population.

`selection_policy` is a closed union. `source: confirmed_payload` requires the
exact user-confirmed payload shown above and `reference: null`.
`source: hash_bound_reference` requires `payload: null` and an exact
`{path, sha256}` reference to bytes containing that same payload schema. The
router must not infer policy fields. The selector validates either source,
allocates the selection id, and atomically publishes it inside the generated
selection directory as schema-version-2 `selection_policy.json` before binding
it into schema-version-2 `candidate_set.json`. Reject stale or cross-selection
policy; the exact selection id and policy reference must equal at every later
handoff.
The `{path, sha256}` form is the hash-bound source reference, and the selector
atomically publishes it inside the generated selection directory after binding.


A governed final-selection request is one routed workflow. For exactly one
candidate, `prepare_selection` returns `next_action: resume_selection` with
`comparison_refs: []`; the coordinator must not invoke the comparator and
directly calls `resume_selection` only after the Final Selector has durably
persisted the literal `[]` ledger. For two or more candidates it returns
`next_action: compare_then_resume`, followed by
`build_selection_comparison` -> `comparison_ready` -> `resume_selection`.
Both branches begin `prepare_selection` -> `candidate_set_ready` and preserve
the same `selection_id`, selection-policy reference, and exact candidate-set
reference. Final-selection comparison v2 evidence is immutable under
`<comparisons_dir>/<selection_id>/<comparison_id>/`; reject an existing output
directory, and retry with a fresh `comparison_id` under the same selection.
Never overwrite evidence reachable from a prior pointer.
The first selector result is a normal nonterminal handoff. It must not write
`final_decision.json` and must not update `current_final.json`. Route the
comparator with the same `selection_id` and same exact candidate-set reference,
collect validated `comparison_ready` refs in comparator dispatch order, not
completion order, require the same `selection_id` and exact candidate-set
reference, and pass the exact ordered array without alteration to the selector.
Then resume without a new user request. Never route a registry-derived comparison first, never allocate a new
selection id on resume, and never treat missing comparison evidence as a reason
to bypass the candidate-set binding.

The Final Selector is the sole producer of `comparison_refs.json`. The router
and coordinator must not write `comparison_refs.json`, acquire the
final-selection lock, or perform an unlocked filesystem write. The selector
persists the multi-candidate request array under that lock in the existing
selection directory before ranking or final-decision publication, then
`resume_selection` re-reads it and requires exact equality. Retries pass the
same array for the same `selection_id` and exact candidate-set reference; a
stale or different array blocks instead of being overwritten or causing an
implicit replacement selection.


This state machine supports complete two-candidate and three-candidate handoffs
from request through pointer publication. The final selector owns the lock
lifecycle for the canonical workspace lock. All governed writers of
selection-transitive evidence participate in that protocol. The selection lock
is the last lock acquired, and a holder must not acquire another lock while
holding it. On any nonterminal, blocked, failed, validation, unchanged-byte, or
publication failure, leave the prior `current_final.json` byte-for-byte
unchanged.

Runtime publishers invoked by the coordinator acquire the final-selection lock
centrally. The coordinator must not pre-acquire it around a publisher command.
Runtime order is canonicalize/discover, `run_digests.jsonl.lock`, then
`final-selection.lock` innermost; the runtime holds both through recovery,
mutation, publication, and validation. Direct Agent publishers use
`governing_workspace_root(subject)`, `final_selection_lock_path(subject)`, and
`hold_final_selection_lock(precomputed_path)` as documented by their leaf
skills. Discovery uses the canonical subject and nearest ancestor
`.open-xquant/workspace.yaml`; valid non-governed workspaces use no lock, while
malformed or unsafe governed configuration must fail closed.

## Install And Upgrade Questions

This section must work from an installed Agent home or a later research
directory where the original source checkout may have been deleted.

Use the cached metadata first:

1. Read `~/.config/open-xquant/agent.yaml` for `preferred_runner_argv` or
   `preferred_runner`.
2. If missing, read `~/.config/open-xquant/agent-install.json` for
   `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`.
3. Do not search unrelated home directories for another checkout.
4. If the current directory is an open-xquant source worktree and the user is
   installing from that checkout, use `uv run oxq ...`.

Common commands:

```bash
<runner> agent status
<runner> agent install --all-targets
<runner> agent install --all-targets --profile multi-agent --yes
<runner> agent install --all-targets --profile standalone-agent --yes
<runner> agent install --repair --yes
<runner> agent upgrade --all-targets --yes
<runner> agent upgrade --all-targets --from-local . --yes
<runner> agent uninstall --all-targets --yes
<runner> agent uninstall --all-targets --purge-config --yes
```

When installing interactively, ask the user to choose one profile for this
machine:

- `multi-agent`: recommended when the Agent supports multi-Agent or subagent
  workflows; installs narrow leaf skills and prebuilt open-xquant worker roles
  where the target has an official agent-role directory.
- `standalone-agent`: for a single Agent that should orchestrate the same
  narrow phase skills itself; does not install prebuilt worker roles.

With `--yes`, the installer uses the recommended profile unless `--profile` is
provided explicitly.

Supported target skill roots:

- Codex: `${CODEX_HOME:-~/.codex}/skills/`
- OpenCode: `~/.config/opencode/skills/`
- Claude Code: `~/.claude/skills/`
- Cursor: `~/.cursor/skills/`
- OpenClaw: `~/.openclaw/skills/`
- TRAE: `~/.trae/skills/`

Supported target role roots for `multi-agent` profile:

- Codex: `${CODEX_HOME:-~/.codex}/agents/*.toml`
- OpenCode: `~/.config/opencode/agents/*.md`
- Claude Code: `~/.claude/agents/*.md`
- Cursor: `~/.cursor/agents/*.md`

Prebuilt roles are `oxq-coordinator`, `oxq-version-manager-worker`,
`oxq-artifact-governor-worker`, `oxq-strategy-brainstorm-worker`,
`oxq-strategy-idea-auditor-worker`, `oxq-strategy-builder-worker`,
`oxq-data-inspection-worker`, `oxq-component-author-worker`,
`oxq-spec-auditor-worker`, `oxq-runtime-auditor-worker`, `oxq-runner-worker`,
`oxq-monitor-worker`, `oxq-report-writer-worker`,
`oxq-report-reviewer-worker`, `oxq-lineage-auditor-worker`,
`oxq-experiment-comparator-worker`, and `oxq-final-selector-worker`.

The installed skills are flat peer directories such as `open-xquant/`,
`brainstorm-strategy-idea/`, `build-strategy-spec/`, and
`write-research-report/`. Do not nest leaf skills under `open-xquant/`.

## Common Sequences

Every current selector result uses schema version 3. `restart_selection`
allocates a new selection id and must not reuse or overwrite the failed
selection directory.

- "Build and test this idea":
  `manage-strategy-version` -> `brainstorm-strategy-idea` ->
  `audit-strategy-idea` ->
  `build-strategy-spec` -> `explore-data` when data availability is unknown ->
  `audit-strategy-spec` -> user confirmation of the full SPEC table ->
  `audit-runtime-semantics` ->
  `run-authorized-backtest` -> `monitor-strategy-run` ->
  `build-report-charts` builds the default professional chart pack ->
  `write-research-report` ->
  `review-research-report`.
- "Compare two experiments": in a version-governed workspace use
  `compare-strategy-versions`; only in a legacy, non-governed workspace use
  `compare-experiments` -> review `<comparisons_dir>/<comparison_id>/`.
- "Compare strategy versions":
  `audit-artifact-lineage` -> `compare-strategy-versions` -> review
  `<comparisons_dir>/<comparison_id>/`.
- "Select a final version":
  `audit-artifact-lineage` -> user confirmation of selection policy ->
  `select-final-version.prepare_selection` ->
  `compare-strategy-versions.build_selection_comparison` when multiple
  candidates need comparison coverage ->
  `select-final-version.resume_selection` with the same selection -> review
  `<final_dir>/current_final.json`.
- "Generate charts for this run":
  `build-report-charts` -> update report through `write-research-report` ->
  run deterministic `oxq report qa` -> use `review-research-report`.
- "Write the final report":
  `write-research-report` resolves missing chart decisions to
  `chart_decision: default_professional_chart_pack` and writes a blocked result
  if registered chart assets are missing or stale ->
  `build-report-charts` builds the default professional chart pack; do not ask the user whether charts are needed ->
  resume `write-research-report` to draft or update `research_report.md` with
  registered assets -> render HTML from the same Markdown -> deterministic
  `oxq report qa` -> `review-research-report`.
- "Review whether this can be traded":
  `review-performance` and `review-research-report`; route to
  `manage-live-trading` only after the user explicitly asks for broker execution.

## Red Lines

- Do not bypass a matching leaf skill after recognizing it applies.
- Do not treat this router as permission to run commands directly.
- Do not skip `oxq spec validate`, reproducibility audit, research audit, or
  report QA when the routed workflow requires them.
- Do not modify metrics, audit, robustness, or backtest artifacts to improve a
  narrative.
- Do not promote unaudited or failed research to paper/live trading.

## Round 25 Governance Routing

The Coordinator is the sole producer of final-selection policy events in the
conversation-owned append-only `confirmations.jsonl`, protected by the
persistent sibling `confirmations.jsonl.lock`. The exact raw JSONL line bytes
are hashed without the terminating LF. Caller self-attestation is invalid;
route only an exact journal reference and reject fabricated, stale, mismatched,
duplicate, or worker-authored confirmation evidence.

Current selection starts with this schema-version-3 request:

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
        "path": "<candidate_run_dir>/runA",
        "digest": "sha256:1111111111111111"
      },
      "report_revision": {
        "path": "<candidate_report_dir>/runA/candidates/report_20260712_181000/candidate_manifest.json",
        "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "report_review": {
        "path": "<candidate_report_dir>/runA/reviews/review_20260712_181500/report_review.json",
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

The policy hash is `sha256-canonical-json-v1` and must equal the event binding.
There is no alternate confirmed-policy artifact.

Route `candidate_scoped_historical_report_revision` only for an explicit
inactive version with a current-state guard, fresh `report_revision_id`, and
fresh `review_revision_id`. Every worker must not reactivate that version and
must not overwrite old evidence. The fixed route is
`write -> review -> lineage -> comparison -> reselection`: require a fresh
lineage audit, a fresh `comparison_id`, then `restart_selection`; prior revision
bytes remain reachable.

Schema-version-3 comparator results route deterministically. Ready plus
`resume_selection` resumes. Only `comparison_id_collision`,
`comparison_build_failed`, and `comparison_publication_failed` permit
`retry_with_fresh_comparison_id`. Stale or mismatched selection-transitive
evidence requires `restart_selection`. Unknown or mixed blocker codes or an
action/code mismatch are a deterministic protocol violation; never infer a
route from prose.

Version/bootstrap and governance publishers use
`<workspace_root>/.open-xquant/transactions/governance/<transaction_id>.json`
with durable backup hashes and state `prepared -> committing -> committed`.
They acquire `workspace-governance.lock`, then `final-selection.lock` last,
perform unchanged-byte checks, and hold both through recovery and fsync. Before
the first replacement, discard staging. After a non-pointer replacement before
`current.json`, roll back. After `current.json` replacement, roll forward only
for exact staged bytes; otherwise roll back the entire transaction.

Final pointer publication must `fsync(<final_dir>)` after atomic replacement.
Every pre-rename failure must leave the prior `current_final.json` byte-for-byte unchanged.
Post-rename directory-sync failure means publication outcome is indeterminate
and must not claim that the prior pointer is unchanged. Recover under
`final-selection.lock`: exact new pointer bytes are revalidated and synced,
exact prior pointer bytes are retried, and any other bytes block. Parent fsync
is required, never prohibited.

`final_decision.json` is the sole canonical decision artifact. Route no second
decision-format output.
