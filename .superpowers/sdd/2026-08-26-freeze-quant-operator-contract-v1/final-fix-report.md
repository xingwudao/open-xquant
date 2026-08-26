# Final Fix Report — Freeze Quant Operator Contract v1

日期：2026-08-26

## 状态与提交

实现提交：`d7b6e987d7b09f376c61c2167c9c209250ba5f96`

该提交只修改冻结契约、示例与 contract tests；未运行 `oxq`、`src/oxq`
SDK、回测或报告脚本，也未实现 runtime adapter。

## 修改文件

- `contracts/quant-operators/operator-contract-v1.md`
- `contracts/quant-operators/compatibility-policy-v1.md`
- `contracts/quant-operators/operator-manifest-v1.schema.json`
- `contracts/quant-operators/reference_validator_v1.py`
- `contracts/quant-operators/hash-profile-v1.md`
- `contracts/quant-operators/examples/valid/equant-ttr-sma.operator.json`
- `contracts/quant-operators/examples/invalid/uppercase-distribution.operator.json`
- `contracts/quant-operators/examples/valid/equant-ttr-sma.binding.json`
- `contracts/quant-operators/examples/valid/provider-source/ettr.py`
- `contracts/quant-operators/examples/valid/equant_ttr-1.0.0-py3-none-any.whl.b64`
- `tests/contracts/test_quant_operator_contract_v1.py`
- `.superpowers/sdd/2026-08-26-freeze-quant-operator-contract-v1/progress.md`
- `.superpowers/sdd/2026-08-26-freeze-quant-operator-contract-v1/final-fix-report.md`

`quant-panel-v1.schema.json` 继续只承担结构层；跨字段关系全部进入发布的
reference validator。现有 `jsonschema>=4.23` dev dependency 已足够，本波没有
修改 `pyproject.toml` 或 `uv.lock`。

## RED/GREEN 证据

### 1. QuantPanel 语义 validator

RED 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q
```

RED 输出：`18 failed, 32 passed`。失败原因是发布文件
`reference_validator_v1.py` 不存在；真实测试覆盖列名重复、required 列缺失、
六类 dtype、bool 不得作为 integer/number、int64 溢出、未声明字段与重复
`(date, code)`。

GREEN 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q \
  -k 'quant_panel or daily_cn_panel'
```

GREEN 输出：`17 passed, 33 deselected`。

### 2. 排序声明与输入列集合

RED 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q \
  -k 'sorted_input or sort_order or input_list or input_columns or equant_sma'
```

RED 输出：`10 failed, 2 passed, 50 deselected`。失败证明旧 Schema 缺少必填
`requires_sorted_input`/条件 `required_sort_order` 和 `uniqueItems`，发布 validator
也没有 required/optional 唯一与互斥检查。

GREEN 输出：`12 passed, 50 deselected`，使用同一命令。

### 3. 参数定义与 OperatorRequest

RED 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q \
  -k 'reference_validator_rejects_default or wrong_parameter_type or \
  conflicting_parameter_constraints or invalid_parameter_pattern or \
  operator_request'
```

RED 输出：`30 failed, 62 deselected`。失败覆盖 default 的 enum/range/pattern/
length/item-count、constraint 类型适用性、冲突 bounds、非法 pattern、未知或
缺失 request 参数、六类 request 类型和请求 constraint 违规。

GREEN 输出：`30 passed, 62 deselected`，使用同一选择表达式。

### 4. 摘要 profile、source files 与完整 source commit

RED 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q \
  -k 'manifest_digest or source_files or source_file or source_commit or \
  sha256 or source_tree_digest or valid_digest_fixtures'
```

RED 输出：`10 failed, 11 passed, 92 deselected`。失败证明旧 manifest digest
自引用、缺少 source file set、接受短/无算法 commit，并且没有 hash reference
functions 或真实 binding fixtures。

GREEN 输出：`21 passed, 92 deselected`。固定向量包括：

- `sha256_file(b"abc")` =
  `sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`
- 两文件 source-tree profile =
  `sha256:76b8e9c973a22ecc7420b637c53c6b01342d2852ab290217ba708e48efcc9658`
- 示例 source tree、exact manifest/schema bytes 和解码后的正式 wheel bytes
  与外部 binding/manifest 摘要逐项一致。

### 5. 随机种子参数绑定

RED 命令：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q \
  -k 'seed_parameter or required_seed'
```

RED 输出：`4 failed, 1 passed, 113 deselected`。失败覆盖 true 分支缺字段、
合法 integer seed 引用无法表达、未知引用和非 integer 引用。

GREEN 输出：`5 passed, 113 deselected`。

## 最终 GREEN 验证

聚焦 contract tests：

```text
uv run --extra dev pytest tests/contracts/test_quant_operator_contract_v1.py -q
```

输出：`118 passed in 0.43s`。原有 35 项边界测试均保留在该套件中。

JSON 解析：

```text
python3.12 -m json.tool contracts/quant-operators/quant-panel-v1.schema.json
python3.12 -m json.tool contracts/quant-operators/operator-manifest-v1.schema.json
```

两条命令均 `exit 0`。

Ruff：

```text
uv run --extra dev ruff check \
  contracts/quant-operators/reference_validator_v1.py \
  contracts/quant-operators/examples/valid/provider-source/ettr.py \
  tests/contracts/test_quant_operator_contract_v1.py
```

输出：`All checks passed!`

Diff 与依赖边界：

```text
git diff --check
rg -n '(?:from|import) oxq|src/oxq' \
  contracts/quant-operators/reference_validator_v1.py \
  tests/contracts/test_quant_operator_contract_v1.py
```

`git diff --check` 为 `exit 0`；搜索无匹配。

## 自审

- JSON Schema 明确只做结构层；provider contract tests 与 OpenXQuant
  certification 都被规范要求依次执行 Schema 和不可绕过的语义 validator。
- `validate_quant_panel()` 覆盖唯一列名、required 列、所有声明 dtype、未知字段
  和唯一主键，测试不再依赖测试私有语义 helper。
- `validate_operator_manifest()` 覆盖输入列集合、参数 default/constraints 和 seed
  引用；`validate_operator_request_parameters()` 拒绝未知、缺失、类型或约束违规。
- `manifest_digest` 已从 Schema 和两个 manifest fixtures 删除，只保留在外部
  binding。`source_files`、全长 source commit 和三类摘要范围都有机器约束或
  reference function。
- v1 compatibility 已限定为 contract/schema 语义和旧实例继续被新 release
  接受的方向；optional field、schema release/digest、旧消费者拒绝权、v2 边界
  及 provider major + recertification 均已明确。
- 未发现对现有 35 项边界行为的回归；未修改 SDK/runtime adapter。

## 顾虑

- `reference_validator_v1.py` 按契约设计假设调用方已先通过对应 JSON Schema；
  对结构损坏的裸对象可能抛出 `KeyError`/`TypeError`，而非契约语义错误。这是
  两层强制顺序的刻意边界，不是可绕过路径。
- wheel conformance fixture 以 base64 文本存储以保持仓库 artifact 可审查；测试
  先还原正式 `.whl` 原始 bytes 再计算 implementation digest。真实 provider
  binding 必须直接摘要发布的 `.whl` 文件。
- 无阻塞性顾虑。

Decision: 最终审查问题已全部修复并提交。
Why: 五组 RED/GREEN 与最终 118 项聚焦测试均有新鲜证据。
Next step: 由控制器复核提交与本报告后决定合并。
