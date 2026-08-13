# Tushare Pro 标准 Downloader 设计

## 目标

在 open-xquant 标准数据模块中新增 `TushareDownloader`，使 Tushare Pro
成为继 yfinance 和 AkShare 之后的第三个内置行情下载源。用户自行提供
Tushare token，下载结果继续落盘为标准 OHLCV Parquet，并由现有
`LocalMarketDataProvider` 用于研究和回测。

首版只支持 A 股日线，默认前复权。新类必须以结构化类型方式满足现有
`oxq.data.providers.Downloader` protocol，不修改 protocol，也不改变
Strategy Spec 的 `data.provider: local` 语义。

## 非目标

- 不支持指数、基金、期货、期权、港股、美股或实时行情。
- 不接入 Tushare 财务或因子数据。
- 不新增数据源 registry 或 entry point。
- 不重构 yfinance、AkShare 或现有 Parquet/manifest 覆盖行为。
- 不实现限流调度、自动重试、全市场批量接口或失败后的自动数据源切换。
- 不从 `.env` 文件隐式加载 token。

## 方案选择

下载器直接组合 Tushare Pro 的 `daily` 与 `adj_factor` 接口，不调用
`pro_bar`。这样可以显式控制 token、复权基准、字段校验和错误语义，避免
把标准模块的行为绑定到 `pro_bar` 的内部重试和空返回逻辑。

数据源仍按现有边界工作：外部 API 只负责下载，标准 Parquet 是回测输入，
`LocalMarketDataProvider` 是运行时 provider。因此不允许把
`data.provider` 设置为 `tushare`。

## 公共 API 与凭据

新增公开类：

```python
TushareDownloader(token: str | None = None)
```

类提供现有 protocol 所要求的两个方法：

```python
download(symbol, start, end, dest_dir=None) -> Path
download_many(symbols, start, end, dest_dir=None) -> dict[str, Path]
```

token 的解析规则是：

1. 非 `None` 的构造参数优先；
2. 未传构造参数时读取 `TUSHARE_TOKEN`；
3. 最终值去除首尾空白后为空即失败。

显式 token 在构造时保存；`TUSHARE_TOKEN` 在第一次需要创建 client 时读取。
client 创建后凭据与实例绑定，之后修改环境变量不会改变该实例使用的 token。

下载器调用 `tushare.pro_api(token)`，不调用会持久化用户凭据的
`tushare.set_token()`。token 只保存在下载器的私有实例状态中，不写入日志、
异常、Parquet、manifest、工具返回值、示例或测试 fixture。实例第一次下载时
懒加载 Tushare SDK 并创建 client，后续 `download_many()` 调用复用该 client。

`data_load_symbols(source="tushare")` 不接受 token 参数，只通过
`TUSHARE_TOKEN` 使用凭据，避免工具参数和调用记录泄露密钥。直接使用 Python
API 的用户可以显式传入 token。

## 输入契约

`symbol` 必须使用 Tushare 的规范证券代码格式：六位数字加大写交易所后缀，
例如 `600519.SH`、`000001.SZ` 或 `920001.BJ`。首版只校验格式
`^[0-9]{6}\.(SH|SZ|BJ)$`，不根据数字前缀维护一份可能过时的证券类型表；
具体代码是否存在及是否有 A 股日线由 Tushare 返回结果决定。

`start` 与 `end` 同时接受 `YYYY-MM-DD` 和 `YYYYMMDD`，内部规范化为
`YYYYMMDD`。日期必须真实存在且 `start <= end`。Tushare 的 `end_date` 是
包含端点的日期语义；文档必须明确这一点。

输出文件名沿用输入代码，例如 `600519.SH.parquet`。现有路径安全检查由统一
代码形状和目标目录边界保证。

## 数据获取与前复权

单个证券的数据流如下：

1. 解析凭据并校验 symbol、start、end。
2. 通过 lazy import 获得 `tushare` 模块和 Pro client。
3. 把用户区间切成包含端点、无重叠且无缺口的日期块；每块最多 3650 个
   日历日。短区间仍只产生一个日期块。
4. 对每个日期块调用 `pro.daily(...)` 和 `pro.adj_factor(...)`，两个接口使用
   完全相同的 `start_date`、`end_date` 边界。
5. 任一分块响应达到 Tushare 的 6000 行单次上限时立即失败，避免把可能截断
   的结果误当作完整结果。
6. 合并所有分块；允许前置或中间空块，但完整合并结果仍必须通过原有非空、
   响应证券、用户请求区间、必需字段、日期唯一性和数值合法性校验。
7. 按 `trade_date` 将每个日线行与唯一复权因子一对一合并。
8. 取复权因子结果中不晚于用户原始完整范围 `end` 的最大 `trade_date`
   对应因子作为区间基准。
9. 对 open、high、low、close 应用前复权公式。
10. 转换成交量、排序、设置索引和时区。
11. 写 Parquet 和 manifest；manifest 仍记录调用方传入的完整 `start` 和
    `end`，而不是内部日期块边界。

前复权公式遵循 Tushare 公布的 qfq 语义：

```text
adjusted_price = raw_price * row_adj_factor / reference_adj_factor
```

`reference_adj_factor` 必须是复权因子结果中不晚于规范化 `end` 的最新有效
正数。它不要求对应日期存在日线行，因此证券在区间末端停牌时仍使用 Tushare
在请求 `end_date` 之前提供的最新复权基准。OHLC 使用 `float64`，不做逐行
舍入。成交量不随复权因子变化。

Tushare `daily.vol` 的单位是“手”，且可能以带两位小数的浮点数表示精确到股
的成交量。标准输出先计算 `vol * 100`，要求结果在浮点容差内接近整数且位于
`int64` 范围内。精确条件固定为
`abs(scaled_volume - round(scaled_volume)) <= 1e-6`，只使用绝对容差，不使用
相对容差；通过后取最近整数并转换为 `int64`。不满足精确到股条件的响应失败，
不静默舍入。Parquet 的 `volume` 单位是“股”。

最终 DataFrame 只保留 `open`、`high`、`low`、`close`、`volume`，按日期
升序，索引名为 `date`，索引时区为 `Asia/Shanghai`。

## 校验与错误处理

以下情况在落盘前抛出 `DownloadError`：

- token 缺失或为空；
- symbol 或日期格式非法，或者开始日期晚于结束日期；
- 任一 `daily` 或 `adj_factor` 分块响应达到 6000 行，因为该块可能已被
  provider 截断；
- 日线或复权因子结果为 `None`、空表或非 DataFrame；
- 缺少必需列；
- 任一响应的 `ts_code` 不等于请求 symbol；
- 任一响应日期落在规范化后的 `[start, end]` 之外；
- `trade_date` 无法解析或重复；
- OHLC、成交量或复权因子包含非数值、缺失或非有限值；
- 原始或复权后 OHLC 不为正数，或不满足 high/low 包络关系；
- 成交量为负数、无法精确表示为整数股或超出 `int64`，或者复权因子不为正数；
- 任一日线日期没有唯一复权因子；
- Tushare 返回权限、积分、限流、服务端或网络错误。

上游异常文字可以作为诊断信息保留，但必须先将实际 token 的所有出现替换为
`***`。包装后的 `DownloadError` 不保留原始异常的 cause、context 或可见
traceback；实现应在离开捕获原始异常的 `except` 块后抛出清洗后的错误。
缺失 token 的错误只说明可传构造参数或设置 `TUSHARE_TOKEN`。

只有导入异常为 `ModuleNotFoundError` 且 `exc.name == "tushare"` 时，才将其
解释为 optional dependency 未安装，并抛出不带异常链、提示执行
`uv sync --extra tushare` 的 `DownloadError`。缺少 Tushare 的传递依赖、
`ImportError` 及其他普通 `Exception` 都包装为去敏、无异常链的
`DownloadError`，不能误报成 Tushare 未安装。

发生输入、请求或落盘前数据校验失败时不写新的 Parquet 或 manifest。成功
进入落盘阶段后沿用现有下载器的直接覆盖行为，本次不引入临时文件或跨
Parquet/manifest 的原子事务。因此 Parquet 写成功而 manifest 写失败时，可能
保留已更新的 Parquet；这是三个内置下载器共享的现有边界，文档不承诺回滚。

`download_many()` 保持现有实现模式：按输入顺序串行调用 `download()` 并
返回 symbol 到路径的映射，任一异常立即向调用方传播。之前已成功写入的证券
不回滚。工具层 `data_load_symbols()` 继续逐个调用 `download()`，分别收集
成功行数和错误文字。

## Manifest

`write_manifest()` 使用 `provider="tushare"`，保留调用方原始的 `start` 和
`end` 参数，并在 `extra` 中记录：

```json
{
  "adjust": "qfq",
  "adjustment_reference_date": "YYYYMMDD",
  "adjustment_reference_factor": 1.2345,
  "source_volume_unit": "lot",
  "volume_unit": "share",
  "tushare_version": "1.4.29"
}
```

`tushare_version` 取运行时模块的 `__version__`；若上游模块没有公开版本，记录
`"unknown"`。manifest 绝不记录 token 或 token 来源。

## 代码接入面

实现需要同步以下位置：

- `src/oxq/data/loaders.py`
  - 新增 `TushareDownloader` 及内部规范化/校验辅助函数。
  - 更新模块 `__all__`。
- `src/oxq/data/__init__.py`
  - 从公共 API 导出 `TushareDownloader`。
- `src/oxq/tools/data.py`
  - 支持 `source="tushare"`。
  - unknown-source 文案列出 yfinance、akshare、tushare。
- `pyproject.toml` 与 `uv.lock`
  - 新增 `tushare = ["tushare>=1.4.29"]` optional extra。
- `src/oxq/cli/doctor.py`
  - 探测 `tushare` 并提示 `uv sync --extra tushare`。
- `src/oxq/cli/sdk_bundle.py`
  - 将 `tushare` 加入 fallback extras；正常路径继续自动发现 extras。
- 硬编码完整 extras 清单的 CLI 测试和安装测试
  - 同步新增 `tushare`。
- `README.md` 与 `docs/agent-guide.md`
  - 文档化安装、凭据、范围、输入格式、复权与数据单位。

不修改 `src/oxq/spec/schema.py`、validator 或 compiler。

## 测试策略

实现采用 TDD。所有常规 CI 测试使用 mock Tushare module/client，不需要真实
token、网络或 Tushare 权限，也不增加默认执行的 live integration test。

下载器测试覆盖：

- `TushareDownloader` 满足 runtime-checkable `Downloader` protocol；
- 显式 token 优先，环境变量回退，空 token 失败；
- 环境变量在首次 client 初始化时读取，client 在实例内懒创建并复用；
- optional dependency 缺失时给出安装提示且不保留原始异常链；
- 缺失传递依赖或其他普通导入异常时返回去敏、无异常链的 import failure，
  不误报 Tushare 未安装；
- symbol 和两种日期格式正确规范化，非法输入在请求前失败；
- `daily` 与 `adj_factor` 的调用参数正确；
- 长区间使用相同的包含端点日期块，边界无重叠和缺口，分块合并后按日期
  排序，且 manifest 保留用户原始完整范围；
- 前置或中间空块可与其他非空块合并，任一块达到 6000 行时在写入前失败；
- 手算样例验证 qfq 公式，以及停牌时使用独立于日线末端的最新因子基准；
- 拒绝混入其他证券或请求区间外日期的响应；
- `vol * 100`、五列 schema、升序和上海时区正确；
- 拒绝无法精确表示为整数股、超出 `int64` 或复权后非法的数值；
- Parquet 与 manifest 内容正确且均不含 token；
- 空响应、错误类型、缺列、重复日期、缺失因子、非有限值、负成交量和
  非正复权因子全部失败；
- 上游异常统一包装，格式化 traceback、cause 和 context 均不泄露 token；
- `download_many()` 返回完整映射并复用 client。

集成测试覆盖：

- `oxq.data` 公共导出；
- `data_load_symbols(source="tushare")` 路由、行数和逐证券错误收集；
- unknown-source 文案；
- doctor 的 optional dependency 与修复命令；
- SDK bundle、agent install/upgrade/uninstall 的 extras 清单；
- `uv.lock` 可重现解析；
- `ruff`、相关测试、完整测试套件。

在实现阶段还要在项目声明支持的 Python 3.12 与 3.13 环境分别执行安装和
import smoke test，确认 `tushare>=1.4.29` 与本项目解释器兼容。

## 文档与安全说明

README 中英文部分和 agent guide 必须提供：

- `uv sync --extra tushare` 安装命令；
- `export TUSHARE_TOKEN="..."` 的配置方式；
- 显式构造参数和环境变量两种 Python 示例；
- `600519.SH` 等规范代码示例；
- 仅支持 A 股日线、默认 qfq、`end` 包含端点、`volume` 为股；
- Tushare 账户权限、积分和限流由 Tushare 平台决定；
- Tushare `daily` 单次最多返回 6000 行；下载器会把长区间自动切成每块最多
  3650 个包含端点的日历日，并让 `daily` 与 `adj_factor` 使用相同边界；
- 任一分块响应达到 6000 行会在写入前失败，以防静默接受截断结果；
- 下载后回测仍使用 `data.provider: local`。

文档还要披露凭据传输由第三方 Tushare SDK 负责。当前官方 SDK 客户端源码中
API endpoint 使用 HTTP；本次将此第三方传输风险作为用户选择 Tushare 时的
已知上游边界接受，不提供 endpoint 覆盖或自建传输实现。open-xquant 的
“不泄露 token”保证只覆盖本地状态、日志、异常和产物，不覆盖 SDK 发往其
服务端的网络传输。用户应自行评估并遵守 Tushare 的服务条款和安全要求。

## 验收标准

- 新类满足未修改的 `Downloader` protocol。
- 使用显式 token 或 `TUSHARE_TOKEN` 可下载并标准化 A 股 qfq 日线。
- 长区间自动安全分块，短区间保持单次调用；任何达到 provider 6000 行上限
  的分块响应都不会被写成完整数据。
- 输出 schema、日期方向、时区、价格和成交量单位符合本设计。
- 输入、请求和落盘前校验失败不产生新输出；落盘阶段沿用现有非原子边界。
- 除明确排除的第三方网络传输外，本地状态、日志、异常和产物均无 token 泄露。
- 三个标准行情源均可通过 `data_load_symbols` 选择。
- Tushare 是独立 optional extra，并被 doctor 和 SDK bundle 正确识别。
- 文档准确说明能力边界和第三方限制。
- 相关测试、完整测试、ruff 和 Python 3.12/3.13 import smoke test 全部通过。

## 参考资料

- [Tushare Pro token 与 Python SDK 初始化](https://tushare.pro/document/1?doc_id=40)
- [Tushare A 股日线接口](https://tushare.pro/document/1?doc_id=27)
- [Tushare 复权因子接口](https://tushare.pro/document/2?doc_id=28)
- [Tushare qfq 公式与 `pro_bar`](https://tushare.pro/document/2?doc_id=146)
- [Tushare 官方 Python 客户端源码](https://github.com/waditu/tushare/blob/master/tushare/pro/client.py)
