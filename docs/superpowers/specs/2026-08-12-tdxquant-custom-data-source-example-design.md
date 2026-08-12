# TdxQuant 自定义数据源示例设计

日期：2026-08-12

## 目标

在 `examples/` 中提供一个可复制的自定义行情下载器示例，演示如何让
open-xquant 从通达信获取 A 股日线数据，同时不把通达信集成加入 SDK、
工具注册表或默认依赖。

示例通过通达信官方 TdxQuant 本地 HTTP 接口调用 `get_market_data`。
它不实现、封装或演示通达信 `7709` 私有协议。

## 成功标准

- 示例类满足 `oxq.data.providers.Downloader` 协议。
- 支持单只和多只 A 股日线下载。
- 支持前复权和不复权，默认前复权。
- 输出 open-xquant 标准 OHLCV Parquet。
- 输出索引为带 `Asia/Shanghai` 时区的 `DatetimeIndex`。
- 写入 provider 为 `tdxquant` 的数据 manifest。
- SDK 的安装依赖、导出、工具注册及 `source` 枚举保持不变。
- 所有自动化测试使用模拟 HTTP 响应，不要求安装或启动通达信。
- README 清楚说明环境要求、数据授权边界和失败排查方式。

## 非目标

- 不实现 `7709` 私有二进制协议。
- 不加入 `TdxQuantDownloader` 到 `src/oxq/data/`。
- 不增加 `tdx` 或 `tdxquant` optional dependency。
- 不增加 `load_symbols(..., source="tdx")`。
- 不支持分钟线、实时行情、Level-2、指数、期货或全市场并发下载。
- 不在仓库或测试夹具中提交真实行情数据。
- 不承诺通达信终端、账号或行情授权的可用性。

## 文件布局

```text
examples/custom_data_sources/
├── README.md
└── tdxquant_downloader.py

tests/examples/
└── test_tdxquant_downloader.py
```

`examples/custom_data_sources/` 是示例代码目录，不属于 wheel 中的
`oxq` 包。测试只验证示例契约和转换逻辑，不把示例提升为公共 SDK API。

## 公共示例接口

示例定义：

```python
class TdxQuantDownloader:
    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:17709/",
        dividend_type: str = "front",
        timeout: float = 10.0,
    ) -> None: ...

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path: ...

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]: ...
```

约束如下：

- `symbol` 必须使用 `代码.市场`，例如 `600519.SH` 或 `000001.SZ`。
- 市场后缀仅接受 `SH`、`SZ` 和 `BJ`，输入统一转为大写。
- `start` 和 `end` 必须是规范的 `YYYY-MM-DD` 日期字符串，发送前转为
  `YYYYMMDD`。
- `start` 不得晚于 `end`。
- `dividend_type` 仅接受 `front` 或 `none`。
- endpoint 仅接受端口为 `17709` 的 HTTP 回环地址：`127.0.0.1`、
  `localhost` 或 `::1`。
- `timeout` 必须有限且大于零。

要求显式市场后缀可以避免随证券规则变化而失效的代码前缀推断。限制回环
地址可防止示例被误用为任意远程 HTTP 客户端，并符合官方本地接口定位。

## 请求和数据流

每次 `download` 执行以下步骤：

1. 校验 symbol、日期、endpoint、复权类型和 timeout。
2. 向 `http://127.0.0.1:17709/` POST JSON 请求。
3. 调用官方方法 `get_market_data`，参数包括：
   - `field_list`: `Open`、`High`、`Low`、`Close`、`Volume`
   - `stock_list`: 单个规范化 symbol
   - `start_time` 和 `end_time`: `YYYYMMDD`
   - `dividend_type`: `front` 或 `none`
   - `period`: `1d`
   - `fill_data`: `false`
4. 检查顶层 `id`、`result.ErrorId` 和证券级 `ErrorId`。
5. 从 `result.Value[symbol]` 读取 `Date` 和 OHLCV 数组。
6. 将数字字符串转为数值，拒绝布尔 OHLC 值，并验证所有数组等长。
7. 用 `Date` 构造 `DatetimeIndex`，本地化到 `Asia/Shanghai`。
8. 在日期范围过滤前验证完整响应：日期唯一、OHLCV 均为有限值，且成交量可
   精确安全地转为 `int64`。
9. 按日期升序排序，按请求区间再次过滤，并验证过滤后的数据非空。
10. 写入 `{symbol}.parquet`。
11. 写入 manifest，provider 为 `tdxquant`。
12. 返回 Parquet 路径。

manifest 的 `extra` 至少包含：

```json
{
  "dividend_type": "front",
  "period": "1d",
  "transport": "tdxquant_http"
}
```

manifest 不记录账号、主机扫描结果或其他敏感信息。

`download_many` 按输入顺序串行调用 `download`。第一版不使用多证券批量请求，
以避免引入 TdxQuant 股票分页语义、并发压力和部分失败恢复机制。

## 输出契约

Parquet 列严格为：

```text
open, high, low, close, volume
```

价格列为浮点数，`volume` 为 `int64`，索引名称为 `date`。输出文件名保留
规范化 symbol，例如：

```text
600519.SH.parquet
600519.SH.manifest.json
```

前复权只使用 TdxQuant 官方返回值。示例不自行重算复权因子，避免与终端
口径产生差异。

## 错误处理

示例复用 `oxq.core.errors.DownloadError` 表达下载和响应错误。

- 连接拒绝：提示启动支持 TQ 的通达信客户端，并确认本地端口 `17709`。
- 超时：报告 endpoint 和 timeout，不自动无限重试。
- 非 2xx HTTP：报告状态码，不输出响应中的潜在敏感内容。
- 非 JSON 或结构缺失：报告响应结构无效。
- TdxQuant `ErrorId` 非 `0`：包含 symbol 和官方错误信息。
- 空数据：报告请求 symbol 和日期范围。
- 数组长度不一致、非法日期、布尔 OHLC、非数值或非有限字段：拒绝写文件。
- 写入失败：保留原始异常链，避免留下成功 manifest。

第一版不做自动重试。连接失败、终端未启动和数据未下载是不同原因，静默
重试会掩盖真实问题并降低示例可理解性。

## 原子性和已有文件

示例遵循现有 Downloader 的简洁覆盖语义：成功请求后覆盖目标 symbol 的
Parquet 和 manifest。响应验证必须在任何写入之前完成。

示例不实现跨文件事务。README 说明它适合学习和本地研究，不适合作为生产
级批量同步器。生产实现应增加临时文件、原子替换、增量更新和失败恢复。

## 测试设计

测试使用 `unittest.mock` 替换标准库 HTTP 调用，不访问网络。

测试覆盖：

- `TdxQuantDownloader` 满足 `Downloader` 协议。
- 请求方法、endpoint 和 JSON 参数正确。
- 规范 `YYYY-MM-DD` 日期正确转为 `YYYYMMDD`，非规范日期被拒绝。
- OHLCV 字符串正确转换和排序。
- 输出索引为 `Asia/Shanghai`。
- Parquet 和 `tdxquant` manifest 正确写入。
- `front` 与 `none` 正确传递。
- `download_many` 串行生成多个文件。
- 非回环 endpoint 被拒绝。
- 非法 symbol、日期和非有限或非正 timeout 被拒绝。
- 连接拒绝、超时、HTTP 错误和非法 JSON 转为清晰错误。
- 顶层和证券级 TdxQuant 错误被拒绝。
- 空数据、数组长度不一致、布尔或非法数字、非有限数据和重复日期均被拒绝；
  完整响应先验证再按日期范围过滤。

可选人工 smoke test 写入 README，不进入默认 pytest：

```bash
uv run python examples/custom_data_sources/tdxquant_downloader.py \
  600519.SH 2024-01-01 2024-12-31
```

## README 内容

README 包含：

- 示例目的与非 SDK API 声明。
- 安装并启动支持 TQ 的通达信终端的前提。
- 在终端中准备盘后数据的说明。
- Python 调用和命令行 smoke 示例。
- 输出路径和标准 schema。
- 前复权与不复权说明。
- 常见连接、空数据和代码格式错误排查。
- 数据仅供用户按其授权范围使用，不得随仓库提交或未经许可再分发。
- 通达信官方 TdxQuant HTTP 和 `get_market_data` 文档链接。

## 依赖与发布边界

HTTP 客户端使用 Python 标准库 `urllib.request`，不新增 `requests` 或
`httpx` 依赖。DataFrame、Parquet、manifest 和错误类型复用 open-xquant
已有公开能力。

不修改：

```text
pyproject.toml
src/oxq/data/loaders.py
src/oxq/data/__init__.py
src/oxq/tools/data.py
src/oxq/cli/sdk_bundle.py
src/oxq/cli/doctor.py
```

因此该示例不会成为 SDK 支持的数据源，也不会随 `import oxq.data` 导入。

## 合规边界

- 只调用通达信公开文档描述的本地 TdxQuant HTTP 接口。
- 不实现、记录或传播 `7709` 私有协议报文。
- 不内置或扫描通达信公共行情主站。
- 不提交真实行情数据、账号或终端配置。
- README 明确软件示例不授予行情数据使用或再分发权。
- 用户必须自行确认其通达信版本、账号和行情数据许可允许相应用途。

## 被否决的方案

### SDK 内置 TdxQuant Downloader

它会让 SDK 对仅在特定本地环境中可用的服务作出长期兼容承诺，也会扩大
安装、文档、doctor、CLI 和测试范围，不符合本次“扩展示例”目标。

### 示例依赖 easy-tdx

虽然代码少且无需通达信客户端，但仍依赖非官方服务器接入方式，且示例会
受第三方 API 和发布节奏影响。

### 自行实现 7709 私有协议

技术上可行，但公开 example 仍是公开发布第三方接入工具，不能通过放置在
`examples/` 来消除服务器授权与协议条款风险。

### 只读取 `.day` 文件

这是适合未来补充的离线示例，但不能演示按日期向官方接口请求数据。当前
设计优先覆盖用户已确认的官方 TdxQuant HTTP 路线。

## 验收

实现完成后应通过：

```bash
uv run pytest -q tests/examples/test_tdxquant_downloader.py
uv run ruff check examples/custom_data_sources tests/examples
uv run mypy examples/custom_data_sources/tdxquant_downloader.py
uv run pytest
```

在线 smoke test 是人工、可选且不作为 CI 或完成声明的前提。

## 官方参考

- [TdxQuant 简介](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1cfsjkbf8f3is)
- [TdxQuant HTTP 接口](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1hdhbmi50d038.html)
- [通达信用户协议](https://www.tdx.com.cn/about/yhxy/index.html?tabindex=1)
