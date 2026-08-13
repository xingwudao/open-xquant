# PyTdx 自定义数据源示例设计

日期：2026-08-13

## 目标

在 `examples/custom_data_sources/` 中新增一个独立的
`PyTdxDownloader` 示例，通过 `pytdx` 直接连接用户显式指定的通达信兼容
行情服务器，下载 A 股日线并生成 open-xquant 可读取的数据文件。

该示例不要求安装或启动通达信客户端，也不进入 SDK 公共 API、数据源枚举或
默认依赖。它与已有的官方本地 TdxQuant HTTP 示例并列，供用户按环境和许可
边界自行选择。

## 成功标准

- 示例类满足 `oxq.data.providers.Downloader` 协议。
- 直连显式 `host:port`，不依赖通达信客户端。
- 支持 `.SH` 和 `.SZ` 日线，支持单只与多只证券串行下载。
- 默认采用与 yfinance `auto_adjust=True` 相同的 OHLC 比例调整语义。
- 支持 `auto_adjust=False` 获取不复权行情。
- 输出标准 OHLCV Parquet 和可校验 manifest。
- `pytdx` 保持 example-only 依赖，不修改 `pyproject.toml`。
- 自动测试完全模拟 `TdxHq_API`，不访问网络。
- 完成实现、代码审核和修复后，真实下载 `510300.SH` 在
  `2020-05-01` 至 `2026-01-01` 的数据并验证产物。

## 非目标

- 不自行实现通达信 `7709` 二进制协议。
- 不内置、扫描、测速或自动切换行情服务器。
- 不提供断线重连、隐式重试、并发下载或全市场同步。
- 不支持 `.BJ`、指数、分钟线、实时行情、Level-2、期货或港股。
- 不追求与通达信桌面客户端的历史复权价逐分一致。
- 不把 `PyTdxDownloader` 加入 `src/oxq/data/`、CLI source 枚举或 SDK 导出。
- 不在仓库中提交真实行情文件或服务器清单。
- 不承诺第三方服务器的可用性、授权或数据完整性。

## 选择的方案

采用“`pytdx` 薄封装 + example 内本地复权”方案：

1. `pytdx` 负责 `7709` 连接、日线报文和除权除息报文解析。
2. 示例负责输入校验、向历史分页、响应验证、复权、标准化和落盘。
3. 所有连接目标由调用方显式提供。

不选择自行实现协议，因为那会显著扩大协议维护、安全和测试范围。也不选择
外部复权数据源，因为行情与公司行动来自不同 provider 会增加日期、证券和
修订口径的不一致风险。

## 文件布局

```text
examples/custom_data_sources/
├── README.md
├── PYTDX.md
├── pytdx_downloader.py
└── tdxquant_downloader.py

tests/examples/
├── test_pytdx_downloader.py
└── test_tdxquant_downloader.py
```

`README.md` 改为两个通达信方案的入口和比较；`PYTDX.md` 只描述直连示例。
现有 TdxQuant 实现及测试不改变行为。

## 公共示例接口

```python
class PyTdxDownloader:
    def __init__(
        self,
        *,
        host: str,
        port: int = 7709,
        auto_adjust: bool = True,
        timeout: float = 5.0,
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

输入约束：

- `host` 必填，只接受不带 scheme、端口、路径或凭据的 IPv4 地址或主机名。
- `port` 必须是非布尔整数且位于 `1..65535`。
- `timeout` 必须是有限正数。
- `auto_adjust` 必须是 `bool`。
- symbol 统一转为大写，只接受六位代码加 `.SH` 或 `.SZ`。
- `.SH` 映射为 pytdx market `1`，`.SZ` 映射为 market `0`。
- `start`、`end` 必须是规范的 `YYYY-MM-DD`，且 `start <= end`。
- 日期范围双端包含，即 `[start, end]`，与现有 TdxQuant example 保持一致。

CLI 采用：

```bash
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py \
  510300.SH 2020-05-01 2026-01-01 \
  --host 127.0.0.1 --port 7709
```

`--host` 必填；`--no-auto-adjust` 关闭默认前复权。示例和文档不提供默认
服务器地址。

## 依赖装载与连接

模块导入时不导入 `pytdx`。真正下载时通过 lazy import 加载
`pytdx.hq.TdxHq_API`，缺少依赖时抛出带上述 `uv run --with` 命令的
`DownloadError`。

API 使用保守参数：

```python
TdxHq_API(
    multithread=False,
    heartbeat=False,
    auto_retry=False,
    raise_exception=True,
)
```

连接显式传入 `host`、`port` 和 `time_out=timeout`。`download` 在一次连接中
完成单只证券的行情和公司行动请求；`download_many` 在同一连接中按输入顺序
处理证券，遇到首个失败即停止。上下文退出时始终关闭连接。

不捕获 `KeyboardInterrupt` 或 `SystemExit`。pytdx 或 socket 的连接、超时和
协议异常转换为带证券和 endpoint 上下文的 `DownloadError`，保留异常链。

## 日线分页与标准化

日线调用固定为：

```python
api.get_security_bars(9, market, code, offset, 800)
```

`offset=0` 从最新日线开始，每次增加 `800` 向历史分页。每页先独立验证，再
加入总结果。停止条件为以下任一项：

- 已获取到严格早于请求 `start` 的交易日；
- 返回页少于 `800`，证明已到服务端历史末端；
- 返回空页，且此前已获得数据。

设置有限的最大页数作为协议异常保护；达到上限仍未满足停止条件时失败。
完全重复的连续页失败，避免服务端忽略 offset 导致无限循环。跨页相同日期且
OHLCV 完全一致时去重；内容冲突时失败。

每条记录要求：

- `datetime` 可解析为日期；日内时间被规范到交易日零点。
- `open`、`high`、`low`、`close` 是有限正数。
- `low <= open <= high`、`low <= close <= high`。
- `vol` 必须有限且非负；针对 `pytdx` 解码器的浮点误差，按半向上
  规则归一化到最近的 `int64`，溢出时拒绝。

合并结果按日期升序，以 `Asia/Shanghai` 构造名为 `date` 的
`DatetimeIndex`。复权完成后才按双端包含的 `[start, end]` 过滤。若过滤结果
为空则失败。

分页从最新交易日开始而不是从请求 `end` 开始，这是必要行为：yfinance 风格
前复权以当前最新交易日为参考，请求结束日之后已经生效的公司行动仍会影响
请求区间内的历史价格。

## 除权除息数据

`auto_adjust=True` 时，在同一连接调用：

```python
api.get_xdxr_info(market, code)
```

返回 `None` 表示服务端失败；空列表表示没有公司行动。记录日期由
`year/month/day` 组成。

类别处理：

- `category=1` 是除权除息，参与复权。
- `category=2..10`、`13..14` 是上市、股本或权证记录，本例忽略。
- `category=11/12` 是扩缩股；如果其日期会影响输出区间，则严格失败。
- 未知 category 如果会影响输出区间也失败，避免静默产生错误价格。

同日完全相同的 `category=1` 记录去重；同日字段冲突时失败，不猜测合并。
晚于最新交易日的已公告未生效记录忽略。早于或等于输出首日、因而不会改变
输出价格的记录也不参与计算。

## yfinance 风格比例前复权

`category=1` 字段单位为：

- `fenhong`：每 10 股现金分红。
- `peigujia`：配股价格。
- `songzhuangu`：每 10 股送转数量。
- `peigu`：每 10 股配股数量。

对除权日 `D`，找到严格早于 `D` 的最后一个交易日收盘价
`previous_close`，计算：

```text
cash = fenhong / 10
bonus = songzhuangu / 10
rights = peigu / 10

reference_price =
    (previous_close - cash + rights * peigujia)
    / (1 + bonus + rights)

event_ratio = reference_price / previous_close
```

所有输入必须存在、为有限非负数；`previous_close`、分母、
`reference_price` 和 `event_ratio` 必须为有限正数，否则失败。

对于每个交易日，累乘日期严格晚于该交易日、且不晚于最新交易日的全部
`event_ratio`：

```text
adjusted_open = raw_open * cumulative_ratio
adjusted_high = raw_high * cumulative_ratio
adjusted_low = raw_low * cumulative_ratio
adjusted_close = raw_close * cumulative_ratio
volume = raw_volume
```

这与 yfinance `auto_adjust=True` 使用 `Adj Close / Close` 同比例调整 OHLC、
不调整成交量的应用语义一致。示例不做逐分舍入，保留 `float64`，以兼容 ETF
可能使用的 `0.001` 价格精度。

该算法是比例复权，不是通达信桌面端可能采用的仿射变换。现金分红存在时，
两者可能产生历史数值差异；README 必须明确“不保证与通达信客户端逐分
一致”。

`auto_adjust=False` 时不请求公司行动，OHLC 原样输出，volume 同样不变。

## 输出与 manifest

Parquet 文件名为规范化 symbol：

```text
510300.SH.parquet
510300.SH.manifest.json
```

列顺序固定为：

```text
open, high, low, close, volume
```

OHLC 为 `float64`，volume 为 `int64`。manifest 的 provider 为 `pytdx`，
`extra` 包含：

```json
{
  "auto_adjust": true,
  "adjustment_method": "xdxr_ratio_yfinance_semantics",
  "adjustment_reference_date": "2026-08-13",
  "applied_event_count": 3,
  "bar_category": 9,
  "host": "explicit.example",
  "period": "1d",
  "port": 7709,
  "pytdx_version": "1.72",
  "transport": "tdx_hq_tcp"
}
```

`adjustment_reference_date` 是服务端返回的最新交易日，不是请求 `end`。
`auto_adjust=False` 时 `adjustment_method` 为 `none`，事件数为 `0`。

所有响应和复权验证在首次写入之前完成。示例沿用现有 Downloader 的覆盖
语义，不实现 Parquet 与 manifest 的跨文件事务。README 说明批量调用可能
留下失败之前已经完成的证券文件，生产实现应增加临时文件、原子替换、增量
更新和失败恢复。

## 错误模型

调用者参数错误使用 `ValueError`：非法 host、port、timeout、auto_adjust、
symbol 或日期范围。

环境、连接、响应、数据和复权错误使用 `oxq.core.errors.DownloadError`：

- 缺少 `pytdx` 或无法读取其版本；
- 连接失败、超时或服务端异常；
- `get_security_bars` / `get_xdxr_info` 返回 `None`；
- 空区间、分页不完整、重复页或冲突日期；
- 日期、OHLCV、公司行动字段非法；
- 找不到除权日前收盘价；
- 不支持的扩缩股或未知事件影响输出区间；
- 复权分母、参考价或比例无效。

错误消息不包含原始二进制响应或潜在敏感数据。第一版不自动重试，因为重试
会掩盖服务器、授权、协议兼容性和输入错误之间的区别。

## 自动测试

`tests/examples/test_pytdx_downloader.py` 使用 fake API 或 mock API class，测试
期间禁止真实 socket。覆盖：

- 满足 `Downloader` 协议；
- lazy import、缺少依赖和版本记录；
- API 初始化、连接参数、关闭和异常转换；
- `.SH` / `.SZ` 映射及非法 `.BJ`；
- 规范日期、双端包含过滤和空区间；
- `800` 条分页、最大页保护、重复页、跨页去重与冲突；
- 日期、OHLC 关系、volume 和响应类型校验；
- Parquet schema、时区、排序和 manifest；
- `auto_adjust=False` 不请求公司行动；
- 现金分红、送转、配股、同日重复和多事件累计；
- 请求结束日之后的已生效事件仍调整历史区间；
- 未生效事件忽略；
- `category=11/12`、未知相关 category、缺少前收盘价和无效因子失败；
- OHLC 使用同一比例且 volume 不变；
- `download_many` 串行、首错停止；
- CLI 默认复权、`--no-auto-adjust`、必填 host 和输出路径。

测试按严格 TDD 执行：先写行为测试并确认因缺少实现而 RED，再写最小实现使
其 GREEN；修复代码审核意见时同样先增加能复现问题的失败测试。

## 文档

`README.md` 改为简短入口，比较：

- TdxQuant：官方本地 HTTP，需要启动支持 TQ 的客户端。
- PyTdx：直接连接兼容行情服务器，不需要客户端，但依赖已归档的第三方
  `pytdx`，并承担服务器条款、稳定性和数据许可风险。

`PYTDX.md` 包含安装、Python API、CLI、复权语义、输出、显式 host、故障排查、
真实 smoke test、非生产限制和合规警告。它不得推荐或内置公共服务器地址。

## 依赖与 SDK 边界

不修改：

```text
pyproject.toml
src/oxq/data/loaders.py
src/oxq/data/__init__.py
src/oxq/tools/data.py
src/oxq/cli/sdk_bundle.py
src/oxq/cli/doctor.py
```

运行时通过 `uv run --with pytdx==1.72` 临时提供依赖。示例复用 pandas、
Parquet、`resolve_data_dir`、`write_manifest` 和 `DownloadError`，但不会成为
SDK 内置数据源。

## 安全和合规边界

- `pytdx` 原项目已经归档，README 明确说明其老旧、个人学习定位和非商业
  使用声明。
- 示例只接受显式连接目标，不扫描、不测速、不维护服务器清单。
- 用户必须自行确认服务器访问、行情使用、存储和再分发符合适用条款。
- 示例不提交真实数据、账号、token、服务器地址或抓包材料。
- 真实 smoke test 产生的数据写入被 git 忽略的临时目录，并在报告中只记录
  行数、日期范围、schema 和 manifest 校验结果。
- 示例定位为学习和本地研究，不能视为生产数据接入承诺。

## 实施后代码审核

实现完成并通过 focused 检查后进行一次独立的需求、正确性、安全性和测试
审核。每条可执行评论先验证，确认为问题后增加回归测试，再做最小修复。
修复后重新运行 focused 和完整验证；没有可执行评论才能进入真实 smoke test。

## 真实 ETF 验收

目标证券和区间固定为：

```text
symbol = 510300.SH
start = 2020-05-01
end = 2026-01-01
auto_adjust = True
```

用户先前的 `202050101` 和 `2026010101` 按上述规范日期解释。测试显式选择
并传入一个当时可达的兼容服务器，但不会把该地址写入源码、文档或提交。

成功必须同时满足：

- 不启动通达信客户端即可建立 `7709` 连接；
- 下载命令退出码为 `0`；
- 输出首日不早于 `2020-05-01`，末日不晚于 `2026-01-01`；
- 行数大于零，日期唯一、升序，schema 和 dtype 正确；
- OHLC 全部有限，volume 非负；
- manifest provider、日期、行数和 SHA-256 校验正确；
- `auto_adjust=True` 和复权方法写入 manifest。

服务器不可用不等于实现失败，但本次用户目标明确要求成功获取，因此在找到
可达、允许测试的显式 endpoint 并完成上述验证前，整体任务不算完成。

## 验证命令

实现完成后至少执行：

```bash
uv run pytest -q tests/examples/test_pytdx_downloader.py
uv run pytest -q tests/examples/test_tdxquant_downloader.py
uv run ruff check examples/custom_data_sources tests/examples
uv run mypy examples/custom_data_sources/pytdx_downloader.py
uv run pytest
git diff --check
```

真实 smoke test 使用 `uv run --with pytdx==1.72` 和显式 `--host`，产物写入
仓库外临时目录。

## 参考

- [pytdx 项目归档声明](https://github.com/rainx/pytdx)
- [pytdx 除权除息字段讨论](https://github.com/rainx/pytdx/issues/8)
- [pytdx 日线分页参数讨论](https://github.com/rainx/pytdx/issues/47)
- [yfinance auto_adjust 当前实现](https://github.com/ranaroussi/yfinance/blob/master/yfinance/utils.py)
- [通达信字段和复权公式交叉验证](https://github.com/injoyai/tdx/blob/master/protocol/model_gbbq.go)
