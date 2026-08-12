# 直接连接行情服务器：PyTdxDownloader

`PyTdxDownloader` 演示如何实现 `oxq.data.providers.Downloader`，通过
`pytdx` 直接连接用户指定的通达信兼容行情服务器，下载沪深 A 股日线并生成
open-xquant 可读取的数据文件。

这是可复制的 example，不是 SDK 内置或承诺长期兼容的数据源。它不会修改
open-xquant 的数据源枚举、CLI、doctor 或依赖。

## 前提

- 不需要安装或启动通达信客户端。
- 需要自行准备有权访问的兼容行情服务器主机和端口。
- 必须显式传入 host；示例不内置、扫描、测速或自动切换服务器。
- 仅支持六位代码加 `.SH` 或 `.SZ`，例如 `510300.SH`。
- 仅支持日线。

`pytdx` 不写入项目依赖，使用临时依赖运行：

```bash
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py --help
```

## Python 调用

```python
from pathlib import Path

from examples.custom_data_sources.pytdx_downloader import PyTdxDownloader

downloader = PyTdxDownloader(
    host="YOUR_TDX_HOST",
    port=7709,
    auto_adjust=True,
)
path = downloader.download(
    "510300.SH",
    "2020-05-01",
    "2026-01-01",
    dest_dir=Path("data/market"),
)
print(path)
```

## 命令行调用

```bash
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py \
  510300.SH 2020-05-01 2026-01-01 \
  --host YOUR_TDX_HOST --port 7709 \
  --dest-dir data/market
```

`start` 和 `end` 均包含在请求区间内。使用 `--no-auto-adjust` 获取不复权
OHLC：

```bash
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py \
  510300.SH 2020-05-01 2026-01-01 \
  --host YOUR_TDX_HOST --no-auto-adjust \
  --dest-dir data/market
```

## 默认复权语义

默认 `auto_adjust=True`。示例从同一行情服务器获取原始日线和
`get_xdxr_info` 除权除息记录，自行构造事件比例。它采用与 yfinance
`auto_adjust=True` 相同的应用方式：

```text
adjusted_open  = raw_open  * cumulative_ratio
adjusted_high  = raw_high  * cumulative_ratio
adjusted_low   = raw_low   * cumulative_ratio
adjusted_close = raw_close * cumulative_ratio
volume         = raw_volume
```

复权以服务器返回的最新交易日为参考。因此，即使某次请求的 `end` 较早，
`end` 之后已经生效的公司行动仍可能改变请求区间内的复权价格。这与每次调用
yfinance 得到当前口径的历史复权数据相似。

该算法是纯比例复权，不是通达信桌面端可能使用的仿射变换；存在现金分红时，
历史价格可能不与通达信客户端逐分一致。示例保留 `float64`，不强制舍入到
`0.01`，以兼容 ETF 的 `0.001` 价格精度。

遇到扩缩股、未知的相关事件、无效字段或找不到除权日前收盘价时，示例明确
失败，不静默退回不复权数据。

## 输出

命令生成：

```text
data/market/510300.SH.parquet
data/market/510300.SH.manifest.json
```

Parquet 列顺序固定为 `open`、`high`、`low`、`close`、`volume`。索引名为
`date`，唯一、升序，时区为 `Asia/Shanghai`；OHLC 是 `float64`，volume 是
`int64` 且不参与复权。

manifest 的 provider 为 `pytdx`，并记录：

- `auto_adjust`
- `adjustment_method`
- `adjustment_reference_date`
- `applied_event_count`
- `bar_category`
- `host` 和 `port`
- `pytdx_version`
- `transport`

## 常见问题

- 提示缺少 `pytdx`：使用带 `--with pytdx==1.72` 的完整命令。
- 连接失败或超时：检查显式 host、端口、网络和服务器访问许可。
- 返回空数据：检查证券市场后缀、日期范围和服务器历史数据完整性。
- 分页重复或不终止：服务端可能不兼容 `pytdx` 的 offset 语义，示例会失败
  而不是无限请求。
- 公司行动失败：相关记录不能由本例的比例公式可靠解释，请不要把失败静默
  当成无复权数据使用。
- `.BJ` 被拒绝：`pytdx==1.72` 早于北交所，本例不做未经验证的市场映射。

## 人工 smoke test

以下命令用于在实现、离线测试和代码审核全部通过后检查一只 A 股 ETF：

```bash
SMOKE_DIR=$(mktemp -d)
uv run --with pytdx==1.72 python \
  examples/custom_data_sources/pytdx_downloader.py \
  510300.SH 2020-05-01 2026-01-01 \
  --host YOUR_TDX_HOST --port 7709 \
  --dest-dir "$SMOKE_DIR"
```

在线 smoke test 依赖外部服务器，不进入 CI；自动测试会替换全部 pytdx API，
不会打开真实 socket。

## 非生产用途警告

本示例适合学习和本地研究，不适合生产级批量同步。它不提供服务器容灾、
限流协调、增量同步、失败恢复，也不保证 Parquet 与 manifest 跨文件原子。
`download_many()` 在首个错误处停止，失败之前已经完成的文件不会回滚。

`pytdx` 原项目已于 2020 年归档。项目作者明确说明其代码用于个人学习，要求
不要用于商业目的，并声明代码老旧、停止维护。使用本例前请阅读
[pytdx 归档声明](https://github.com/rainx/pytdx)，自行评估依赖安全性、协议
兼容性和用途限制。

仓库不附带服务器清单、真实行情、账号或访问授权。本示例代码不授予任何
行情数据访问、存储或再分发权。用户必须自行确认所连接服务器的条款、行情
许可和适用法律允许相应用途；不要把下载的行情文件提交到仓库或未经许可
传播。

## 参考

- [pytdx 项目与归档声明](https://github.com/rainx/pytdx)
- [pytdx 除权除息字段讨论](https://github.com/rainx/pytdx/issues/8)
- [pytdx 日线分页参数讨论](https://github.com/rainx/pytdx/issues/47)
- [yfinance auto_adjust 实现](https://github.com/ranaroussi/yfinance/blob/master/yfinance/utils.py)
