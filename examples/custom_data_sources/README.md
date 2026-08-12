# 自定义数据源示例：通达信

本目录提供两个相互独立的通达信 Downloader example。它们都只演示如何扩展
`oxq.data.providers.Downloader`，不是 SDK 内置或承诺长期兼容的数据源。

[TdxQuantDownloader]

- Pros: 使用通达信官方本地 HTTP 接口，由终端提供前复权结果。
- Cons: 必须安装并启动支持 TQ 的通达信客户端，并准备盘后数据。
- Best for: 已有合适通达信客户端和官方本地接口的研究环境。
- Risk: 受客户端版本、账号、接口权限和本地数据状态影响。

[PyTdxDownloader]

- Pros: [直接连接行情服务器](PYTDX.md)，不需要安装或启动客户端。
- Cons: 必须显式提供服务器，且依赖已归档的第三方 `pytdx==1.72`。
- Best for: 需要研究自定义网络数据源和本地比例复权的非生产示例。
- Risk: 用户自行承担服务器条款、行情许可、稳定性和协议兼容风险。

## TdxQuantDownloader：官方本地 HTTP

本目录演示如何实现 `oxq.data.providers.Downloader`，通过通达信官方
TdxQuant 本地 HTTP 接口下载 A 股日线并生成 open-xquant 可读取的数据文件。
它是可复制的 example，不是 SDK 内置或承诺长期兼容的数据源。

## 前提

- 安装并启动支持 TQ 的通达信客户端。
- 按通达信客户端说明准备所需盘后数据。
- 本地接口可访问 `http://127.0.0.1:17709/`。
- 使用 `代码.市场` 格式，例如 `600519.SH` 或 `000001.SZ`。

本示例只允许连接 `127.0.0.1`、`localhost` 或 `::1`，不会扫描行情主站，
也不实现通达信 `7709` 私有协议。

## Python 调用

```python
from pathlib import Path

from examples.custom_data_sources.tdxquant_downloader import TdxQuantDownloader

downloader = TdxQuantDownloader(dividend_type="front")
path = downloader.download(
    "600519.SH",
    "2024-01-01",
    "2024-12-31",
    dest_dir=Path("data/market"),
)
print(path)
```

## 人工 smoke test

先启动通达信客户端，再运行：

```bash
uv run python examples/custom_data_sources/tdxquant_downloader.py \
  600519.SH 2024-01-01 2024-12-31 \
  --dest-dir data/market
```

不复权数据使用 `--dividend-type none`。默认是前复权 `front`。在线 smoke
test 是人工检查，不是 CI 或仓库测试的前提。

## 输出

命令生成：

```text
data/market/600519.SH.parquet
data/market/600519.SH.manifest.json
```

Parquet 列顺序是 `open`、`high`、`low`、`close`、`volume`。索引名是
`date`，时区是 `Asia/Shanghai`；`volume` 沿用通达信接口的成交量口径。

## 非生产用途警告

本示例仅供学习和本地研究，不适合生产级批量同步。它直接覆盖目标 Parquet
和 manifest，且不提供跨文件原子性、增量同步或失败恢复。生产实现必须使用
临时文件和原子替换，并实现增量更新与失败恢复后再用于批量任务。

## 常见问题

- 连接被拒绝：确认支持 TQ 的客户端正在运行，且本地端口为 `17709`。
- 返回空数据：确认客户端已准备对应证券和日期范围的盘后数据。
- 代码格式错误：必须提供六位代码和 `.SH`、`.SZ` 或 `.BJ` 后缀。
- 超时：缩小日期范围，并确认客户端没有被其他耗时任务阻塞。

## 数据和授权边界

仓库不附带真实通达信行情、账号或终端配置。本示例代码不授予任何行情数据
使用权或再分发权。请依据你的通达信版本、账号、数据许可和适用条款使用，
不要把下载的行情文件提交到仓库或未经许可传播。

## 官方文档

- [TdxQuant 简介](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1cfsjkbf8f3is)
- [TdxQuant HTTP 接口](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1hdhbmi50d038.html)
- [get_market_data](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1ctuhthaq5qmg/mindoc-1h10g60jt68sc.html)
- [通达信用户协议](https://www.tdx.com.cn/about/yhxy/index.html?tabindex=1)
