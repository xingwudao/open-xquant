from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from oxq.core.errors import DownloadError, SymbolNotFoundError
from oxq.data.providers import Downloader


def load_example_module() -> ModuleType:
    module_name = "examples.modules.11_tdx_data_and_universe"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name)


class FileWritingDownloader:
    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        raise AssertionError("download must not be called")

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        assert dest_dir is not None
        index = pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03"],
            name="date",
            tz="Asia/Shanghai",
        )
        output: dict[str, Path] = {}
        for offset, symbol in enumerate(symbols):
            path = dest_dir / f"{symbol}.parquet"
            frame = pd.DataFrame(
                {
                    "open": [10.0 + offset, 11.0 + offset],
                    "high": [11.0 + offset, 12.0 + offset],
                    "low": [9.0 + offset, 10.0 + offset],
                    "close": [10.5 + offset, 11.5 + offset],
                    "volume": [1000 + offset, 2000 + offset],
                },
                index=index,
            )
            frame.to_parquet(path)
            output[symbol] = path
        return output


class NeverCalledDownloader:
    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        raise AssertionError("download must not be called")

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        raise AssertionError("download_many must not be called")


class MappingOnlyDownloader:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> Path:
        raise AssertionError("download must not be called")

    def download_many(
        self,
        symbols: list[str],
        start: str,
        end: str,
        dest_dir: Path | None = None,
    ) -> dict[str, Path]:
        return self.paths


def test_tdx_data_context_downloader_doubles_satisfy_downloader_protocol() -> None:
    assert isinstance(FileWritingDownloader(), Downloader)
    assert isinstance(NeverCalledDownloader(), Downloader)
    assert isinstance(MappingOnlyDownloader({}), Downloader)


def test_build_tdx_data_context_reopens_data_and_builds_universe(
    tmp_path: Path,
) -> None:
    module = load_example_module()
    context = module.build_tdx_data_context(
        FileWritingDownloader(),
        ["510300.SH", "159915.SZ"],
        "2024-01-01",
        "2024-01-31",
        tmp_path,
    )

    snapshot = context.universe.get_universe("2024-01-31")
    assert snapshot.symbols == ("510300.SH", "159915.SZ")
    assert snapshot.source == "static:tdx-example"
    assert context.downloaded_paths == {
        "510300.SH": tmp_path / "510300.SH.parquet",
        "159915.SZ": tmp_path / "159915.SZ.parquet",
    }
    bars = context.market.get_bars("510300.SH", "2024-01-01", "2024-01-31")
    assert list(bars["close"]) == [10.5, 11.5]
    assert str(bars.index.tz) == "Asia/Shanghai"


def test_build_tdx_data_context_rejects_empty_symbols_before_downloading(
    tmp_path: Path,
) -> None:
    module = load_example_module()

    with pytest.raises(ValueError, match="symbols must not be empty"):
        module.build_tdx_data_context(
            NeverCalledDownloader(), [], "2024-01-01", "2024-01-31", tmp_path
        )


def test_build_tdx_data_context_rejects_duplicate_symbols_before_downloading(
    tmp_path: Path,
) -> None:
    module = load_example_module()

    with pytest.raises(ValueError, match="symbols must be unique"):
        module.build_tdx_data_context(
            NeverCalledDownloader(),
            ["510300.SH", "510300.SH"],
            "2024-01-01",
            "2024-01-31",
            tmp_path,
        )


def test_build_tdx_data_context_rejects_missing_download_output(
    tmp_path: Path,
) -> None:
    module = load_example_module()

    with pytest.raises(DownloadError, match="exactly one path"):
        module.build_tdx_data_context(
            MappingOnlyDownloader({"510300.SH": tmp_path / "510300.SH.parquet"}),
            ["510300.SH", "159915.SZ"],
            "2024-01-01",
            "2024-01-31",
            tmp_path,
        )


def test_build_tdx_data_context_rejects_extra_download_output(
    tmp_path: Path,
) -> None:
    module = load_example_module()

    with pytest.raises(DownloadError, match="exactly one path"):
        module.build_tdx_data_context(
            MappingOnlyDownloader(
                {
                    "510300.SH": tmp_path / "510300.SH.parquet",
                    "159915.SZ": tmp_path / "159915.SZ.parquet",
                    "600519.SH": tmp_path / "600519.SH.parquet",
                }
            ),
            ["510300.SH", "159915.SZ"],
            "2024-01-01",
            "2024-01-31",
            tmp_path,
        )


def test_build_tdx_data_context_requires_downloaded_parquet_files(
    tmp_path: Path,
) -> None:
    module = load_example_module()

    with pytest.raises(SymbolNotFoundError, match="No data for '510300.SH'"):
        module.build_tdx_data_context(
            MappingOnlyDownloader(
                {
                    "510300.SH": tmp_path / "510300.SH.parquet",
                    "159915.SZ": tmp_path / "159915.SZ.parquet",
                }
            ),
            ["510300.SH", "159915.SZ"],
            "2024-01-01",
            "2024-01-31",
            tmp_path,
        )
