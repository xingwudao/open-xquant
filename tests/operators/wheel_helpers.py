"""Wheel fixtures for untrusted-archive validation tests."""

from __future__ import annotations

import base64
import csv
import hashlib
import stat
import zipfile
from collections.abc import Callable
from io import StringIO
from pathlib import Path

from oxq.operators.release_index import ReleaseTarget, ReleaseWheel


def wheel_record(
    path: Path,
    *,
    distribution: str = "equant-core",
    version: str = "1.0.0",
    requires: tuple[str, ...] = (),
    tag: str = "py3-none-any",
    metadata_text: str | None = None,
    wheel_text: str | None = None,
    entries: dict[str, bytes] | None = None,
    record: str | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
    mutate_info: Callable[[zipfile.ZipInfo], None] | None = None,
) -> ReleaseWheel:
    """Write a small valid wheel, with optional hostile archive mutation."""
    filename = f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    contents = {f"{distribution.replace('-', '_')}/__init__.py": b""}
    if entries:
        contents.update(entries)
    metadata = metadata_text or (
        "Metadata-Version: 2.1\nName: "
        + distribution
        + "\nVersion: "
        + version
        + "\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requires)
    )
    contents[f"{dist_info}/METADATA"] = metadata.encode()
    wheel_metadata = wheel_text or f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"
    contents[f"{dist_info}/WHEEL"] = wheel_metadata.encode()
    record_name = f"{dist_info}/RECORD"
    if record is None:
        rows = []
        for name, value in sorted(contents.items()):
            digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            rows.append((name, f"sha256={digest}", str(len(value))))
        rows.append((record_name, "", ""))
        stream = StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        record = stream.getvalue()
    contents[record_name] = record.encode()
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, value in contents.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            if mutate_info is not None:
                mutate_info(info)
            archive.writestr(info, value)
    if mutate_info is encrypted_info:
        raw = bytearray(path.read_bytes())
        for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            start = 0
            while (index := raw.find(signature, start)) >= 0:
                raw[index + offset] |= 0x01
                start = index + 4
        path.write_bytes(raw)
    data = path.read_bytes()
    return ReleaseWheel(
        filename,
        "https://github.com/example/" + filename,
        len(data),
        "sha256:" + hashlib.sha256(data).hexdigest(),
        distribution,
        version,
        "implementation",
        (tag,),
    )


def target(*wheels: ReleaseWheel) -> ReleaseTarget:
    return ReleaseTarget("cp312", "cp312", "macosx_14_0_arm64", wheels[0], wheels)


def symlink_info(info: zipfile.ZipInfo) -> None:
    info.external_attr = (stat.S_IFLNK | 0o777) << 16


def encrypted_info(info: zipfile.ZipInfo) -> None:
    info.flag_bits |= 0x1


def fifo_info(info: zipfile.ZipInfo) -> None:
    info.external_attr = (stat.S_IFIFO | 0o644) << 16
