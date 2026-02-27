from __future__ import annotations

import os
from pathlib import Path


def resolve_data_dir(dest_dir: Path | None = None) -> Path:
    """Resolve data storage directory. Priority: parameter > OXQ_DATA_DIR > default."""
    if dest_dir is not None:
        return dest_dir
    env = os.environ.get("OXQ_DATA_DIR")
    if env:
        return Path(env) / "market"
    return Path.home() / ".oxq" / "data" / "market"
