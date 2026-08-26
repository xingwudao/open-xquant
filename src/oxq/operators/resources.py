"""Access packaged operator contract resources as local paths."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.resources import as_file, files
from pathlib import Path

_FROZEN_SURFACE = {
    "quant_panel_schema": "quant-panel-v1.schema.json",
    "operator_manifest_schema": "operator-manifest-v1.schema.json",
    "operator_binding_schema": "operator-binding-v1.schema.json",
    "reference_validator": "reference_validator_v1.py",
}
_CERTIFICATION_PROFILE = {
    "provider_catalog": "provider-catalog-v1.schema.json",
    "candidate_build": "candidate-build-v1.schema.json",
    "numerical_baseline": "numerical-baseline-v1.schema.json",
    "certification_record": "certification-record-v1.schema.json",
}


def _source_contract_directory(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "contracts" / name


@contextmanager
def _materialize_resources(
    packaged_directory: str,
    source_directory: str,
    names: dict[str, str],
) -> Iterator[dict[str, Path]]:
    directory = files("oxq.operators").joinpath(packaged_directory)
    if directory.is_dir():
        with ExitStack() as stack:
            yield {
                name: Path(stack.enter_context(as_file(directory.joinpath(filename))))
                for name, filename in names.items()
            }
        return

    fallback = _source_contract_directory(source_directory)
    if not fallback.is_dir():
        raise FileNotFoundError(f"operator contract resources are unavailable: {fallback}")
    yield {name: fallback / filename for name, filename in names.items()}


@contextmanager
def materialize_contract_surface() -> Iterator[dict[str, Path]]:
    """Yield local paths for the byte-frozen operator contract surface."""
    with _materialize_resources("contracts/v1", "quant-operators", _FROZEN_SURFACE) as paths:
        yield paths


@contextmanager
def materialize_certification_profile() -> Iterator[dict[str, Path]]:
    """Yield local paths for the certification intake profile schemas."""
    with _materialize_resources(
        "certification_profile/v1",
        "operator-certification",
        _CERTIFICATION_PROFILE,
    ) as paths:
        yield paths
