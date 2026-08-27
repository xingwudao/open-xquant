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
_DISTRIBUTION_PROFILE = {
    "certification_record_v2": "certification-record-v2.schema.json",
    "certification_bundle_manifest": "certification-bundle-manifest-v1.schema.json",
}
_INSTALL_PROFILE = {
    "operator_release": "operator-release-v1.schema.json",
    "runtime_protocol": "operator-runtime-protocol-v1.schema.json",
    "official_providers": "official-providers-v1.json",
    "official_environment_providers": "official-environment-providers-v1.json",
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


@contextmanager
def materialize_operator_distribution_profile() -> Iterator[dict[str, Path]]:
    """Yield local paths for distribution certification schemas."""
    package = files("oxq.operators")
    packaged = {
        "certification_record_v2": package.joinpath(
            "distribution_profile/v1/certification-record-v2.schema.json"
        ),
        "certification_bundle_manifest": package.joinpath(
            "distribution_profile/v1/certification-bundle-manifest-v1.schema.json"
        ),
    }
    if all(path.is_file() for path in packaged.values()):
        with ExitStack() as stack:
            yield {name: Path(stack.enter_context(as_file(path))) for name, path in packaged.items()}
        return
    yield {
        "certification_record_v2": _source_contract_directory("operator-certification")
        / "certification-record-v2.schema.json",
        "certification_bundle_manifest": _source_contract_directory("operator-distribution")
        / "certification-bundle-manifest-v1.schema.json",
    }


@contextmanager
def materialize_operator_install_profile() -> Iterator[dict[str, Path]]:
    """Yield local paths for operator installation contracts."""
    with _materialize_resources(
        "install_profile/v1",
        "operator-install",
        _INSTALL_PROFILE,
    ) as paths:
        yield paths
