"""Resolve verified environment operator callables for research use."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from oxq.operators.environment_index import CertifiedOperatorRef
from oxq.operators.environment_provider import InstalledEnvironmentProvider, verify_installed_provider
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.formats import sha256_bytes


@dataclass(frozen=True)
class EnvironmentOperatorBinding:
    operator_id: str
    operator_version: str
    provider_requirement: str
    manifest: Mapping[str, object]
    callable: Callable[..., object]


@dataclass(frozen=True)
class _VerifiedModuleSource:
    module_name: str
    path: Path
    digest: str
    is_package: bool


_TRUSTED_RUNTIME_MODULES: dict[str, tuple[Path, str, ModuleType]] = {}
_RUNTIME_RESOLUTION_LOCK = threading.RLock()


def resolve_environment_operator(
    operator_id: str,
    operator_version: str,
    provider_requirement: str,
) -> EnvironmentOperatorBinding:
    """Return a callable binding from a verified installed environment provider."""
    installed = verify_installed_provider(provider_requirement)
    if installed.provider.certification_state != "research-certified":
        raise _error(
            "environment_provider_not_research_certified",
            "environment provider is not research-certified",
            operator_id,
        )

    operator = _find_certified_operator(installed, operator_id, operator_version)
    manifest = installed.manifests.get(operator.manifest_path)
    if manifest is None:
        raise _error(
            "environment_operator_manifest_missing",
            "certified environment operator manifest is missing",
            operator_id,
        )
    _validate_manifest_identity(manifest, operator_id, operator_version)
    _validate_manifest_certification_state(manifest, operator_id)

    module_name = manifest.get("module")
    callable_name = manifest.get("callable")
    if not isinstance(module_name, str) or not isinstance(callable_name, str):
        raise _error(
            "environment_operator_implementation_invalid",
            "certified environment operator implementation is invalid",
            operator_id,
        )

    with _RUNTIME_RESOLUTION_LOCK:
        sources = _verified_module_sources(installed)
        origin = _verified_module_origin(sources, module_name, operator_id)
        _reject_untrusted_preloaded_modules(sources, operator_id)
        try:
            module = _import_verified_module(module_name, sources)
        except ImportError as exc:
            raise _error(
                "environment_operator_module_unavailable",
                "certified environment operator module is unavailable",
                operator_id,
            ) from exc
        except Exception as exc:
            raise _error(
                "environment_operator_module_unavailable",
                "certified environment operator module is unavailable",
                operator_id,
            ) from exc
        _verify_module_object(module, origin, sources[module_name].digest, operator_id)
        implementation = getattr(module, callable_name, None)
        if not callable(implementation):
            raise _error(
                "environment_operator_callable_missing",
                "certified environment operator callable is unavailable",
                operator_id,
            )
        _verify_callable_owner(implementation, sources, operator_id)
        protected_callable = _verified_callable(
            module_name,
            callable_name,
            origin,
            sources,
            operator_id,
        )

    return EnvironmentOperatorBinding(
        operator_id=operator_id,
        operator_version=operator_version,
        provider_requirement=provider_requirement,
        manifest=manifest,
        callable=protected_callable,
    )


def _verified_module_sources(
    installed: InstalledEnvironmentProvider,
) -> dict[str, _VerifiedModuleSource]:
    sources: dict[str, _VerifiedModuleSource] = {}
    for runtime in installed.runtime_files.values():
        try:
            path = runtime.path.resolve(strict=True)
        except OSError:
            continue
        module_name, is_package = _module_name_from_runtime_path(runtime.package_path)
        if module_name is None:
            continue
        sources[module_name] = _VerifiedModuleSource(
            module_name=module_name,
            path=path,
            digest=runtime.digest,
            is_package=is_package,
        )
    return sources


def _module_name_from_runtime_path(package_path: str) -> tuple[str | None, bool]:
    path = Path(package_path)
    if path.suffix != ".py":
        return None, False
    if path.name == "__init__.py":
        parts = path.parts[:-1]
        if not parts:
            return None, False
        return ".".join(parts), True
    return ".".join((*path.parts[:-1], path.stem)), False


def _verified_module_origin(
    sources: Mapping[str, _VerifiedModuleSource],
    module_name: str,
    operator_id: str,
) -> Path:
    source = sources.get(module_name)
    if source is None:
        raise _error(
            "environment_operator_module_unavailable",
            "certified environment operator module is unavailable",
            operator_id,
        )
    return _verify_source_path(source, operator_id)


def _verify_source_path(
    source: _VerifiedModuleSource,
    operator_id: str,
) -> Path:
    if not source.path.is_file() or source.path.is_symlink():
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        )
    try:
        raw = source.path.read_bytes()
    except OSError as exc:
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        ) from exc
    if sha256_bytes(raw) != source.digest:
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        )
    return source.path


def _reject_untrusted_preloaded_modules(
    sources: Mapping[str, _VerifiedModuleSource],
    operator_id: str,
) -> None:
    provider_roots = _provider_runtime_roots(sources)
    for module_name, loaded in tuple(sys.modules.items()):
        if loaded is None:
            continue
        if module_name.partition(".")[0] not in provider_roots:
            continue
        source = sources.get(module_name)
        trusted = _TRUSTED_RUNTIME_MODULES.get(module_name)
        if source is None or trusted is None or trusted[2] is not loaded:
            raise _error(
                "environment_operator_module_preloaded",
                "certified environment operator module is already loaded",
                operator_id,
            )
        _verify_module_object(loaded, source.path, source.digest, operator_id)


def _import_verified_module(
    module_name: str,
    sources: Mapping[str, _VerifiedModuleSource],
) -> ModuleType:
    _drop_trusted_runtime_modules(sources)
    modules_before = set(sys.modules)
    try:
        with _verified_runtime_importer(sources):
            module = importlib.import_module(module_name)
    except BaseException:
        _cleanup_failed_provider_modules(modules_before, sources)
        raise
    for name, source in sources.items():
        loaded_source = sys.modules.get(name)
        if loaded_source is not None:
            _TRUSTED_RUNTIME_MODULES[name] = (source.path, source.digest, loaded_source)
    return module


def _verified_callable(
    module_name: str,
    callable_name: str,
    origin: Path,
    sources: Mapping[str, _VerifiedModuleSource],
    operator_id: str,
) -> Callable[..., object]:
    def invoke(*args: object, **kwargs: object) -> object:
        with _RUNTIME_RESOLUTION_LOCK:
            _reject_untrusted_preloaded_modules(sources, operator_id)
            module = _import_verified_module(module_name, sources)
            _verify_module_object(module, origin, sources[module_name].digest, operator_id)
            implementation = getattr(module, callable_name, None)
            if not callable(implementation):
                raise _error(
                    "environment_operator_callable_missing",
                    "certified environment operator callable is unavailable",
                    operator_id,
                )
            _verify_callable_owner(implementation, sources, operator_id)
            modules_before = set(sys.modules)
            with _verified_runtime_importer(sources):
                try:
                    result = implementation(*args, **kwargs)
                except BaseException:
                    _cleanup_failed_provider_modules(modules_before, sources)
                    raise
            _record_trusted_runtime_modules(sources)
            return result

    invoke.__name__ = callable_name
    invoke.__qualname__ = callable_name
    return invoke


def _record_trusted_runtime_modules(
    sources: Mapping[str, _VerifiedModuleSource],
) -> None:
    for name, source in sources.items():
        loaded_source = sys.modules.get(name)
        if loaded_source is not None:
            _TRUSTED_RUNTIME_MODULES[name] = (source.path, source.digest, loaded_source)


def _drop_trusted_runtime_modules(
    sources: Mapping[str, _VerifiedModuleSource],
) -> None:
    for module_name in sorted(sources, key=lambda name: name.count("."), reverse=True):
        loaded = sys.modules.get(module_name)
        trusted = _TRUSTED_RUNTIME_MODULES.get(module_name)
        if loaded is not None and trusted is not None and trusted[2] is loaded:
            sys.modules.pop(module_name, None)
            _TRUSTED_RUNTIME_MODULES.pop(module_name, None)


def _cleanup_failed_provider_modules(
    modules_before: set[str],
    sources: Mapping[str, _VerifiedModuleSource],
) -> None:
    provider_roots = _provider_runtime_roots(sources)
    for module_name in tuple(sys.modules):
        if module_name in modules_before:
            continue
        if module_name.partition(".")[0] not in provider_roots:
            continue
        sys.modules.pop(module_name, None)
        _TRUSTED_RUNTIME_MODULES.pop(module_name, None)


def _provider_runtime_roots(
    sources: Mapping[str, _VerifiedModuleSource],
) -> frozenset[str]:
    return frozenset(name.partition(".")[0] for name in sources)


@contextmanager
def _verified_runtime_importer(
    sources: Mapping[str, _VerifiedModuleSource],
) -> Iterator[None]:
    finder = _VerifiedRuntimeFinder(sources)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass


class _VerifiedRuntimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources: Mapping[str, _VerifiedModuleSource]) -> None:
        self._sources = sources
        self._provider_roots = _provider_runtime_roots(sources)

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        source = self._sources.get(fullname)
        if source is None:
            if fullname.partition(".")[0] in self._provider_roots:
                raise ImportError(
                    f"provider runtime module is not declared: {fullname}",
                    name=fullname,
                )
            return None
        loader = _VerifiedRuntimeLoader(source)
        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=str(source.path),
            is_package=source.is_package,
        )
        if source.is_package:
            spec.submodule_search_locations = [str(source.path.parent)]
        return spec


class _VerifiedRuntimeLoader(importlib.abc.Loader):
    def __init__(self, source: _VerifiedModuleSource) -> None:
        self._source = source

    def create_module(
        self,
        spec: importlib.machinery.ModuleSpec,
    ) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        raw = self._source.path.read_bytes()
        if sha256_bytes(raw) != self._source.digest:
            raise ImportError("verified runtime module digest mismatch")
        module.__file__ = str(self._source.path)
        module.__loader__ = self
        if self._source.is_package:
            module.__package__ = self._source.module_name
            module.__path__ = [str(self._source.path.parent)]  # type: ignore[attr-defined]
        else:
            module.__package__ = self._source.module_name.rpartition(".")[0]
        code = compile(raw, str(self._source.path), "exec")
        exec(code, module.__dict__)


def _verify_module_object(
    module: ModuleType,
    origin: Path,
    digest: str,
    operator_id: str,
) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        )
    try:
        module_origin = Path(module_file).resolve(strict=True)
    except OSError as exc:
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        ) from exc
    if module_origin != origin:
        raise _error(
            "environment_operator_module_unverified",
            "certified environment operator module origin is unverified",
            operator_id,
        )
    _verify_source_path(
        _VerifiedModuleSource(module.__name__, origin, digest, hasattr(module, "__path__")),
        operator_id,
    )


def _verify_callable_owner(
    implementation: Callable[..., object],
    sources: Mapping[str, _VerifiedModuleSource],
    operator_id: str,
) -> None:
    owner = inspect.getmodule(implementation)
    if owner is None:
        raise _error(
            "environment_operator_callable_unverified",
            "certified environment operator callable owner is unverified",
            operator_id,
        )
    source = sources.get(owner.__name__)
    trusted = _TRUSTED_RUNTIME_MODULES.get(owner.__name__)
    if source is None or trusted is None or trusted[2] is not owner:
        raise _error(
            "environment_operator_callable_unverified",
            "certified environment operator callable owner is unverified",
            operator_id,
        )
    _verify_module_object(owner, source.path, source.digest, operator_id)


def _find_certified_operator(
    installed: InstalledEnvironmentProvider,
    operator_id: str,
    operator_version: str,
) -> CertifiedOperatorRef:
    for operator in installed.provider.operators:
        if operator.operator_id == operator_id and operator.operator_version == operator_version:
            return operator
    raise _error(
        "environment_operator_not_certified",
        f"environment operator is not certified: {operator_id}@{operator_version}",
        operator_id,
    )


def _validate_manifest_identity(
    manifest: Mapping[str, object],
    operator_id: str,
    operator_version: str,
) -> None:
    if (
        manifest.get("operator_id") != operator_id
        or manifest.get("operator_version") != operator_version
    ):
        raise _error(
            "environment_operator_manifest_mismatch",
            "certified environment operator manifest identity does not match",
            operator_id,
        )


def _validate_manifest_certification_state(
    manifest: Mapping[str, object],
    operator_id: str,
) -> None:
    state = manifest.get("certification_state")
    if state != "research-certified":
        raise _error(
            "environment_operator_not_research_certified",
            "environment operator manifest is not research-certified",
            operator_id,
        )


def _error(
    code: str,
    message: str,
    operator_id: str,
) -> OperatorCertificationError:
    return OperatorCertificationError(
        code,
        message,
        stage="environment_runtime",
        operator_id=operator_id,
    )
