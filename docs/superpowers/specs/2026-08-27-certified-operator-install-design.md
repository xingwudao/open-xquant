# Certified Operator Environment Install Design

## Status

Revised on 2026-08-27 after the managed installed-release store failed
security review. This document supersedes the earlier self-managed install
store design.

## Goal

Make certified provider operators usable when the provider package is installed
as a normal Python distribution in the same environment as `open-xquant`.

The first supported user flow is:

```bash
pip install equant-core==1.0.0 equant-ttr==1.0.0
oxq operator verify equant-py==1.0.0
```

After verification, `open-xquant` can discover and invoke the certified
`equant-py` operators from the installed Python environment.

## Core decision

`open-xquant` is not a package manager.

Phase one delegates package installation to standard Python tooling such as
`pip`, `uv pip`, Poetry, or the user's environment manager. `open-xquant` only
owns certification policy, metadata validation, operator discovery, and
research-runtime invocation.

## Responsibilities

### `equant-py`

`equant-py` is the certified operator provider name. Its installed code is
packaged as canonical split Python distributions such as `equant-core` and
`equant-ttr`. It owns:

- operator implementation code;
- packaged operator manifests;
- packaged numerical baseline evidence;
- packaged provider catalog or equivalent discovery metadata;
- normal Python distribution metadata and versioning.

The certified release is a normal wheel set such as
`equant_core-1.0.0-*.whl` and `equant_ttr-1.0.0-*.whl`.

### `open-xquant`

`open-xquant` owns:

- official provider/version policy;
- certification records and bundle validation;
- environment package discovery;
- installed package version checks;
- packaged manifest and baseline digest checks;
- operator lookup and research invocation;
- user-facing CLI verification.

`open-xquant` does not store provider wheels in its own managed install
directory and does not install provider code into its SDK bundle.

## Trust model

Phase one trusts only exact provider versions that are listed in the
`open-xquant` official certification index.

The index binds:

- provider name, initially `equant-py`;
- Python distribution closure, initially `equant-core` and `equant-ttr`;
- exact version, initially `1.0.0`;
- expected package metadata;
- expected manifest paths and SHA-256 digests;
- expected baseline paths and SHA-256 digests;
- certified operator IDs and versions;
- certification state, initially `research-certified`.

If the installed package is missing, has a different version, omits certified
metadata, or contains bytes that do not match the index, verification fails.

## User-facing commands

Verification:

```bash
oxq operator verify equant-py==1.0.0
```

Listing:

```bash
oxq operator list --provider equant-py
```

Optional convenience install can be added later:

```bash
oxq operator install equant-py==1.0.0
```

In phase one, `operator install` must not implement its own wheel store. It may
print the exact `pip install` command and then run verification after the user
has installed the package, or it may be omitted.

## Runtime model

The first implementation may import provider code from the current Python
environment after verification. This is acceptable because the user explicitly
installed the provider package into that environment.

The runtime still checks certification state before resolving an operator. A
strategy cannot silently use an uncertified provider version.

Exact dependency isolation is a future enhancement. It is not required for
phase one.

## What is removed from the earlier design

The revised phase one removes:

- self-managed installed-release store;
- local wheel closure storage;
- `OPEN_XQUANT_OPERATOR_HOME`;
- path-based atomic release publication;
- exact-wheel child runtime as a requirement for installed providers;
- remote GitHub Release download as part of `oxq operator install`;
- `open-xquant` acting as a package manager.

Certification bundle export/import may remain useful for producing and
auditing official evidence, but importing a bundle is not the mechanism that
installs provider code.

## Non-goals

This design does not provide:

- automatic dependency solving;
- automatic package installation;
- arbitrary third-party provider trust;
- cryptographic publisher signatures;
- live-trading or broker authority;
- runtime sandboxing against malicious installed Python packages.

## Acceptance criteria

- `oxq operator verify equant-py==1.0.0` succeeds only when the exact certified
  package version is installed.
- Verification fails on missing package, wrong version, missing manifest,
  changed manifest bytes, missing baseline, or changed baseline bytes.
- `oxq operator list --provider equant-py` lists certified operators from the
  verified installed package.
- `open-xquant` no longer needs an installed-release store to use certified
  operators.
- The documentation states clearly that users install provider packages with
  normal Python package tooling.
