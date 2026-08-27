# Operator Environment Acceptance

The official environment provider route verifies installed package bytes from a
normal Python environment. For `equant-py==1.0.0`, the canonical provider name
and installed Python distribution are both `equant-py`.

The official index in
`contracts/operator-install/official-environment-providers-v1.json` pins:

- Certification state: `research-certified`
- Certified operators: 60
- Manifest paths: `compat/open_xquant/manifests/*.operator.json`
- Baseline path: `compat/open_xquant/numerical_baselines/technical-v1.json`
- Runtime modules: `ettr` and `equant`
- Manifest, baseline, and runtime-file SHA-256 digests computed from the
  packaged `equant-py` wheel contents

Acceptance uses a temporary Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install /path/to/open-xquant /path/to/equant_py-1.0.0-py3-none-any.whl
.venv/bin/oxq operator verify equant-py==1.0.0 --json
.venv/bin/oxq operator list --provider equant-py --json
```

Both `oxq` commands must report `research-certified` and exactly 60 operators.
Runtime acceptance also resolves `equant.ttr.sma@1.0.0` through
`resolve_environment_operator()` and calls the returned callable on a minimal
valid panel. Before importing the manifest module, open-xquant checks that
`importlib` resolves it to a verified runtime file from the installed
`equant-py` distribution and rechecks that file's SHA-256 digest.
