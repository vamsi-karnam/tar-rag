# Installing AcmeKit

AcmeKit is distributed as a single Python package on PyPI. Install it
into a virtual environment with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install acmekit
```

## System requirements

- Python 3.10 or later
- ~50 MB of free disk space for the package + dependencies
- An internet connection on first run (for the bundled config download)

## Optional extras

If you want the visualisation panel, install the `viz` extra:

```bash
pip install "acmekit[viz]"
```

## Verifying the install

After installation, run the bundled smoke check:

```bash
acmekit --version
acmekit doctor
```

`acmekit doctor` walks through the import path, the configured cache
directory, and the optional dependencies. A green result means you are
ready to run the first-run walkthrough in `first_run.md`.
