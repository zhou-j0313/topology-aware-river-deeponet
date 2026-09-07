# Topology-Aware DeepONet for River Networks

This repository provides a public research baseline for learning unsteady-flow
operators on branched river networks. It combines a topology-aware DeepONet
with time-series boundary conditioning, reach-level graph message passing, and
hard enforcement of initial, external-boundary, and junction constraints.

> **Public-release note**
>
> The paper-specific composite loss and its weighting strategy are not included.
> This release uses masked mean-squared error, boundary mean-squared error, and
> a simple first-order spatiotemporal consistency regularizer. Peak and junction
> soft-loss hooks are retained for API compatibility but disabled by default.

## Features

- Multi-reach river-network topology with repeated junction endpoints.
- Multiple upstream-discharge and downstream-stage boundary conditions.
- Topology-aware branch encoder and continuous-coordinate DeepONet trunks.
- Full, sparse, or boundary-only supervision.
- Variable event lengths with padding and explicit time masks.
- Irregular cross-section geometry and differentiable hydraulic lookup tables.
- Training, validation, prediction, diagnostics, and Excel result export.
- Three compact benchmark events in English CSV format.

## Repository layout

```text
.
|-- data/
|   |-- cross_sections.csv
|   |-- river_network.csv
|   |-- boundary_sections.csv
|   |-- case_001/
|   |-- case_002/
|   `-- case_003/
|-- src/
|   `-- river_deeponet.py
|-- tests/
|   `-- test_public_baseline.py
|-- DATASET.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately when GPU acceleration is
required. The code automatically falls back to CPU when CUDA is unavailable.

## Quick start

Run a one-epoch smoke test before launching a full experiment.

Linux/macOS:

```bash
EPOCHS=1 python src/river_deeponet.py
```

Windows PowerShell:

```powershell
$env:EPOCHS = "1"
python src/river_deeponet.py
```

For a full run, omit `EPOCHS` or set it to the desired number. Optional
environment variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `EPOCHS` | `3000` | Number of training epochs |
| `BATCH_SIZE` | `1` | Event batch size |
| `DEVICE` | `cuda` | Preferred PyTorch device |
| `CASE_ROOT_DIR` | `data/` | Directory containing case folders |
| `PINN_CASE_PREFIX` | `case_` | Case-directory prefix |

The default split uses `case_001` and `case_002` for training and `case_003`
for boundary-driven validation. Validation targets are used only for evaluation.

## Input data

All model inputs are CSV files. See [DATASET.md](DATASET.md) for schemas,
units, and the role of each file. The bundled data intentionally excludes
solver executables, raw solver work directories, logs, and temporary files.

## Outputs

The default script writes:

- `training_diagnostics.xlsx`: loss history and topology diagnostics.
- `training_results/`: event-level predictions and comparison tables.
- `prediction_results.xlsx`: optional prediction output when a
  `prediction_case/` directory is supplied.

Generated outputs and model checkpoints are ignored by Git.

## Reproducibility

NumPy and PyTorch random seeds are fixed to 42. Exact GPU results can still vary
across hardware, CUDA, cuDNN, and PyTorch versions. Record the package versions,
device, and training configuration used for published experiments.

## Citation

If this repository supports your work, cite the associated paper and this
software release. Update [CITATION.cff](CITATION.cff) with the final paper DOI,
authors, and repository URL before creating a release.

## License

The code and bundled public dataset are released under the MIT License. See
[LICENSE](LICENSE). Confirm that you have the right to redistribute any data
you add to this repository.
