# PeerJ AI Application reproducibility materials

This folder is a clean, read-only-derived copy of the materials for:

**The impact of training protocol and evaluation robustness on performance assessment in weakly supervised whole-slide image classification**

This repository contains the code and supporting materials accompanying the PeerJ Computer Science AI Application submission. It was assembled from the retained experiment workspace without modifying the original records.

## Folder map

- `code/`: model, feature-extraction, training, evaluation, and statistical-analysis code.
- `metadata/`: public slide identifiers, labels, and dataset partitions. Local filesystem paths were removed.
- `results/`: selected result files corresponding to the manuscript analyses.
- `figures/`: selected figures and interpretability outputs.
- `docs/`: CED-MIL-lite design documentation.

## What is included

- ABMIL, TransMIL, CLAM, DSMIL, and CED-MIL-lite implementations.
- WSI scanning, tissue filtering, patch feature extraction, training, and evaluation utilities.
- Multi-seed and patient-level cross-validation drivers.
- Bootstrap confidence intervals, paired permutation tests, DeLong analysis, and ensemble/model-soup utilities.
- TCGA-COAD/READ, MUC-versus-NOS, and Camelyon-16 manifests.
- Selected final summaries and fold-level logs for the analyses used in the manuscript.

## What is not included

- Original TCGA or Camelyon-16 whole-slide images.
- Pretrained third-party encoder weights.
- Pre-extracted Phikon-v2 or UNI feature tensors; the corresponding folders in the original workspace were empty.
- Restricted third-party data or model files whose redistribution is controlled by their original providers.

## Installation

The project source declares Python 3.11 or later and PyTorch 2.2 or later. The
reported hardware platform was Ubuntu Linux with an NVIDIA GeForce RTX 5090
(32 GB). Because this GPU uses the NVIDIA Blackwell architecture, the reference
reproduction environment uses Python 3.11, PyTorch 2.7, and CUDA 12.8.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./code
```

Install the CUDA 12.8 build of PyTorch 2.7 before installing the project. The
remaining dependency list in `code/pyproject.toml` was assembled from the
project imports; OpenSlide and encoder-specific packages may additionally be
required for feature extraction.

## Computing environment

```text
Operating system: Ubuntu Linux
GPU: NVIDIA GeForce RTX 5090
GPU memory: 32 GB
Python: 3.11 (reference reproduction environment)
PyTorch: 2.7 (reference reproduction environment)
CUDA: 12.8 (reference reproduction environment)
CPU and RAM: not required to use the supplied fold-level result records
cuDNN: supplied with the selected PyTorch CUDA distribution
```

## Data preparation

1. Obtain TCGA-COAD/READ slides from the NCI Genomic Data Commons under the applicable access and use terms.
2. Obtain Camelyon-16 from its official challenge/data source under its applicable terms.
3. Place raw slides outside this public repository, for example under `data/raw_wsi/`.
4. Select the manifest corresponding to the manuscript analysis being reproduced. The package intentionally retains separate 117-slide, 95-slide, and 101-slide TCGA manifests because the reported analyses use distinct archived cohorts; these cohorts must not be treated as numerically interchangeable.
5. Generate patch features with the feature-extraction command in `wsi_hint.cli`, or deposit the exact feature tensors used in a suitable repository if redistribution is permitted.

## Example commands

The package exposes the CLI through Python:

```bash
python -m wsi_hint.cli --help
python -m wsi_hint.cli extract-features --help
python -m wsi_hint.cli benchmark-kfold --help
python -m wsi_hint.cli protocol-plugin-ablation-multiseed --help
```

Canonical multi-seed driver:

```bash
python code/scripts/run_multi_seed.py canonical-coad-read --dry-run
```

Before executing it, update the selected manifest and feature directory for the local data installation.

## Result map

See `RESULTS_INDEX.md` for the mapping between result folders and manuscript analyses.

## Data sources

- TCGA-COAD and TCGA-READ: NCI Genomic Data Commons, https://portal.gdc.cancer.gov/
- Camelyon-16: official challenge website, https://camelyon16.grand-challenge.org/

The original whole-slide images remain subject to the access and use conditions of their respective providers.
