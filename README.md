# PatchAnalyzer (Development)

PatchAnalyzer is a PyQt5-based graphical user interface for streamlined analysis of patch clamp electrophysiology data.
The development branch contains the latest features under active development.

## Features
- Load metadata and electrophysiology sweeps from files
- Current-clamp and voltage-clamp analysis views
- Group and map pages for organizing experiments
- Logging utilities to track processing steps

## Requirements
- Python 3.11
- PyQt5
- numpy, pandas
- matplotlib, pyqtgraph
- scipy
- pyabf

## Conda environment
Create and activate a conda environment named `patchanalyzer` with Python 3.11:

```bash
conda create -n patchanalyzer python=3.11
conda activate patchanalyzer
```

## Installation
With the environment active, install the required dependencies:

```bash
pip install PyQt5 numpy pandas matplotlib pyqtgraph scipy pyabf
```

## Usage
Run the application from the project root:

```bash
python start.py
```

This launches the main PatchAnalyzer window.

## License
This project is licensed under the [MIT License](LICENSE).