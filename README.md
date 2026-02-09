# SeisFMTD

**Seismic Full Moment Tensor Determination using Hamiltonian Monte Carlo**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SeisFMTD is a Python package for seismic full moment tensor inversion using Hamiltonian Monte Carlo (HMC) methods. It provides tools for:

- Full moment tensor inversion with uncertainty quantification
- Cut-and-paste waveform fitting
- Green's function handling and processing

## Installation

### From source

```bash
git clone https://github.com/GangYang-gy/SeisFMTD.git
cd SeisFMTD
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy
- SciPy
- ObsPy
- Matplotlib
- [Pyrocko](https://pyrocko.org/) - Seismology toolbox
- [MTfit](https://github.com/djpugh/MTfit) - Moment tensor fitting
- utm - UTM coordinate conversion
- seisgen - Seismogram generation (may require separate installation)

## Package Structure

```
SeisFMTD/                          # Repository root
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── examples/                      # Usage examples (not part of package)
│   ├── run_hmc_mt_example.py
│   └── run_hmc_mtd_example.py
└── SeisFMTD/                      # Python package
    ├── __init__.py
    ├── pyCAPLunar/                # Core CAP functionality
    │   ├── DCAP.py
    │   ├── DCut.py
    │   ├── DPaste.py
    │   ├── DFilters.py
    │   ├── DMisfit.py
    │   └── GTools.py
    ├── pyCAPSolvers/              # HMC solvers
    │   ├── DHMC_linear.py
    │   └── DHMC_mtd_linear.py
    └── MTTools/                   # Moment tensor utilities
        ├── DMomentTensors.py
        ├── DLune.py
        └── DOrientations.py
```

## Quick Start

```python
from pyCAPLunar import DCAP
from pyCAPSolvers import DHMC_linear

# Example usage (to be added)
```

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- **`run_hmc_example.py`** - Complete template for HMC moment tensor inversion

## Documentation

Documentation is coming soon.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{seisfmtd,
  author = {Yang, Gang},
  title = {SeisFMTD: Seismic Full Moment Tensor Determination},
  year = {2026},
  url = {https://github.com/GangYang-gy/SeisFMTD}
}
```

## Contact

- Gang Yang - [GitHub](https://github.com/GangYang-gy)
