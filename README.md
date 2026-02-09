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
SeisFMTD/
├── pyCAPLunar/       # Core CAP (Cut-and-Paste) functionality
│   ├── DCAP.py       # Main CAP inversion module
│   ├── DCut.py       # Waveform cutting utilities
│   ├── DPaste.py     # Waveform pasting/stitching
│   ├── DFilters.py   # Signal filtering functions
│   ├── DMisfit.py    # Misfit calculations
│   └── GTools.py     # Green's function tools
├── pyCAPSolvers/     # HMC solvers
│   ├── DHMC_linear.py      # Linear HMC solver for full moment tensor
│   └── DHMC_mtd_linear.py  # Linear HMC solver for full moment tensor plus depth
├── MTools/     # Moment tensor tools
    ├── DMomentTensors.py   # Moment tensor operations
    ├── DLune.py            # Lune diagram tools
    └── DOrientations.py    # Orientation conversions
```

## Quick Start

```python
from pyCAPLunar import DCAP
from pyCAPSolvers import DHMC_linear

# Example usage (to be added)
```

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
