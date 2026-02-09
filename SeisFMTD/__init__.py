"""
SeisFMTD: Seismic Full Moment Tensor Determination
===================================================

A Python package for seismic full moment tensor inversion using 
Hamiltonian Monte Carlo (HMC) methods.

Submodules
----------
pyCAPLunar : Core CAP (Cut-and-Paste) waveform processing
pyCAPSolvers : HMC solvers for moment tensor inversion
MTTools : Moment tensor operations and conversions
"""

__version__ = "0.1.0"
__author__ = "Gang Yang"

from . import pyCAPLunar
from . import pyCAPSolvers
from . import MTTools
