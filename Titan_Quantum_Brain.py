"""Compatibility wrapper for the Titan quantum brain modules.

The project scripts import ``TitanBrain`` or reference ``Titan_Quantum_Brain`` as a
module. This file exposes the public class name expected by the rest of the repo
without requiring any external quantum dependencies at import time.
"""

from Titan_Neural_Architecture import TitanBrain

TitanQuantumBrain = TitanBrain

__all__ = ["TitanBrain", "TitanQuantumBrain"]
