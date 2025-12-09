# Sparse Bayesian Message Passing under Structural Uncertainty


## Overview

We propose **Sparse Bayesian Message Passing under Structural Uncertainty (SpaM)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python SpaM.py [dataset]
- 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
