# Sparse Bayesian Message Passing under Structural Uncertainty

<img width="1261" height="465" alt="Image" src="https://github.com/user-attachments/assets/da743d28-547d-4ad7-82de-f1409f4ac1ef" />

## Overview

We propose **Sparse Bayesian Message Passing under Structural Uncertainty (SpaM)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python model.py [dataset]
- [dataset] $\rightarrow$ 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
