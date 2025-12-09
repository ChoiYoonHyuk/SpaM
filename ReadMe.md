# Sparse Bayesian Message Passing under Structural Uncertainty

<img width="2074" height="1056" alt="Image" src="https://github.com/user-attachments/assets/076b8c37-3f6d-48b1-aa83-fed6639efe4c" />

## Overview

We propose **Sparse Bayesian Message Passing under Structural Uncertainty (SpaM)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python model.py [dataset]
- [dataset] $\rightarrow$ 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
