# Sparse Bayesian Message Passing under Structural Uncertainty

<img width="4547" height="1024" alt="Image" src="https://github.com/user-attachments/assets/525f9306-4647-4483-85df-fda2909976ff" />

## Overview

We propose **Sparse Bayesian Message Passing under Structural Uncertainty (SpaM)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python model.py [dataset]
- [dataset] $\rightarrow$ 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
