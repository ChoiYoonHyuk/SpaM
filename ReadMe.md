# Sparse Tri-State Message Passing with Predictive Marginalization

<img width="4547" height="1024" alt="Image" src="https://github.com/user-attachments/assets/92befa69-d4d1-4ebe-b2ba-ef066c254192" />

## Overview

We propose **Sparse Tri-State Message Passing with Predictive Marginalization (SpaM)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python main.py [dataset]
- [dataset] $\rightarrow$ 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
