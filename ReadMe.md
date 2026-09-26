# Persistent Tri-State Message Passing

<img width="2547" height="1024" alt="Image" src="https://github.com/user-attachments/assets/9ca7425b-b16d-4595-be59-9676dc6621f2" />

## Overview

We propose **Persistent Tri-State Message Passing (P3MP)** that performs local sparse coding over neighbors and aggregates positive/negative relations through sign-aware channels, yielding robust node representations under noisy and heterophilic edges. 

## Execution

To train SpaM on nine heterophilic benchmarks ([dataset] from 0 to 8):
- python main.py [dataset]
- [dataset] $\rightarrow$ 0: RomanEmpire, 1: Minesweeper, ..., 8: Wisconsin
