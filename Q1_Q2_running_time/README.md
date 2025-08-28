# Running time analysis

## Overview

This folder contains the source code for evaluating the running time of different algorithms for distance-decaying core decomposition.

- Corresponding parts in the paper: Section 6.2 (Q1) and Section 6.3 (Q2)
- Source code files:
  - `main_naive.cpp`: the naive peeling algorithm
  - `main_ours.cpp`: the proposed algorithm
  - `main_noGF.cpp`: the proposed algorithm without global filters
  - `main_noLF.cpp`: the proposed algorithm without local filters
  - `main_noFB.cpp`: the proposed algorithm without fallback from-scratch update
  - `Makefile`: the Makefile for compiling the code
  - `run_all.sh`: the script for running the algorithms

## How to run

### 1. Compile the code

```bash
make
```

### 2. Run the algorithms with different configurations

```bash
bash run_all.sh poly 1
bash run_all.sh poly 5
bash run_all.sh poly 10
bash run_all.sh exp 2
bash run_all.sh exp 5
bash run_all.sh exp 10
```

### 3. Read the results

Check the log files in the `log` folder.
