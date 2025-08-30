# High-coreness nodes as high-influence indicators

## Overview

This folder contains the source code for evaluating different core models for identifying influential nodes.

- Corresponding parts in the paper: Section 6.4 (Q3-E1)
- Source code files:
  - `KHCore.cpp` and `KHCore.h`: computational algorithm for distance-generalized core decomposition
    - The source code is cloned from [the official repository](https://github.com/BITDataScience/khcore).
    - **Reference:** Dai et al. "Scaling up distance-generalized core decomposition." CIKM'21
  - `main_ours.cpp`: the proposed algorithm for distance-decaying core decomposition
  - `influence_{IC/SIR}_each_node.cpp`: simulation of IC/SIR model
  - `read_res.py`: result analysis
  - `Makefile`: the Makefile for compiling the code
  - `run_khcore.sh`: the script for running the distance-generalized core decomposition
  - `run_ours.sh`: the script for running the proposed algorithm
  - `run_simu.sh`: the script for running the simulation

## How to run

### 1. Compile the code

```bash
make
```

### 2. Run the simulation

```bash
bash run_simu.sh
```

### 3. Compute the metrics

```bash
bash run_ours.sh
bash run_khcore.sh
```

### 4. Read the results

```bash
python read_res.py
```
