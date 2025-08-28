# High-coreness nodes as dense-subgraph detectors

## Overview

This folder contains the source code for evaluating different core models for detecting planted dense subgraphs.

- Corresponding parts in the paper: Section 6.4 (Q3-E3)
- Source code files:
  - `KHCore.cpp` and `KHCore.h`: computational algorithm for distance-generalized core decomposition
    - The source code is cloned from [the official repository](https://github.com/BITDataScience/khcore).
    - **Reference:** Dai et al. "Scaling up distance-generalized core decomposition." CIKM'21
  - `main_ours.cpp`: the proposed algorithm for distance-decaying core decomposition
  - `gen_graph.py`: generating random graphs with planted dense subgraphs
  - `read_res.py`: result analysis
  - `run_khcore.sh`: the script for running the distance-generalized core decomposition
  - `run_ours.sh`: the script for running the proposed algorithm

## How to run

### 1. Generate graphs with planted dense subgraphs

```bash
python gen_graph.py
```

### 2. Compute the metrics

```bash
bash run_ours.sh
bash run_khcore.sh
```

### 3. Read the results

```bash
python read_res.py
```
