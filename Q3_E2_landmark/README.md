# High-coreness nodes as shortest-path landmarks

## Overview

This folder contains the source code for evaluating different core models for selecting shortest-path landmarks.

- Corresponding parts in the paper: Section 6.4 (Q3-E2)
- Source code files:
  - `KHCore.cpp` and `KHCore.h`: computational algorithm for distance-generalized core decomposition
    - The source code is cloned from [the official repository](https://github.com/BITDataScience/khcore).
    - **Reference:** Dai et al. "Scaling up distance-generalized core decomposition." CIKM'21
  - `main_ours.cpp`: the proposed algorithm for distance-decaying core decomposition
  - `compute_errors.cpp`: estimation error computation
  - `generate_landmarks.py`: landmark generation
  - `read_res.py`: result analysis
  - `run_khcore.sh`: the script for running the distance-generalized core decomposition
  - `run_ours.sh`: the script for running the proposed algorithm
  - `run_landmark.sh`: the script for estimation error computation using the generated landmarks

## How to run

### 1. Generate landmarks

```bash
python generate_landmarks.py
```

### 2. Compute the metrics

```bash
bash run_ours.sh
bash run_khcore.sh
```

### 3. Compute the relative errors

```bash
bash run_landmark.sh
```

### 4. Read the results

```bash
python read_res.py
```
