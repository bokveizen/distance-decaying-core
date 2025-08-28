# Hyperparameter sensitivity analysis

## Overview

This folder contains the source code for evaluating how the choices of hyperparameters affect the performance of our algorithm.

- Corresponding parts in the paper: Section 6.6 (Q5)
- Source code files:  
  - `main_ours.cpp`: the proposed algorithm
  - `run_hyperparam.sh`: the script for running the proposed algorithm with different hyperparameters

## How to run

### 1. Compile the code

```bash
make
```

### 2. Run the algorithms with different configurations

```bash
bash run_hyperparam.sh poly 1
bash run_hyperparam.sh poly 5
bash run_hyperparam.sh poly 10
bash run_hyperparam.sh exp 2
bash run_hyperparam.sh exp 5
bash run_hyperparam.sh exp 10
```

### 3. Read the results

Check the log files in the `log` folder.
