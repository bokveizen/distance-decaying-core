# Scalability analysis

## Overview

This folder contains the source code for evaluating how the running time of our algorithm scales with the size of the graph.

- Corresponding parts in the paper: Section 6.5 (Q4)
- Source code files:  
  - `main_naive.cpp`: the naive peeling algorithm
  - `main_ours.cpp`: the proposed algorithm
  - `bfs_traversal.cpp`: BFS traversal
  - `bfs_subgraph.cpp`: subgraph generation based on BFS traversal sequence
  - `generate_bfs.sh`: script for generating BFS traversal sequences
  - `generate_subgraph.sh`: script for generating subgraphs based on BFS traversal sequences
  - `generate_lastfm.sh`: script for generating subgraphs for the LastFM dataset
  - `run_lastfm_naive.sh`: script for running the naive peeling algorithm on the generated subgraphs  
  - `run_lastfm_ours.sh`: script for running the proposed algorithm on the generated subgraphs

## How to run

### 1. Generate subgraphs using snowball sampling

```bash
bash generate_lastfm.sh 42
```

### 2. Run the computational code

```bash
bash run_lastfm_ours.sh 42
bash run_lastfm_naive.sh 42
```

### 3. Read the results

Check the log files in the `log` folder.
