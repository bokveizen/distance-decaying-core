g++ -O3 bfs_subgraph.cpp -std=c++17 -o build/bfs_subgraph -march=native

dataset=$1
start_node=$2
k=$3

network="../data/${dataset}.txt"
bfs_sequence="bfs_seq/${dataset}/${start_node}.txt"
subgraph_output="subgraphs/${dataset}/${start_node}_${k}.txt"

mkdir -p subgraphs/${dataset}

./build/bfs_subgraph "$network" "$bfs_sequence" "$k" "$subgraph_output"