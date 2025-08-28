#!/bin/bash
mkdir -p build
g++ -O3 main_naive.cpp -std=c++17 -o build/main_naive -fopenmp -march=native

dataset="soc-lastfm"
p_output="output/naive"
p_log="log/naive"
mkdir -p $p_output
mkdir -p $p_log

type="poly"
param="1.0"
k_global=2
k_local=2

k_list=(10000 20000 40000 80000 160000 320000 640000)

p_input="subgraphs/${dataset}"
init_node=42

mkdir -p "${p_output}/${dataset}"
mkdir -p "${p_log}/${dataset}"

for k in "${k_list[@]}"; do
    network="${p_input}/${init_node}_${k}.txt"
    f_output="${p_output}/${dataset}/${k}.txt"
    f_log="${p_log}/${dataset}/${k}.log"

    ./build/main_naive -i "$network" -o "${f_output}" -d "$type" -p "$param" -kg "$k_global" -kl "$k_local" > "${f_log}"
done