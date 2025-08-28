#!/bin/bash
mkdir -p build
g++ -O3 main_ours.cpp -std=c++17 -o build/main_ours -fopenmp -march=native

dataset_list=(
    "web-polblogs"
    "web-google"
    "soc-wiki-Vote"
    "web-edu"
    "web-BerkStan"
    "web-webbase-2001"
    "web-spam"
    "web-indochina-2004"   
    "ca-CondMat"
    "soc-epinions"
    "ca-HepPh"
    "ca-AstroPh"
    "soc-brightkite"
    "soc-douban"
    "web-sk-2005"
    "soc-slashdot"
    "soc-Slashdot0811"
    "ca-dblp-2010"
    "soc-gowalla"
    "soc-delicious"
    "soc-youtube"
    "soc-flickr"
    "soc-lastfm"    
)

decay_type=$1
decay_param=$2
p_output="output/${decay_type}_${decay_param}"
p_log="log/${decay_type}_${decay_param}"
mkdir -p $p_output
mkdir -p $p_log

config_list=(
    "1 2 0.01"
    "3 2 0.01"
    "2 1 0.01"
    "2 3 0.01"
    "2 2 0.001"
    "2 2 0.1"
)


for dataset in "${dataset_list[@]}"; do
    network="../data/${dataset}.txt"

    mkdir -p "${p_output}/${dataset}"
    mkdir -p "${p_log}/${dataset}"

    for config in "${config_list[@]}"; do
        IFS=' ' read -r k_global k_local t <<< "$config"

        echo "Running $dataset with config $config"

        f_output="${p_output}/${dataset}/${k_global}_${k_local}_${t}.txt"
        f_log="${p_log}/${dataset}/${k_global}_${k_local}_${t}.log"
    
        ./build/main_ours -i "$network" -o "${f_output}" -d "$decay_type" -p "$decay_param" -kg "$k_global" -kl "$k_local" -t "$t" > "${f_log}"

        echo "Done $dataset with config $config"
    done
done