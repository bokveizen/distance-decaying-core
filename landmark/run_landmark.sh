# Compute errors of shortest path computation using landmarks

input=compute_errors.cpp
output=compute_errors

g++ -O3 $input -std=c++17 -o $output -fopenmp -march=native

p_graph=../data

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
)

metric_list=(
    "khcore_1"
    "khcore_2"
    "khcore_3"
    "khcore_4"
    "khcore_5"
    "khcore_6"
    "poly_1"
    "poly_5"
    "poly_10"
    "exp_2"
    "exp_5"
    "exp_10"    
)

k_list=(5 10 20 50 100)

p_output=landmark_errors
mkdir -p $p_output

for dataset in ${dataset_list[@]}; do
    network="${p_graph}/${dataset}.txt"
    mkdir -p $p_output/$dataset

    for k in ${k_list[@]}; do
        p_landmark=landmark/$dataset/$k
        p_output_k=$p_output/$dataset/$k

        for metric in ${metric_list[@]}; do
            p_landmark_metric=$p_landmark/$metric
            p_res_k=$p_output_k/$metric
            mkdir -p $p_res_k
            for i in {0..9}; do
                ./$output $network $p_landmark_metric/${i}.txt $p_res_k/${i}.txt
            done
        done
    done
done