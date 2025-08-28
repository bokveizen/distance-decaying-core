input=main_ours.cpp
output=main_ours
g++ -O3 $input -std=c++17 -o $output -fopenmp -march=native

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

exp_list=(2 5 10)
poly_list=(1 5 10)

p_graph=../data

# Process each network
for dataset in ${dataset_list[@]}; do
    network="${p_graph}/${dataset}.txt"    

    echo "$dataset start"

    mkdir -p ../metric/$dataset

    for exp in ${exp_list[@]}; do
        p_output_exp=../metric/$dataset/exp_${exp}.txt
        if [ ! -f "$p_output_exp" ]; then
            ./$output -i "$network" -o "$p_output_exp" -d exp -p ${exp}
        fi
    done

    for poly in ${poly_list[@]}; do
        p_output_poly=../metric/$dataset/poly_${poly}.txt
        if [ ! -f "$p_output_poly" ]; then
            ./$output -i "$network" -o "$p_output_poly" -d poly -p ${poly}
        fi
    done
        
    echo "$dataset done"
done