er_config_list=(
    "200_15_0.2_0.01_0.01"
    "200_15_0.2_0.03_0.03"
    "200_15_0.2_0.05_0.05"
    "200_30_0.2_0.01_0.01"
    "200_30_0.2_0.03_0.03"
    "200_30_0.2_0.05_0.05"
    "200_45_0.2_0.01_0.01"
    "200_45_0.2_0.03_0.03"
    "200_45_0.2_0.05_0.05"
    "200_60_0.2_0.01_0.01"
    "200_60_0.2_0.03_0.03"
    "200_60_0.2_0.05_0.05"
)

small_world_config_list=(
    "200_15_3_0.2_0.01"
    "200_15_3_0.2_0.03"
    "200_15_3_0.2_0.05"
    "200_30_6_0.2_0.01"
    "200_30_6_0.2_0.03"
    "200_30_6_0.2_0.05"
    "200_45_9_0.2_0.01"
    "200_45_9_0.2_0.03"
    "200_45_9_0.2_0.05"
    "200_60_12_0.2_0.01"
    "200_60_12_0.2_0.03"
    "200_60_12_0.2_0.05"
)

mkdir -p bin
mkdir -p metric
g++ -std=c++11 -O3 KHCore.cpp -o run_khcore -fopenmp

n_generated=10


# ER
for er_config in ${er_config_list[@]}; do        
    p_graph="generated_graphs/er/${er_config}"
    for ((i=0; i<n_generated; i++)); do
        p_bin="bin/er/${er_config}"
        mkdir -p ${p_bin}        

        p_txt="${p_graph}/graph_${i}.txt"
        p_bin="${p_bin}/graph_${i}.bin"

        if [ ! -f "$p_bin" ]; then
            ./run_khcore txt-to-bin "$p_txt" "$p_bin"
        fi    

        p_metric="metric/er/${er_config}/${i}"
        mkdir -p ${p_metric}
            
        for d in 1 2 3 4 5 6; do
            if [ ! -f "${p_metric}/khcore_${d}.txt" ]; then
                ./run_khcore decompose "$p_bin" $d 16 >> "${p_metric}/khcore_${d}.txt"
            fi
        done
    done
done

# Small-world
for small_world_config in ${small_world_config_list[@]}; do
    p_graph="generated_graphs/small_world/${small_world_config}"
    for ((i=0; i<n_generated; i++)); do
        p_bin="bin/small_world/${small_world_config}"    
        mkdir -p ${p_bin}        

        p_txt="${p_graph}/graph_${i}.txt"
        p_bin="${p_bin}/graph_${i}.bin"

        if [ ! -f "$p_bin" ]; then
            ./run_khcore txt-to-bin "$p_txt" "$p_bin"
        fi

        p_metric="metric/small_world/${small_world_config}/${i}"
        mkdir -p ${p_metric}

        for d in 1 2 3 4 5 6; do
            if [ ! -f "${p_metric}/khcore_${d}.txt" ]; then
                ./run_khcore decompose "$p_bin" $d 16 >> "${p_metric}/khcore_${d}.txt"
            fi
        done
    done
done