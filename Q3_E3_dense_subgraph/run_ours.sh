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

input=main_ours.cpp
output=main_ours
g++ -O3 $input -std=c++17 -o $output -fopenmp -march=native

exp_list=(2 5 10)
poly_list=(1 5 10)

n_generated=10

# ER
for er_config in ${er_config_list[@]}; do        
    p_graph="generated_graphs/er/${er_config}"
    for ((i=0; i<n_generated; i++)); do
        mkdir -p metric/er/${er_config}/${i}
        # Exp
        for exp in ${exp_list[@]}; do
            p_output_exp=metric/er/${er_config}/${i}/exp_${exp}.txt
            if [ ! -f "$p_output_exp" ]; then
                ./$output -i "$p_graph/graph_${i}.txt" -o "$p_output_exp" -d exp -p ${exp}
            fi
        done
        # Poly
        for poly in ${poly_list[@]}; do
            p_output_poly=metric/er/${er_config}/${i}/poly_${poly}.txt
            if [ ! -f "$p_output_poly" ]; then
                ./$output -i "$p_graph/graph_${i}.txt" -o "$p_output_poly" -d poly -p ${poly}
            fi
        done
    done
done

# Small-world
for small_world_config in ${small_world_config_list[@]}; do
    p_graph="generated_graphs/small_world/${small_world_config}"
    for ((i=0; i<n_generated; i++)); do
        mkdir -p metric/small_world/${small_world_config}/${i}
        # Exp
        for exp in ${exp_list[@]}; do
            p_output_exp=metric/small_world/${small_world_config}/${i}/exp_${exp}.txt
            if [ ! -f "$p_output_exp" ]; then
                ./$output -i "$p_graph/graph_${i}.txt" -o "$p_output_exp" -d exp -p ${exp}
            fi
        done
        # Poly
        for poly in ${poly_list[@]}; do
            p_output_poly=metric/small_world/${small_world_config}/${i}/poly_${poly}.txt
            if [ ! -f "$p_output_poly" ]; then
                ./$output -i "$p_graph/graph_${i}.txt" -o "$p_output_poly" -d poly -p ${poly}
            fi
        done
    done
done