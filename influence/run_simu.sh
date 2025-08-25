p_data=../data

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

mkdir -p IC_simu
mkdir -p LT_simu
mkdir -p SIR_simu

for dataset in ${dataset_list_p2[@]}; do
    mkdir -p IC_simu/${dataset}
    mkdir -p LT_simu/${dataset}
    mkdir -p SIR_simu/${dataset}

    # IC
    p_list=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
    for p in ${p_list[@]}; do
        ./influence_IC_each_node $p_data/${dataset}.txt 10000 $p IC_simu/${dataset}/${p}.txt
    done
    
    # LT
    p_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
    for p in ${p_list[@]}; do
        ./influence_LT_each_node $p_data/${dataset}.txt 10000 $p uniform LT_simu/${dataset}/${p}.txt
    done

    # SIR
    p_list=(0.01 0.03 0.05 0.08 0.12 0.17 0.25 0.35 0.50 0.75)
    for p in ${p_list[@]}; do
        ./influence_SIR_each_node $p_data/${dataset}.txt 10000 $p 1.0 SIR_simu/${dataset}/${p}.txt
    done
done
