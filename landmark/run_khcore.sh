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

mkdir -p ../bin
mkdir -p ../metric
g++ -std=c++11 -O3 KHCore.cpp -o run_khcore -fopenmp

for ds_name in ${dataset_list[@]}; do
    p_txt="../data/${ds_name}.txt"
    p_bin="../bin/${ds_name}.bin"  

    if [ ! -f "$p_bin" ]; then
        ./run_khcore txt-to-bin "$p_txt" "$p_bin"
    fi

    mkdir -p ../metric/${ds_name}
    for d in 1 2 3 4 5 6; do
        if [ ! -f "../metric/${ds_name}/khcore_${d}.txt" ]; then
            ./run_khcore decompose "$p_bin" $d 16 >> ../metric/${ds_name}/khcore_${d}.txt
        fi
    done
done