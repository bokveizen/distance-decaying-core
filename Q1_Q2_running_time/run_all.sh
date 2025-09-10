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

for dataset in ${dataset_list[@]}; do
    network="../data/${dataset}.txt"
    mkdir -p "${p_output}/${dataset}"
    mkdir -p "${p_log}/${dataset}"

    echo "$dataset start"

    ./build/main_naive -i "$network" -o "${p_output}/${dataset}/naive.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/naive.log"
    ./build/main_ours -i "$network" -o "${p_output}/${dataset}/ours.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/ours.log"
    ./build/main_noGF -i "$network" -o "${p_output}/${dataset}/noGF.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/noGF.log"
    ./build/main_noLF -i "$network" -o "${p_output}/${dataset}/noLF.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/noLF.log"
    ./build/main_noFB -i "$network" -o "${p_output}/${dataset}/noFB.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/noFB.log"
    ./build/main_noPR -i "$network" -o "${p_output}/${dataset}/noPR.txt" -d $decay_type -p $decay_param > "${p_log}/${dataset}/noPR.log"
    
    # print the dataset name
    echo "$dataset done"
done