bash generate_bfs.sh soc-lastfm 42

k_list=(10000 20000 40000 80000 160000 320000 640000)

for k in "${k_list[@]}"; do
    bash generate_subgraph.sh soc-lastfm 42 $k
done