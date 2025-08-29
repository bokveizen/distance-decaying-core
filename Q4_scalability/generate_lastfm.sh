init_node=$1

bash generate_bfs.sh soc-lastfm $init_node

k_list=(10000 20000 40000 80000 160000 320000 640000)

for k in "${k_list[@]}"; do
    bash generate_subgraph.sh soc-lastfm $init_node $k
done