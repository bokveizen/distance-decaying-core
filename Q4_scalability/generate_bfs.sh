dataset=$1

# Build the BFS traversal program
echo "Building BFS traversal program..."
mkdir -p build
g++ -O3 bfs_traversal.cpp -std=c++17 -o build/bfs_traversal -fopenmp -march=native

echo "BFS traversal program built successfully."

# Set up paths
network="../data/${dataset}.txt"
p_output="bfs_seq"
# Create output directories
mkdir -p "${p_output}/${dataset}"

# Get starting node from input argument
start_node=$2

echo "Generating BFS sequence from starting node: $start_node"

# Counter for progress
count=0
    
f_output="${p_output}/${dataset}/${start_node}.txt"
    
# Run BFS traversal
./build/bfs_traversal "$network" "$start_node" "$f_output"

