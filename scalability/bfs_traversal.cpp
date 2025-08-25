/*
 * BFS Traversal with OpenMP Parallelization
 * 
 * Compile with: g++ -fopenmp -O3 -o bfs_traversal bfs_traversal.cpp -std=c++17
 * 
 * Environment variables:
 *   OMP_NUM_THREADS=<number> - Set number of threads
 *   OMP_SCHEDULE=<type>      - Set scheduling policy
 */

#include <bits/stdc++.h>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <iomanip>

using namespace std;

struct Graph {
    int n;
    vector<unordered_set<int>> adj;
    
    Graph(int _n) : n(_n) {
        adj.resize(_n);
    }

    Graph() : n(0) {}

    void addEdge(int u, int v) {
        if (u >= 0 && u < n && v >= 0 && v < n) {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
};

// Progress bar utility class
class ProgressBar {
private:
    int total;
    int current;
    int bar_width;
    double start_time;
    
public:
    ProgressBar(int total_items, int width = 50) : total(total_items), current(0), bar_width(width) {
        start_time = omp_get_wtime();
    }
    
    void update(int visited_count) {
        current = visited_count;
        double progress = (double)current / total;
        int pos = bar_width * progress;
        
        // Calculate ETA
        double elapsed = omp_get_wtime() - start_time;
        double eta = (elapsed / progress) - elapsed;
        
        cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " << fixed << setprecision(1) << (progress * 100.0) << "% ";
        cout << "(" << current << "/" << total << ") ";
        
        if (current > 0 && current < total) {
            cout << "ETA: " << fixed << setprecision(1) << eta << "s";
        } else if (current == total) {
            cout << "Completed in " << fixed << setprecision(1) << elapsed << "s";
        }
        
        cout.flush();
    }
    
    void finish() {
        update(total);
        cout << endl;
    }
};

// BFS traversal function with OpenMP optimizations and progress bar
vector<int> bfsTraversal(const Graph& G, int start_node) {
    vector<int> traversal_order;
    vector<bool> visited(G.n, false);
    queue<int> q;
    
    // Initialize progress bar
    ProgressBar progress(G.n);
    cout << "BFS Progress:" << endl;
    
    // Start BFS from the given node
    q.push(start_node);
    visited[start_node] = true;
    traversal_order.push_back(start_node);
    progress.update(1);
    
    while (!q.empty()) {
        // Process current level
        int level_size = q.size();
        vector<int> current_level;
        vector<vector<int>> level_neighbors(level_size);
        
        // Extract current level nodes
        for (int i = 0; i < level_size; i++) {
            current_level.push_back(q.front());
            q.pop();
        }
        
        // Parallel processing of neighbors for current level
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < level_size; i++) {
            int current = current_level[i];
            // Convert unordered_set to vector and sort for consistent ordering
            vector<int> neighbors(G.adj[current].begin(), G.adj[current].end());
            sort(neighbors.begin(), neighbors.end());
            level_neighbors[i] = neighbors;
        }
        
        // Sequential addition of new nodes (to maintain BFS order)
        int nodes_added_this_level = 0;
        for (int i = 0; i < level_size; i++) {
            for (int neighbor : level_neighbors[i]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                        traversal_order.push_back(neighbor);
                        nodes_added_this_level++;
                    }
                }
            }
        }
        
        // Update progress bar after processing this level
        if (nodes_added_this_level > 0) {
            progress.update(traversal_order.size());
        }
    }
    
    // Finish progress bar
    progress.finish();
    cout << endl;
    
    return traversal_order;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <initial_node> <output_file_path>" << endl;
        cerr << "  graph_file_path: path to the input graph file (edge list format)" << endl;
        cerr << "  initial_node: the starting node for BFS traversal" << endl;
        cerr << "  output_file_path: path to the output file for BFS sequence" << endl;
        return 1;
    }

    string file_path = argv[1];
    int initial_node = stoi(argv[2]);
    string output_path = argv[3];

    // Read graph
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        return 1;
    }

    set<int> unique_vertices;
    vector<pair<int, int>> edges;
    string line;

    // Read edges from file
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v))
            continue;
        unique_vertices.insert(u);
        unique_vertices.insert(v);
        edges.push_back({u, v});
    }
    file.close();

    // Create vertex mapping and graph
    unordered_map<int, int> vertex_map;
    vector<int> reverse_map;
    int new_id = 0;
    for (int v : unique_vertices) {
        vertex_map[v] = new_id++;
        reverse_map.push_back(v);
    }

    int n = unique_vertices.size();
    Graph G(n);

    cout << "Number of nodes: " << n << ", number of edges: " << edges.size() << endl;

    // Check if initial node exists in the graph
    if (vertex_map.find(initial_node) == vertex_map.end()) {
        cerr << "Error: Initial node " << initial_node << " not found in the graph" << endl;
        return 1;
    }

    // Build the graph with parallel edge processing
    cout << "Building graph with OpenMP..." << endl;
    double start_time = omp_get_wtime();
    
    // Note: Graph construction needs to be sequential due to shared data structure
    // But we can parallelize the edge mapping preparation
    vector<pair<int, int>> mapped_edges(edges.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < edges.size(); i++) {
        int u = edges[i].first;
        int v = edges[i].second;
        mapped_edges[i] = {vertex_map[u], vertex_map[v]};
    }
    
    // Sequential graph construction (adjacency lists are not thread-safe)
    for (const auto &[mapped_u, mapped_v] : mapped_edges) {
        G.addEdge(mapped_u, mapped_v);
    }
    
    double end_time = omp_get_wtime();
    cout << "Graph construction completed in " << (end_time - start_time) << " seconds" << endl;

    // Get the mapped initial node
    int mapped_initial_node = vertex_map[initial_node];

    // Perform BFS traversal with timing
    cout << "Starting BFS from node " << initial_node << " (mapped to " << mapped_initial_node << ")" << endl;
    cout << "Using " << omp_get_max_threads() << " OpenMP threads" << endl;
    
    start_time = omp_get_wtime();
    vector<int> bfs_sequence = bfsTraversal(G, mapped_initial_node);
    end_time = omp_get_wtime();
    
    cout << "BFS traversal completed in " << (end_time - start_time) << " seconds" << endl;

    // Write results to output file
    ofstream output_file(output_path);
    if (!output_file.is_open()) {
        cerr << "Error: Could not open output file " << output_path << endl;
        return 1;
    }

    cout << "BFS traversal visits " << bfs_sequence.size() << " nodes" << endl;
    
    // Write the BFS sequence (convert back to original node IDs)
    for (int mapped_node : bfs_sequence) {
        output_file << reverse_map[mapped_node] << endl;
    }
    output_file.close();

    cout << "BFS traversal completed. Results written to " << output_path << endl;

    // Check if all nodes were visited (connected component)
    if (bfs_sequence.size() == n) {
        cout << "All nodes were visited - the graph is connected" << endl;
    } else {
        cout << "Only " << bfs_sequence.size() << " out of " << n 
             << " nodes were visited - the graph has multiple connected components" << endl;
    }

    return 0;
} 