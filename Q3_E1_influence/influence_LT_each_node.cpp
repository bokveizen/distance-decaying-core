/*
 * Linear Threshold (LT) Model Simulation with OpenMP Parallelization
 * This version tests each node as a single seed and outputs influence to file
 * 
 * Compilation: g++ -std=c++17 -O3 -fopenmp -o influence_LT_each_node influence_LT_each_node.cpp
 * Or simply use: make
 * 
 * Usage: ./influence_LT_each_node <graph_file> <n_sim> <edge_weight> <threshold_distribution> <output_file>
 * Example: ./influence_LT_each_node sample_graph.txt 1000 0.1 uniform results.txt
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <unordered_map>
#include <queue>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

struct Edge {
    int to;
    double weight;
};

class Graph {
public:
    int n;
    vector<vector<Edge>> adj;
    
    Graph(int n) : n(n), adj(n) {}
    
    void addEdge(int u, int v, double weight) {
        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight}); // For undirected graph
    }
    
    // Normalize edge weights so that incoming weights to each node sum to at most 1
    void normalizeWeights() {
        vector<double> in_weight_sum(n, 0.0);
        
        // Calculate total incoming weight for each node
        for (int u = 0; u < n; u++) {
            for (const Edge& edge : adj[u]) {
                in_weight_sum[edge.to] += edge.weight;
            }
        }
        
        // Normalize weights
        for (int u = 0; u < n; u++) {
            for (Edge& edge : adj[u]) {
                if (in_weight_sum[edge.to] > 1.0) {
                    edge.weight /= in_weight_sum[edge.to];
                }
            }
        }
    }
};

// Generate threshold for a node based on distribution type
double generateThreshold(const string& distribution, mt19937& gen) {
    if (distribution == "uniform") {
        uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    } else if (distribution == "normal") {
        normal_distribution<double> dist(0.5, 0.2);
        double threshold = dist(gen);
        return max(0.0, min(1.0, threshold)); // Clamp to [0,1]
    } else if (distribution == "low") {
        uniform_real_distribution<double> dist(0.0, 0.3);
        return dist(gen);
    } else if (distribution == "high") {
        uniform_real_distribution<double> dist(0.7, 1.0);
        return dist(gen);
    } else {
        // Default to uniform if unknown distribution
        uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    }
}

// LT Model Simulation
int simulateLT(const Graph& graph, const vector<int>& seeds, const string& threshold_dist, mt19937& gen) {
    vector<bool> activated(graph.n, false);
    vector<double> thresholds(graph.n);
    
    // Generate thresholds for all nodes
    for (int i = 0; i < graph.n; i++) {
        thresholds[i] = generateThreshold(threshold_dist, gen);
    }
    
    // Initialize with seed nodes
    queue<int> newly_activated;
    for (int seed : seeds) {
        if (seed >= 0 && seed < graph.n && !activated[seed]) {
            activated[seed] = true;
            newly_activated.push(seed);
        }
    }
    
    // Propagate activation in waves
    while (!newly_activated.empty()) {
        vector<int> candidates_to_check;
        
        // Collect all neighbors of newly activated nodes
        while (!newly_activated.empty()) {
            int current = newly_activated.front();
            newly_activated.pop();
            
            for (const Edge& edge : graph.adj[current]) {
                int neighbor = edge.to;
                if (!activated[neighbor]) {
                    candidates_to_check.push_back(neighbor);
                }
            }
        }
        
        // Remove duplicates
        sort(candidates_to_check.begin(), candidates_to_check.end());
        candidates_to_check.erase(unique(candidates_to_check.begin(), candidates_to_check.end()), 
                                 candidates_to_check.end());
        
        // Check each candidate for activation
        for (int candidate : candidates_to_check) {
            if (activated[candidate]) continue;
            
            // Calculate total influence from activated neighbors
            double total_influence = 0.0;
            for (const Edge& edge : graph.adj[candidate]) {
                if (activated[edge.to]) {
                    total_influence += edge.weight;
                }
            }
            
            // Activate if threshold is exceeded
            if (total_influence >= thresholds[candidate]) {
                activated[candidate] = true;
                newly_activated.push(candidate);
            }
        }
    }
    
    // Count activated nodes
    int activated_count = 0;
    for (bool isActive : activated) {
        if (isActive) activated_count++;
    }
    
    return activated_count;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <n_sim> <edge_weight> <threshold_distribution> <output_file>" << endl;
        cerr << "  graph_file_path: path to graph file with format 'node1 node2'" << endl;
        cerr << "  n_sim: number of simulation repetitions per node" << endl;
        cerr << "  edge_weight: uniform weight for all edges (0.0 to 1.0)" << endl;
        cerr << "  threshold_distribution: 'uniform', 'normal', 'low', or 'high'" << endl;
        cerr << "  output_file: path to output file for results" << endl;
        return 1;
    }

    string file_path = argv[1];
    int n_sim = stoi(argv[2]);
    double edge_weight = stod(argv[3]);
    string threshold_dist = argv[4];
    string output_file = argv[5];

    if (n_sim <= 0) {
        cerr << "Error: n_sim must be positive" << endl;
        return 1;
    }

    if (edge_weight < 0.0 || edge_weight > 1.0) {
        cerr << "Error: edge_weight must be between 0.0 and 1.0" << endl;
        return 1;
    }

    // Validate threshold distribution
    if (threshold_dist != "uniform" && threshold_dist != "normal" && 
        threshold_dist != "low" && threshold_dist != "high") {
        cerr << "Error: threshold_distribution must be 'uniform', 'normal', 'low', or 'high'" << endl;
        return 1;
    }

    // Read graph
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        return 1;
    }

    set<int> unique_vertices;
    vector<pair<int, int>> edges;
    string line;

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
    Graph graph(n);

    cout << "Number of nodes: " << n << ", number of edges: " << edges.size() << endl;
    cout << "Edge weight: " << edge_weight << ", Threshold distribution: " << threshold_dist << endl;

    for (const auto& [u, v] : edges) {
        int mapped_u = vertex_map[u];
        int mapped_v = vertex_map[v];
        graph.addEdge(mapped_u, mapped_v, edge_weight);
    }

    // Normalize edge weights to ensure valid LT model
    graph.normalizeWeights();

    // Open output file
    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file " << output_file << endl;
        return 1;
    }

    // Write header
    outfile << "node_id,mean_influence,std_dev,min_influence,max_influence" << endl;

    cout << "Running " << n_sim << " simulations for each of " << n << " nodes..." << endl;
    
    // Get number of threads for progress reporting
    int num_threads = omp_get_max_threads();
    cout << "Using " << num_threads << " threads for parallel execution" << endl;

    // Test each node as a single seed
    for (int node_idx = 0; node_idx < n; node_idx++) {
        vector<int> single_seed = {node_idx};
        vector<int> results(n_sim);
        double sum = 0.0;
        double sum_squares = 0.0;

        #pragma omp parallel reduction(+:sum,sum_squares)
        {
            // Each thread gets its own random number generator with different seed
            int thread_id = omp_get_thread_num();
            auto seed = chrono::high_resolution_clock::now().time_since_epoch().count() + thread_id + node_idx * 1000;
            mt19937 gen(seed);
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < n_sim; i++) {
                int activated_count = simulateLT(graph, single_seed, threshold_dist, gen);
                results[i] = activated_count;
                sum += activated_count;
                sum_squares += activated_count * activated_count;
            }
        }

        // Calculate statistics
        double mean = sum / n_sim;
        double variance = (sum_squares / n_sim) - (mean * mean);
        double std_dev = sqrt(variance);
        int min_val = *min_element(results.begin(), results.end());
        int max_val = *max_element(results.begin(), results.end());

        // Output results (using original node ID)
        int original_node_id = reverse_map[node_idx];
        outfile << original_node_id << "," << fixed << setprecision(6) 
                << mean << "," << std_dev << "," << min_val << "," << max_val << endl;

        // Progress reporting
        if ((node_idx + 1) % 100 == 0 || node_idx == n - 1) {
            cout << "Progress: " << (node_idx + 1) << "/" << n << " nodes completed..." << endl;
        }
    }

    outfile.close();
    cout << "Results written to " << output_file << endl;

    return 0;
} 