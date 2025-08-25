/*
 * Independent Cascade (IC) Model Simulation with OpenMP Parallelization
 * This version tests each node as a single seed and outputs influence to file
 * 
 * Compilation: g++ -std=c++17 -O3 -fopenmp -o influence_each_node influence_IC_each_node.cpp
 * Or simply use: make
 * 
 * Usage: ./influence_each_node <graph_file> <n_sim> <activation_probability> <output_file>
 * Example: ./influence_each_node sample_graph.txt 1000 0.1 results.txt
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
    double probability;
};

class Graph {
public:
    int n;
    vector<vector<Edge>> adj;
    
    Graph(int n) : n(n), adj(n) {}
    
    void addEdge(int u, int v, double prob) {
        adj[u].push_back({v, prob});
        adj[v].push_back({u, prob}); // For undirected graph
    }
};

// IC Model Simulation
int simulateIC(const Graph& graph, const vector<int>& seeds, mt19937& gen) {
    vector<bool> activated(graph.n, false);
    vector<bool> tried(graph.n, false);
    queue<int> active_queue;
    
    // Initialize with seed nodes
    for (int seed : seeds) {
        if (seed >= 0 && seed < graph.n) {
            activated[seed] = true;
            active_queue.push(seed);
        }
    }
    
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    while (!active_queue.empty()) {
        int current = active_queue.front();
        active_queue.pop();
        
        if (tried[current]) continue;
        tried[current] = true;
        
        // Try to activate neighbors
        for (const Edge& edge : graph.adj[current]) {
            int neighbor = edge.to;
            if (!activated[neighbor]) {
                double random_val = dist(gen);
                if (random_val < edge.probability) {
                    activated[neighbor] = true;
                    active_queue.push(neighbor);
                }
            }
        }
    }
    
    // Count activated nodes
    int count = 0;
    for (bool isActive : activated) {
        if (isActive) count++;
    }
    
    return count;
}

vector<int> parseSeeds(const string& seed_string) {
    vector<int> seeds;
    stringstream ss(seed_string);
    string token;
    
    while (getline(ss, token, ',')) {
        try {
            int seed = stoi(token);
            seeds.push_back(seed);
        } catch (const exception& e) {
            cerr << "Warning: Invalid seed value: " << token << endl;
        }
    }
    
    return seeds;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <n_sim> <activation_probability> <output_file>" << endl;
        cerr << "  graph_file_path: path to graph file with format 'node1 node2'" << endl;
        cerr << "  n_sim: number of simulation repetitions per node" << endl;
        cerr << "  activation_probability: uniform probability for all edges (0.0 to 1.0)" << endl;
        cerr << "  output_file: path to output file for results" << endl;
        return 1;
    }

    string file_path = argv[1];
    int n_sim = stoi(argv[2]);
    double activation_prob = stod(argv[3]);
    string output_file = argv[4];

    if (n_sim <= 0) {
        cerr << "Error: n_sim must be positive" << endl;
        return 1;
    }

    if (activation_prob < 0.0 || activation_prob > 1.0) {
        cerr << "Error: activation_probability must be between 0.0 and 1.0" << endl;
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
    cout << "Activation probability: " << activation_prob << endl;

    for (const auto& [u, v] : edges) {
        int mapped_u = vertex_map[u];
        int mapped_v = vertex_map[v];
        graph.addEdge(mapped_u, mapped_v, activation_prob);
    }

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
                int activated_count = simulateIC(graph, single_seed, gen);
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
