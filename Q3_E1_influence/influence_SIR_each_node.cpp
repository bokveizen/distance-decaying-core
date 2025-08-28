/*
 * SIR (Susceptible-Infected-Recovered) Model Simulation with OpenMP Parallelization
 * This version tests each node as a single seed and outputs influence to file
 * 
 * Compilation: g++ -std=c++17 -O3 -fopenmp -o influence_SIR_each_node influence_SIR_each_node.cpp
 * Or simply use: make
 * 
 * Usage: ./influence_SIR_each_node <graph_file> <n_sim> <infection_probability> <recovery_probability> <output_file>
 * Example: ./influence_SIR_each_node sample_graph.txt 1000 0.1 0.3 results.txt
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <unordered_map>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

enum NodeState {
    SUSCEPTIBLE = 0,
    INFECTED = 1,
    RECOVERED = 2
};

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

// SIR Model Simulation
int simulateSIR(const Graph& graph, const vector<int>& seeds, double recovery_prob, mt19937& gen) {
    vector<NodeState> states(graph.n, SUSCEPTIBLE);
    vector<int> infected_nodes;
    
    // Initialize with seed nodes
    for (int seed : seeds) {
        if (seed >= 0 && seed < graph.n) {
            states[seed] = INFECTED;
            infected_nodes.push_back(seed);
        }
    }
    
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Continue simulation until no infected nodes remain
    while (!infected_nodes.empty()) {
        vector<int> new_infections;
        vector<int> remaining_infected;
        
        // Process each infected node
        for (int infected : infected_nodes) {
            // Try to infect susceptible neighbors
            for (const Edge& edge : graph.adj[infected]) {
                int neighbor = edge.to;
                if (states[neighbor] == SUSCEPTIBLE) {
                    double random_val = dist(gen);
                    if (random_val < edge.probability) {
                        states[neighbor] = INFECTED;
                        new_infections.push_back(neighbor);
                    }
                }
            }
            
            // Check if this infected node recovers
            double recovery_random = dist(gen);
            if (recovery_random < recovery_prob) {
                states[infected] = RECOVERED;
            } else {
                remaining_infected.push_back(infected);
            }
        }
        
        // Update infected nodes list for next time step
        infected_nodes = remaining_infected;
        infected_nodes.insert(infected_nodes.end(), new_infections.begin(), new_infections.end());
    }
    
    // Count recovered nodes (total nodes that were ever infected)
    int recovered_count = 0;
    for (NodeState state : states) {
        if (state == RECOVERED) {
            recovered_count++;
        }
    }
    
    return recovered_count;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <n_sim> <infection_probability> <recovery_probability> <output_file>" << endl;
        cerr << "  graph_file_path: path to graph file with format 'node1 node2'" << endl;
        cerr << "  n_sim: number of simulation repetitions per node" << endl;
        cerr << "  infection_probability: probability of infection along edges (0.0 to 1.0)" << endl;
        cerr << "  recovery_probability: probability of recovery per time step (0.0 to 1.0)" << endl;
        cerr << "  output_file: path to output file for results" << endl;
        return 1;
    }

    string file_path = argv[1];
    int n_sim = stoi(argv[2]);
    double infection_prob = stod(argv[3]);
    double recovery_prob = stod(argv[4]);
    string output_file = argv[5];

    if (n_sim <= 0) {
        cerr << "Error: n_sim must be positive" << endl;
        return 1;
    }

    if (infection_prob < 0.0 || infection_prob > 1.0) {
        cerr << "Error: infection_probability must be between 0.0 and 1.0" << endl;
        return 1;
    }

    if (recovery_prob < 0.0 || recovery_prob > 1.0) {
        cerr << "Error: recovery_probability must be between 0.0 and 1.0" << endl;
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
    cout << "Infection probability: " << infection_prob << ", Recovery probability: " << recovery_prob << endl;

    for (const auto& [u, v] : edges) {
        int mapped_u = vertex_map[u];
        int mapped_v = vertex_map[v];
        graph.addEdge(mapped_u, mapped_v, infection_prob);
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
                int recovered_count = simulateSIR(graph, single_seed, recovery_prob, gen);
                results[i] = recovered_count;
                sum += recovered_count;
                sum_squares += recovered_count * recovered_count;
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