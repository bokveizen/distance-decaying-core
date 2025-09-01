#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>

// Check if OpenMP is available
#ifdef _OPENMP
#include <omp.h>
#define HAVE_OPENMP 1
#else
// Provide stubs for OpenMP functions
#define HAVE_OPENMP 0
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

// For memory tracking
#include <sys/resource.h>

using namespace std;

// Function to get current memory usage in MB
double getCurrentMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // Convert from KB to MB
    return usage.ru_maxrss / 1024.0;
}

// Function to print memory usage with a label
void printMemoryUsage(const string &label) {
    cout << label << ": " << fixed << setprecision(2) << getCurrentMemoryUsage() << " MB" << endl;
}

struct Graph {
    int n;
    vector<vector<int>> adj;
    
    Graph(int _n) : n(_n) {
        adj.resize(_n);
    }

    Graph() : n(0) {}

    void addEdge(int u, int v) {
        if (u >= 0 && u < n && v >= 0 && v < n) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
};

const int INF_INT = numeric_limits<int>::max();

// Function to compute distances from a single source node using BFS
vector<int> computeDistancesFromSource(const Graph &G, int source) {
    int n = G.n;
    vector<int> distances(n, INF_INT);
    queue<int> q;

    distances[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : G.adj[u]) {
            if (distances[v] == INF_INT) {
                distances[v] = distances[u] + 1;
                q.push(v);
            }
        }
    }

    return distances;
}

// Compute all-pairs shortest paths using BFS from each node
vector<vector<int>> computeAllPairsDistances(const Graph &G) {
    int n = G.n;
    vector<vector<int>> distances(n);
    
    const bool use_parallel = (HAVE_OPENMP && n > 100);

#if HAVE_OPENMP
#pragma omp parallel for shared(distances) if (use_parallel)
#endif
    for (int i = 0; i < n; ++i) {
        distances[i] = computeDistancesFromSource(G, i);
    }
    
    return distances;
}

// Parse landmark nodes from a file
vector<int> parseLandmarksFromFile(const string &landmark_file_path, const unordered_map<int, int> &vertex_map) {
    vector<int> landmarks;
    
    ifstream file(landmark_file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open landmark file " << landmark_file_path << endl;
        return landmarks;
    }
    
    string line;
    while (getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;
            
        // Remove whitespace
        line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            try {                
                int orig_id = stoi(line);
                int internal_id = vertex_map.at(orig_id);
                landmarks.push_back(internal_id);
            } catch (const std::exception& e) {
                cerr << "Warning: Invalid landmark node ID: " << line << endl;
            }
        }
    }
    
    return landmarks;
}

// Compute landmark-based bounds
pair<int, int> computeLandmarkBounds(int s, int t, const vector<int> &landmarks, 
                                   const vector<vector<int>> &distances) {
    int lower_bound = 0;
    int upper_bound = INF_INT;
    
    for (int landmark : landmarks) {
        int dist_s_landmark = distances[s][landmark];
        int dist_t_landmark = distances[t][landmark];
        
        // Skip if either distance is infinite (unreachable)
        if (dist_s_landmark == INF_INT || dist_t_landmark == INF_INT) {
            continue;
        }
        
        // Lower bound: max_{v in landmarks} |dist(s,v) - dist(t,v)|
        int triangle_lower = abs(dist_s_landmark - dist_t_landmark);
        lower_bound = max(lower_bound, triangle_lower);
        
        // Upper bound: min_{v in landmarks} dist(s,v) + dist(t,v)
        int triangle_upper = dist_s_landmark + dist_t_landmark;
        upper_bound = min(upper_bound, triangle_upper);
    }
    
    return {lower_bound, upper_bound};
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <landmark_file_path> <output_file_path>" << endl;
        cerr << "  landmark_file_path: path to a file containing landmark node IDs (one per line)" << endl;
        cerr << "  Output: aggregated statistics (average relative error and standard deviation)" << endl;
        return 1;
    }

    string file_path = argv[1];
    string landmark_file_path = argv[2];
    string output_path = argv[3];

    cout << "Landmark-based shortest path computation (aggregated version)" << endl;
    cout << "Landmarks: " << landmark_file_path << endl;

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

    for (const auto &[u, v] : edges) {
        int mapped_u = vertex_map[u];
        int mapped_v = vertex_map[v];
        G.addEdge(mapped_u, mapped_v);
    }

    // Parse landmarks
    vector<int> landmarks = parseLandmarksFromFile(landmark_file_path, vertex_map);
    if (landmarks.empty()) {
        cerr << "Error: No valid landmark nodes found in " << landmark_file_path << endl;
        return 1;
    }

    cout << "Found " << landmarks.size() << " landmark nodes" << endl;
    cout << "Landmark nodes (internal IDs): ";
    for (int i = 0; i < landmarks.size(); ++i) {
        if (i > 0) cout << ", ";
        cout << landmarks[i];
    }    
    cout << endl;

    printMemoryUsage("After graph construction");

    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    cout << "Computing all-pairs shortest paths..." << endl;
    
    // Compute all-pairs shortest paths
    vector<vector<int>> distances = computeAllPairsDistances(G);

    auto computation_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(computation_time - start_time);
    cout << "All-pairs computation time: " << fixed << setprecision(3) 
         << duration.count() / 1000.0 << " seconds" << endl;

    printMemoryUsage("After distance computation");

    cout << "Computing aggregated relative error statistics..." << endl;

    // Variables for aggregated statistics
    long long total_pairs = 0;
    long long valid_pairs = 0;
    double sum_relative_error = 0.0;
    double sum_squared_relative_error = 0.0;

    // Variables for thread-local statistics
    vector<long long> thread_total_pairs(omp_get_max_threads(), 0);
    vector<long long> thread_valid_pairs(omp_get_max_threads(), 0);
    vector<double> thread_sum_relative_error(omp_get_max_threads(), 0.0);
    vector<double> thread_sum_squared_relative_error(omp_get_max_threads(), 0.0);
    
    // Compute aggregated statistics for all pairs
#if HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 10)
#endif
    for (int i = 0; i < n; ++i) {
        int thread_id = omp_get_thread_num();
        
        for (int j = i + 1; j < n; ++j) {
            thread_total_pairs[thread_id]++;
            
            int actual_distance = distances[i][j];
            
            // Only process pairs that are reachable
            if (actual_distance != INF_INT && actual_distance > 0) {
                // Compute landmark-based bounds
                auto [lower_bound, upper_bound] = computeLandmarkBounds(i, j, landmarks, distances);
                
                // Only consider pairs with valid bounds
                if (upper_bound != INF_INT) {
                    thread_valid_pairs[thread_id]++;
                    
                    // Use average of lower and upper bounds as estimation
                    double estimation = (lower_bound + upper_bound) / 2.0;
                    
                    // Compute relative error: abs(estimation - actual) / actual
                    double relative_error = abs(estimation - actual_distance) / static_cast<double>(actual_distance);
                    
                    // Update running statistics
                    thread_sum_relative_error[thread_id] += relative_error;
                    thread_sum_squared_relative_error[thread_id] += relative_error * relative_error;
                }
            }
        }
        
        // Progress reporting (only from main thread)
        if (thread_id == 0 && ((i + 1) % 100 == 0 || i == n - 1)) {
            double progress = (double)(i + 1) / n * 100.0;
            cout << "Progress: " << fixed << setprecision(1) << progress << "% complete" << endl;
        }
    }
    
    // Aggregate results from all threads
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        total_pairs += thread_total_pairs[t];
        valid_pairs += thread_valid_pairs[t];
        sum_relative_error += thread_sum_relative_error[t];
        sum_squared_relative_error += thread_sum_squared_relative_error[t];
    }

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total computation time: " << fixed << setprecision(3) 
         << total_duration.count() / 1000.0 << " seconds" << endl;

    printMemoryUsage("After all computations");

    // Compute and output final statistics
    cout << "\n=== RESULTS ===" << endl;
    cout << "Total node pairs: " << total_pairs << endl;
    cout << "Valid pairs (reachable with bounds): " << valid_pairs << endl;
    cout << "Percentage of valid pairs: " << fixed << setprecision(2) 
         << (100.0 * valid_pairs / total_pairs) << "%" << endl;

    // Write results to output file
    ofstream output_file(output_path);
    if (!output_file.is_open()) {
        cerr << "Error: Could not open output file " << output_path << endl;
        return 1;
    }

    if (valid_pairs > 0) {
        double mean_relative_error = sum_relative_error / valid_pairs;
        double variance = (sum_squared_relative_error / valid_pairs) - (mean_relative_error * mean_relative_error);
        double std_dev_relative_error = sqrt(max(0.0, variance)); // Ensure non-negative due to numerical precision
        
        // Write only the two numbers to output file
        output_file << fixed << setprecision(6) << mean_relative_error << " " << std_dev_relative_error << endl;
        
        cout << "\nRelative Error Statistics:" << endl;
        cout << "  Average relative error: " << fixed << setprecision(6) << mean_relative_error << endl;
        cout << "  Standard deviation: " << fixed << setprecision(6) << std_dev_relative_error << endl;
        cout << "  Average relative error (%): " << fixed << setprecision(3) << (mean_relative_error * 100) << "%" << endl;
        cout << "  Standard deviation (%): " << fixed << setprecision(3) << (std_dev_relative_error * 100) << "%" << endl;
        cout << "Results written to " << output_path << endl;
    } else {
        // Write placeholder values for no valid pairs
        output_file << "-1 -1" << endl;
        cout << "No valid pairs found for analysis." << endl;
        cout << "Placeholder values (-1 -1) written to " << output_path << endl;
    }

    return 0;
}
