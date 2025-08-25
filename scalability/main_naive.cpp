#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <vector>

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

const int TH_OPENMP_SLOW = 20;
const int TH_OPENMP_FAST = 20;

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
    vector<unordered_set<int>> adj;
    Graph(int _n) : n(_n) {
        // Explicitly resize the adjacency list vector
        adj.resize(_n);
    }

    // Default constructor
    Graph() : n(0) {}

    void addEdge(int u, int v) {
        // Safety check to prevent out of bounds access
        if (u >= 0 && u < n && v >= 0 && v < n) {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
};

// make the graph global
Graph G;
vector<double> centrality;
unordered_set<int> remaining;
unordered_set<int> updated;
const int INF_INT = numeric_limits<int>::max(); // Prevent overflow in distance calculations
int MAX_CACHE_SIZE = 100;

unordered_map<int, double> precomputed_discounts;
vector<double> discount_cache; // Vector for O(1) lookup when max distance is known
string discount_type = "poly";
double discount_param = 1.0;
int dist_max = INF_INT;

const int MAX_GF_H = 10;
vector<int> cnt_gfs;

void init_cache() {
    // use cache size of min(dist_max, MAX_CACHE_SIZE)
    int cache_size = min(dist_max, MAX_CACHE_SIZE);

    discount_cache.resize(cache_size);

    for (int i = 1; i < cache_size; ++i) {
        if (discount_type == "poly") {
            discount_cache[i] = 1.0 / pow(i, discount_param); // poly(i) = 1/i^discount_param
        } else if (discount_type == "exp") {
            discount_cache[i] = 1.0 / pow(discount_param, i - 1); // exp(i) = 1/discount_param^(i-1)
        } else {
            cerr << "Error: Invalid discount type: " << discount_type << endl;
            assert(false);
        }
    }
}

double get_discount(int dist_uv) {
    if (dist_uv < 0) {
        cerr << "Error: Invalid distance value: " << dist_uv << endl;
        assert(false);
    }

    if (dist_uv == 0) {
        return 1.0; // No discount for distance 0 (self)
    }

    if (dist_uv == INF_INT) {
        return 0;
    }

    // Try vector cache first if available and in range
    if (dist_uv < discount_cache.size()) {
        return discount_cache[dist_uv];
    }

    // Fallback to hash map
    auto it = precomputed_discounts.find(dist_uv);
    if (it != precomputed_discounts.end()) {
        return it->second;
    }

    // Otherwise, compute and store
    double discount;
    if (discount_type == "poly") {
        discount = 1.0 / pow(dist_uv, discount_param); // poly(i) = 1/i^discount_param
    } else if (discount_type == "exp") {
        discount = 1.0 / pow(discount_param, dist_uv - 1); // exp(i) = 1/discount_param^(i-1)
    } else {
        cerr << "Error: Invalid discount type: " << discount_type << endl;
        assert(false);
    }

    // Store in hash map for future use
    precomputed_discounts[dist_uv] = discount;

    return discount;
}

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

        for (const auto v : G.adj[u]) {
            if (distances[v] == INF_INT) {
                distances[v] = distances[u] + 1;
                q.push(v);
            }
        }
    }

    return distances;
}

// Function to perform bounded BFS from a single source node
vector<int> computeBoundedNodesFromSource(const Graph &G, int source, int radius) {
    vector<int> nodes_within_radius;
    queue<pair<int, int>> q; // pair of (node, distance)
    unordered_set<int> visited;

    q.push(make_pair(source, 0));
    visited.insert(source);
    nodes_within_radius.push_back(source);

    while (!q.empty()) {
        int u = q.front().first;
        int dist = q.front().second;
        q.pop();

        if (dist < radius) {
            for (const auto v : G.adj[u]) {
                if (visited.find(v) == visited.end()) {
                    visited.insert(v);
                    nodes_within_radius.push_back(v);
                    q.push(make_pair(v, dist + 1));
                }
            }
        }
    }

    return nodes_within_radius;
}

double computeCentralityFromDistances(const vector<int> &distances) {
    double centrality = 0.0;
    for (int u = 0; u < distances.size(); ++u) {
        if (distances[u] != INF_INT && distances[u] > 0) {
            centrality += get_discount(distances[u]);
        }
    }
    return centrality;
}
// Variant function that returns both distances and nodes at each level using standard BFS
pair<vector<int>, vector<vector<int>>> computeDistancesLevelsFromSource(const Graph &G,
                                                                        int source) {
    int n = G.n;
    vector<int> distances(n, INF_INT);
    vector<vector<int>> levels;
    queue<int> q;

    distances[source] = 0;
    q.push(source);

    // Track the maximum distance encountered to size the levels vector
    int max_distance = 0;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        int current_distance = distances[u];

        // Ensure levels vector has enough capacity
        if (current_distance >= levels.size()) {
            levels.resize(current_distance + 1);
        }

        // Add current node to its level
        levels[current_distance].push_back(u);
        max_distance = max(max_distance, current_distance);

        for (const auto v : G.adj[u]) {
            if (distances[v] == INF_INT) {
                distances[v] = distances[u] + 1;
                q.push(v);
            }
        }
    }

    return make_pair(distances, levels);
}

// Compute centrality for a single node
double computeSingleCentrality(int v, const Graph &graph = G) {
    double local_centrality = 0.0;

    // Compute distances from v to all other nodes
    vector<int> distances = computeDistancesFromSource(graph, v);

    for (int u = 0; u < graph.n; ++u) {
        if (u == v)
            continue;
        int dist_uv = distances[u];
        if (dist_uv != INF_INT && dist_uv > 0) {
            local_centrality += get_discount(dist_uv);
        }
    }
    return local_centrality;
}

// Compute centrality values based on distance discount - parallelized with reduction
vector<double> computeAllCentrality() {
    int n = G.n;
    vector<double> centrality(n, 0.0);

    // Only use parallelism for larger graphs
    const bool use_parallel = (HAVE_OPENMP && n >= TH_OPENMP_SLOW);

// Parallelize the outer loop for computing centrality with dynamic scheduling
#if HAVE_OPENMP
#pragma omp parallel for schedule(dynamic, 32) shared(G) if (use_parallel)
#endif
    for (int v = 0; v < n; ++v) {
        centrality[v] = computeSingleCentrality(v);
    }
    return centrality;
}

// Update graph metrics after node removal - parallelized updates with size threshold
// v_rem: node to remove
// k_global: hop threshold for global filters
// k_local: hop threshold for local filters
// multiple_nodes_to_remove: whether multiple nodes are being removed in this round
// remaining_after_removal: the remaining nodes after this round
// G_after_removal: the graph after this round

// Hidden inputs from the global variables:
// G: the current graph
// centrality: the centrality values of all nodes
// remaining: the remaining nodes
// updated: the updated nodes

void nodeRemoval(int v_rem, int k_global, int k_local, bool multiple_nodes_to_remove,
                 Graph &G_after_removal) {
    auto [distances, levels] = computeDistancesLevelsFromSource(G, v_rem);        

    unordered_set<int> remaining_cur;
    remaining_cur.reserve(remaining.size());
    // Only add nodes that are not in the "updated" set
    for (int u : remaining) {
        if (updated.find(u) == updated.end()) {
            remaining_cur.insert(u);
        }
    }
    vector<int> all_remaining_vec(remaining.begin(), remaining.end());
    
    centrality[v_rem] = 0;
    int k_max = max(k_global, k_local);
    if (levels.size() < k_max + 1) {
        levels.resize(k_max + 1); // fill with empty vectors
    }

    // Construct the graph G_new after removing v_rem
    auto G_new = G;
    for (int neighbor : levels[1]) {
        G_new.adj[neighbor].erase(v_rem);
    }
    G_new.adj[v_rem].clear();

    // Global filters
    // Special case h = 1: Check whether all layer-1 nodes are pairwise adjacent (distance = 1)
    bool gf_1 = true;
    for (int i = 0; i < levels[1].size() && gf_1; ++i) {
        int u = levels[1][i];
        for (int j = i + 1; j < levels[1].size(); ++j) {
            int v = levels[1][j];
            if (G.adj[u].count(v) == 0) {
                gf_1 = false;
                break;
            }
        }
        if (!gf_1)
            break;
    }
    if (gf_1) {
        bool use_parallel = (HAVE_OPENMP && all_remaining_vec.size() >= TH_OPENMP_FAST);
        vector<int> remaining_cur_vec(remaining_cur.begin(), remaining_cur.end());

        #if HAVE_OPENMP
        #pragma omp parallel for shared(centrality, distances)
        #endif
        for (int u : remaining_cur_vec) {
            if (distances[u] != INF_INT && distances[u] > 0) {
                double discount = get_discount(distances[u]);
                centrality[u] -= discount;
            }
        }
        G = std::move(G_new);
        cnt_gfs[1]++;
        return;
    }

    // For h >= 2, we just use bounded BFS
    for (int h = 2; h <= k_global; ++h) {
        vector<int> &layer_h_nodes = levels[h];
        // Bounded BFS for layer_h_nodes in parallel
        bool use_parallel = (HAVE_OPENMP && layer_h_nodes.size() >= TH_OPENMP_SLOW);
        int radius = 2 * h - 1;
        bool gf_h = true;
        #if HAVE_OPENMP
        #pragma omp parallel for shared(layer_h_nodes, gf_h) if (use_parallel)
        #endif
        for (int i = 0; i < layer_h_nodes.size(); ++i) {
            // Early exit if another thread already found gf_h to be false
            if (!gf_h)
                continue;

            int u = layer_h_nodes[i];
            vector<int> nodes_in_h = computeBoundedNodesFromSource(G, u, radius);
            unordered_set<int> nodes_in_h_set(nodes_in_h.begin(), nodes_in_h.end());
            // gf_h = false if nodes_in_h does not contain all nodes in layer_h_nodes_set
            for (int v : layer_h_nodes) {
                if (nodes_in_h_set.find(v) == nodes_in_h_set.end()) {
                    gf_h = false;
                    break;
                }
            }
        }
        if (gf_h) {
            // When global filter is hit for h, we collect the nodes from the first (h - 1) layers
            vector<int> local_nodes;
            for (int i = 1; i < h; ++i) {
                local_nodes.insert(local_nodes.end(), levels[i].begin(), levels[i].end());
            }
            vector<vector<int>> nbr_dist_before(local_nodes.size(), vector<int>(G.n, INF_INT));
            vector<vector<int>> nbr_dist_after(local_nodes.size(), vector<int>(G.n, INF_INT));
            bool use_parallel = (HAVE_OPENMP && local_nodes.size() >= TH_OPENMP_SLOW);
            #if HAVE_OPENMP
            #pragma omp parallel for shared(local_nodes) if (use_parallel)
            #endif
            for (int i = 0; i < local_nodes.size(); ++i) {
                int nbr_i = local_nodes[i];
                nbr_dist_before[i] = computeDistancesFromSource(G, nbr_i);
                nbr_dist_after[i] = computeDistancesFromSource(G_new, nbr_i);
                // Skip if nbr_i is in the updated set
                if (updated.find(nbr_i) != updated.end())
                    continue;
                // Update centrality for local node nbr_i
                double centrality_i = 0;
                for (int u : all_remaining_vec) {
                    if (u == nbr_i)
                        continue;
                    if (nbr_dist_after[i][u] != INF_INT && nbr_dist_after[i][u] > 0) {
                        auto new_discount = get_discount(nbr_dist_after[i][u]);
                        centrality_i += new_discount;
                    }
                }
                centrality[nbr_i] = centrality_i;
            }

            // Remove the local nodes from remaining_cur
            for (int u : local_nodes) {
                remaining_cur.erase(u);
            }

            vector<int> remaining_cur_vec(remaining_cur.begin(), remaining_cur.end());
            // Update the centrality for the nodes in remaining_cur
            use_parallel = (HAVE_OPENMP && remaining_cur_vec.size() >= TH_OPENMP_FAST);
            #if HAVE_OPENMP
            #pragma omp parallel for shared(remaining_cur_vec) if (use_parallel)
            #endif
            for (int u : remaining_cur_vec) {
                // Loss from v_rem itself
                if (distances[u] != INF_INT && distances[u] > 0) {
                    double discount = get_discount(distances[u]);
                    centrality[u] -= discount;
                }
                // Loss from the local nodes
                for (int i = 0; i < local_nodes.size(); ++i) {
                    if (nbr_dist_before[i][u] == INF_INT)
                        continue;
                    auto old_discount = get_discount(nbr_dist_before[i][u]);
                    auto new_discount =
                        nbr_dist_after[i][u] == INF_INT ? 0 : get_discount(nbr_dist_after[i][u]);
                    centrality[u] += (new_discount - old_discount);
                }
            }
            G = std::move(G_new);
            cnt_gfs[h]++;
            return;
        }
    }

    // When it reaches here, the global filter is not hit for any considered h
    // So we now consider local filters

    // First collect local nodes up to k_local hops
    vector<int> local_nodes;
    vector<pair<int, int>> hop_ranges(k_local + 1, {-1, -1}); // (start_idx, end_idx) for each hop
    for (int i = 1; i <= k_local; ++i) {
        int start_idx = local_nodes.size();
        local_nodes.insert(local_nodes.end(), levels[i].begin(), levels[i].end());
        int end_idx = local_nodes.size() - 1;
        hop_ranges[i] = {start_idx, end_idx};
    }
    vector<vector<int>> nbr_dist_before(local_nodes.size(), vector<int>(G.n, INF_INT));
    vector<vector<int>> nbr_dist_after(local_nodes.size(), vector<int>(G.n, INF_INT));
    bool use_parallel = (HAVE_OPENMP && local_nodes.size() >= TH_OPENMP_SLOW);
    #if HAVE_OPENMP
    #pragma omp parallel for shared(local_nodes) if (use_parallel)
    #endif
    for (int i = 0; i < local_nodes.size(); ++i) {
        int nbr_i = local_nodes[i];
        nbr_dist_before[i] = computeDistancesFromSource(G, nbr_i);
        nbr_dist_after[i] = computeDistancesFromSource(G_new, nbr_i);
        // Skip if nbr_i is in the updated set
        if (updated.find(nbr_i) != updated.end())
            continue;
        // Update centrality for local node nbr_i
        double centrality_i = 0;
        for (int u : all_remaining_vec) {
            if (u == nbr_i)
                continue;
            if (nbr_dist_after[i][u] != INF_INT && nbr_dist_after[i][u] > 0) {
                auto new_discount = get_discount(nbr_dist_after[i][u]);
                centrality_i += new_discount;
            }
        }
        centrality[nbr_i] = centrality_i;
    }

    // Remove the local nodes from remaining_cur
    for (int u : local_nodes) {
        remaining_cur.erase(u);
    }
    vector<int> remaining_cur_vec(remaining_cur.begin(), remaining_cur.end());

    // For each node in remaining_cur_vec, first check whether it is
    // (1) fast node, i.e., only local update is required; we also record the number of hops we need
    // to consider for it (2) slow node, i.e., we need to do full BFS for it
    vector<vector<int>> fast_nodes(
        k_local); // e.g., fast_nodes[0] contains nodes that only lose centrality from v_rem itself
    vector<int> slow_nodes;
    slow_nodes.reserve(remaining_cur_vec.size());

    for (int u : remaining_cur_vec) {
        if (distances[u] == INF_INT)
            continue; // u and v_rem are not connected

        bool lf_any = false;
        // Check the local filters for each hop
        for (int h = 1; h <= k_local; ++h) {
            bool lf_h = true;
            // If all h-hop neighbors do not change distance, then u is a fast node w.r.t. h - 1
            auto [start_idx, end_idx] = hop_ranges[h];
            for (int i = start_idx; i <= end_idx; ++i) {
                int nbr_i = local_nodes[i];
                int dist_before_i = nbr_dist_before[i][u];
                int dist_after_i = nbr_dist_after[i][u];
                if (dist_before_i != dist_after_i) {
                    lf_h = false;
                    break;
                }
            }

            if (lf_h) {
                fast_nodes[h - 1].push_back(u);
                lf_any = true;
                break;
            }
        }
        if (!lf_any) {
            slow_nodes.push_back(u);
        }
    }

    // Deal with fast nodes
    for (int h = 0; h < k_local; ++h) {
        auto &fast_nodes_h = fast_nodes[h];
        // Skip if fast_nodes_h is empty
        if (fast_nodes_h.empty())
            continue;

        bool use_parallel = (HAVE_OPENMP && fast_nodes_h.size() >= TH_OPENMP_FAST);
        #if HAVE_OPENMP
        #pragma omp parallel for shared(fast_nodes_h) if (use_parallel)
        #endif
        for (int u : fast_nodes_h) {
            // Loss from v_rem itself
            if (distances[u] != INF_INT && distances[u] > 0) {
                double discount = get_discount(distances[u]);
                centrality[u] -= discount;
            }
            // Loss from the local nodes up to h hops
            if (h > 0) {
                auto end_idx = hop_ranges[h].second;
                for (int i = 0; i <= end_idx; ++i) {
                    if (nbr_dist_before[i][u] == INF_INT)
                        continue;
                    auto old_discount = get_discount(nbr_dist_before[i][u]);
                    auto new_discount = get_discount(nbr_dist_after[i][u]);
                    centrality[u] += (new_discount - old_discount);
                }
            }
        }
    }

    // Deal with slow nodes
    use_parallel = (HAVE_OPENMP && slow_nodes.size() >= TH_OPENMP_SLOW);
    #if HAVE_OPENMP
    #pragma omp parallel for shared(slow_nodes) if (use_parallel)
    #endif
    for (int u : slow_nodes) {
        if (multiple_nodes_to_remove) {
            centrality[u] = computeSingleCentrality(u, G_after_removal);
        } else {
            centrality[u] = computeSingleCentrality(u, G_new);
        }
    }

    if (multiple_nodes_to_remove) {
        // add slow nodes into updated
        updated.insert(slow_nodes.begin(), slow_nodes.end());
    }

    G = std::move(G_new);
}

// Optimized batch node removal function
void batchRemoveNodes(const vector<int>& nodes_to_remove, Graph& target_graph) {
    // Pre-compute all affected neighbors to minimize hash table operations
    unordered_set<int> nodes_to_remove_set(nodes_to_remove.begin(), nodes_to_remove.end());
    unordered_map<int, vector<int>> neighbors_to_update;
    
    // Collect all neighbors that need updates
    for (int node : nodes_to_remove) {
        for (int neighbor : target_graph.adj[node]) {
            if (nodes_to_remove_set.find(neighbor) == nodes_to_remove_set.end()) {
                neighbors_to_update[neighbor].push_back(node);
            }
        }
    }
    
    // Batch remove nodes from their neighbors' adjacency lists
    const bool use_parallel = (HAVE_OPENMP && neighbors_to_update.size() >= TH_OPENMP_FAST);
    vector<pair<int, vector<int>>> neighbors_vec(neighbors_to_update.begin(), neighbors_to_update.end());
    
    #if HAVE_OPENMP
    #pragma omp parallel for shared(neighbors_vec, target_graph) if (use_parallel)
    #endif
    for (auto& [neighbor, nodes_to_erase] : neighbors_vec) {
        for (int node : nodes_to_erase) {
            target_graph.adj[neighbor].erase(node);
        }
    }
    
    // Clear adjacency lists for removed nodes
    #if HAVE_OPENMP
    #pragma omp parallel for shared(nodes_to_remove, target_graph) if (use_parallel)
    #endif
    for (int node : nodes_to_remove) {
        target_graph.adj[node].clear();
    }
}

// Memory-optimized batch node removal with capacity pre-allocation
void batchRemoveNodesOptimized(const vector<int>& nodes_to_remove, Graph& target_graph) {
    if (nodes_to_remove.empty()) return;
    
    // Pre-compute all affected neighbors to minimize hash table operations
    unordered_set<int> nodes_to_remove_set;
    nodes_to_remove_set.reserve(nodes_to_remove.size());
    nodes_to_remove_set.insert(nodes_to_remove.begin(), nodes_to_remove.end());
    
    unordered_map<int, vector<int>> neighbors_to_update;
    neighbors_to_update.reserve(nodes_to_remove.size() * 10); // Rough estimate
    
    // Collect all neighbors that need updates
    for (int node : nodes_to_remove) {
        const auto& adj_list = target_graph.adj[node];
        for (int neighbor : adj_list) {
            if (nodes_to_remove_set.find(neighbor) == nodes_to_remove_set.end()) {
                neighbors_to_update[neighbor].push_back(node);
            }
        }
    }
    
    // Batch remove nodes from their neighbors' adjacency lists
    const bool use_parallel = (HAVE_OPENMP && neighbors_to_update.size() >= TH_OPENMP_FAST);
    
    if (use_parallel) {
        vector<pair<int, vector<int>>> neighbors_vec;
        neighbors_vec.reserve(neighbors_to_update.size());
        neighbors_vec.assign(neighbors_to_update.begin(), neighbors_to_update.end());
        
        #if HAVE_OPENMP
        #pragma omp parallel for shared(neighbors_vec, target_graph)
        #endif
        for (size_t i = 0; i < neighbors_vec.size(); ++i) {
            auto& [neighbor, nodes_to_erase] = neighbors_vec[i];
            for (int node : nodes_to_erase) {
                target_graph.adj[neighbor].erase(node);
            }
        }
    } else {
        for (auto& [neighbor, nodes_to_erase] : neighbors_to_update) {
            for (int node : nodes_to_erase) {
                target_graph.adj[neighbor].erase(node);
            }
        }
    }
    
    // Clear adjacency lists for removed nodes in parallel
    const bool use_parallel_clear = (HAVE_OPENMP && nodes_to_remove.size() >= TH_OPENMP_FAST);
    #if HAVE_OPENMP
    #pragma omp parallel for shared(nodes_to_remove, target_graph) if (use_parallel_clear)
    #endif
    for (size_t i = 0; i < nodes_to_remove.size(); ++i) {
        target_graph.adj[nodes_to_remove[i]].clear();
    }
}

// Optimized graph creation after node removal without full copy
Graph createGraphAfterRemoval(const vector<int>& nodes_to_remove) {
    Graph G_after_removal = G;  // Still need a copy, but we'll optimize the removal
    batchRemoveNodesOptimized(nodes_to_remove, G_after_removal);
    return G_after_removal;
}

// Advanced optimization: Lazy graph with marked deleted nodes
struct LazyGraph {
    const Graph* original_graph;
    unordered_set<int> deleted_nodes;
    
    LazyGraph(const Graph* g) : original_graph(g) {}
    LazyGraph(const Graph* g, const vector<int>& to_delete) : original_graph(g) {
        deleted_nodes.insert(to_delete.begin(), to_delete.end());
    }
    
    bool isDeleted(int node) const {
        return deleted_nodes.find(node) != deleted_nodes.end();
    }
    
    // Get valid neighbors (excluding deleted nodes)
    vector<int> getValidNeighbors(int node) const {
        vector<int> valid_neighbors;
        if (isDeleted(node)) return valid_neighbors;
        
        for (int neighbor : original_graph->adj[node]) {
            if (!isDeleted(neighbor)) {
                valid_neighbors.push_back(neighbor);
            }
        }
        return valid_neighbors;
    }
};

// Modified BFS that works with LazyGraph
vector<int> computeDistancesFromSourceLazy(const LazyGraph& lazy_graph, int source) {
    int n = lazy_graph.original_graph->n;
    vector<int> distances(n, INF_INT);
    
    if (lazy_graph.isDeleted(source)) {
        return distances;
    }
    
    queue<int> q;
    distances[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        vector<int> valid_neighbors = lazy_graph.getValidNeighbors(u);
        for (int v : valid_neighbors) {
            if (distances[v] == INF_INT) {
                distances[v] = distances[u] + 1;
                q.push(v);
            }
        }
    }
    
    return distances;
}

// Optimized single centrality computation with lazy graph
double computeSingleCentralityLazy(int v, const LazyGraph& lazy_graph) {
    if (lazy_graph.isDeleted(v)) return 0.0;
    
    double local_centrality = 0.0;
    vector<int> distances = computeDistancesFromSourceLazy(lazy_graph, v);
    
    for (int u = 0; u < lazy_graph.original_graph->n; ++u) {
        if (u == v || lazy_graph.isDeleted(u)) continue;
        
        int dist_uv = distances[u];
        if (dist_uv != INF_INT && dist_uv > 0) {
            local_centrality += get_discount(dist_uv);
        }
    }
    return local_centrality;
}

// Ultra-fast graph creation using lazy deletion (no copying needed!)
LazyGraph createLazyGraphAfterRemoval(const vector<int>& nodes_to_remove) {
    return LazyGraph(&G, nodes_to_remove);
}

int main(int argc, char *argv[]) {
    // Default values
    string file_path = "";
    string output_path = "";
    bool verbose = false;
    discount_type = "poly";
    discount_param = 1.0;
    dist_max = INF_INT; // Always set to infinity
    int k_global = 2;
    int k_local = 2;
    double from_scratch_threshold = 0.01;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                file_path = argv[++i];
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-d" || arg == "--discount-type") {
            if (i + 1 < argc) {
                discount_type = argv[++i];
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-p" || arg == "--discount-param") {
            if (i + 1 < argc) {
                discount_param = stod(argv[++i]);
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-kg" || arg == "--k-global") {
            if (i + 1 < argc) {
                k_global = stoi(argv[++i]);
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-kl" || arg == "--k-local") {
            if (i + 1 < argc) {
                k_local = stoi(argv[++i]);
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-t" || arg == "--threshold") {
            if (i + 1 < argc) {
                from_scratch_threshold = stod(argv[++i]);
            } else {
                cerr << "Error: " << arg << " requires a value" << endl;
                return 1;
            }
        }
        else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " -i <graph_file> -o <output_file> [options]" << endl;
            cout << endl;
            cout << "Required arguments:" << endl;
            cout << "  -i, --input <file>           Input graph file path" << endl;
            cout << "  -o, --output <file>          Output file path" << endl;
            cout << endl;
            cout << "Optional arguments:" << endl;
            cout << "  -d, --discount-type <type>   Discount function: 'poly' or 'exp' (default: poly)" << endl;
            cout << "  -p, --discount-param <val>   Discount parameter value (default: 1.0)" << endl;
            cout << "  -kg, --k-global <val>        Global filter hop threshold, 0=disabled (default: 2)" << endl;
            cout << "  -kl, --k-local <val>         Local filter hop threshold, 0=disabled (default: 2)" << endl;
            cout << "  -t, --threshold <val>        From-scratch computation threshold ratio (default: 0.01)" << endl;
            cout << "  -v, --verbose                Print verbose output" << endl;
            cout << "  -h, --help                   Show this help message" << endl;
            cout << endl;
            cout << "Examples:" << endl;
            cout << "  " << argv[0] << " -i graph.txt -o output.txt -v" << endl;
            cout << "  " << argv[0] << " -i graph.txt -o output.txt -d exp -p 2.0 -kg 3 -kl 2" << endl;
            return 0;
        }
        else if (file_path.empty()) {
            // First positional argument (for backward compatibility)
            file_path = arg;
        }
        else if (output_path.empty()) {
            // Second positional argument (for backward compatibility)
            output_path = arg;
        }
        else {
            cerr << "Error: Unknown argument '" << arg << "'" << endl;
            cerr << "Use -h or --help for usage information" << endl;
            return 1;
        }
    }
    
    // Check required arguments
    if (file_path.empty() || output_path.empty()) {
        cerr << "Error: Both input and output file paths are required" << endl;
        cerr << "Usage: " << argv[0] << " -i <graph_file> -o <output_file> [options]" << endl;
        cerr << "Use -h or --help for detailed usage information" << endl;
        return 1;
    }

    if (discount_type != "poly" && discount_type != "exp") {
        cerr << "Error: discount_type must be 'poly' or 'exp'" << endl;
        return 1;
    }

    if (discount_param <= 0 || k_global < 0 || k_local < 0 || from_scratch_threshold <= 0 || from_scratch_threshold > 1) {
        cerr << "Error: discount_param must be positive, k_global and k_local must be non-negative, "
             << "from_scratch_threshold must be between 0 and 1" << endl;
        return 1;
    }

    cout << "discount_type: " << discount_type << ", discount_param: " << discount_param << ", k_global: " << k_global << ", k_local: " << k_local << ", from_scratch_threshold: " << from_scratch_threshold << endl;

    // Initialize cache
    init_cache();
    cnt_gfs.resize(k_global + 1, 0);

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
    G = Graph(n);

    cout << "Number of nodes: " << n << ", number of edges: " << edges.size() << endl;

    for (const auto &[u, v] : edges) {
        int mapped_u = vertex_map[u];
        int mapped_v = vertex_map[v];
        G.addEdge(mapped_u, mapped_v);
    }

    for (int i = 0; i < n; ++i)
        remaining.insert(i);

    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    // Compute initial centrality values for all nodes
    centrality = computeAllCentrality();

    // Print the time used for initial centrality computation
    auto initial_centrality_time = chrono::high_resolution_clock::now();
    auto initial_centrality_duration = chrono::duration_cast<chrono::microseconds>(initial_centrality_time - start_time);
    cout << "Time used for initial centrality computation: " << fixed << setprecision(3) << initial_centrality_duration.count() / 1000.0 << " milliseconds" << endl;

    cout << "Initial centrality computed" << endl;

    vector<double> coreness(n, 0.0);

    double cur_coreness = 0.0;
    int total_nodes = n;
    bool early_stopping = false;

    // Find the minimum centrality value
    int min_node = *remaining.begin();
    double min_centrality = centrality[min_node];
    for (int node : remaining) {
        if (centrality[node] < min_centrality) {
            min_centrality = centrality[node];
        }
    }
    cur_coreness = min_centrality;

    while (!remaining.empty()) {
        // Collect nodes to remove
        vector<int> nodes_to_remove;
        for (int node : remaining) {
            if (centrality[node] <= cur_coreness) {
                nodes_to_remove.push_back(node);
            }
        }
        while (!nodes_to_remove.empty()) {
            // Remove nodes from remaining set first
            for (int node : nodes_to_remove) {
                coreness[node] = cur_coreness;
                remaining.erase(node);
            }
            // Use optimized batch removal
            batchRemoveNodesOptimized(nodes_to_remove, G);
            // Recompute centrality for the remaining nodes
            vector<int> remaining_vec(remaining.begin(), remaining.end());
            const bool use_parallel = (HAVE_OPENMP && remaining_vec.size() >= TH_OPENMP_SLOW);
            #if HAVE_OPENMP
            #pragma omp parallel for shared(centrality) if (use_parallel)
            #endif
            for (int node : remaining_vec) {
                centrality[node] = computeSingleCentrality(node, G);
            }            

            if (verbose) {
                // Print the current progress
                auto current_time = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::microseconds>(current_time - start_time);
                cout << "Processing coreness: " << cur_coreness << ", remaining nodes: " << remaining.size()
                    << " (" << fixed << setprecision(1) << (100.0 * remaining.size() / n) << "%)"
                    << ", elapsed time: " << fixed << setprecision(3) << elapsed.count() / 1000.0
                    << " milliseconds" << endl;
            }

            // Re-collect nodes to remove
            nodes_to_remove.clear();
            for (int node : remaining) {
                if (centrality[node] <= cur_coreness) {
                    nodes_to_remove.push_back(node);
                }
            }
        }

        if (verbose) {
            // Calculate and display elapsed time at the beginning of each iteration
            auto current_time = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::microseconds>(current_time - start_time);

            // Print the current coreness, remaining nodes, and elapsed time
            cout << "Finishing coreness: " << cur_coreness << ", remaining nodes: " << remaining.size()
                << " (" << fixed << setprecision(1) << (100.0 * remaining.size() / n) << "%)"
                << ", elapsed time: " << fixed << setprecision(3) << elapsed.count() / 1000.0
                << " milliseconds" << endl;
        }

        if (remaining.empty()) {
            break;
        }
        // Find the next coreness level
        min_node = *remaining.begin();
        min_centrality = centrality[min_node];
        for (int node : remaining) {
            if (centrality[node] < min_centrality) {
                min_centrality = centrality[node];
            }
        }
        cur_coreness = min_centrality;
    }

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

    printMemoryUsage("After all iterations");
    cout << "Running time: " << fixed << setprecision(3) << duration.count() / 1000.0 << " milliseconds"
         << endl;

    // Output results
    ofstream output_file(output_path);
    if (!output_file.is_open()) {
        cerr << "Error: Could not open output file " << output_path << endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        output_file << reverse_map[i] << " " << coreness[i] << endl;
    }

    cout << "Input graph: " << n << " nodes, " << edges.size() << " edges" << endl;

    // Print the times of global filter hit
    for (int i = 1; i <= k_global; ++i) {
        cout << "Global filter hit at " << i << "-hop: " << cnt_gfs[i] << endl;
    }

    return 0;
}
