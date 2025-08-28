/*
 * BFS Subgraph Extraction
 * 
 * Extracts an induced subgraph from the first k nodes in a BFS traversal sequence.
 * This provides a way to sample connected subgraphs from larger networks.
 * 
 * Compile with: g++ -O3 -o bfs_subgraph bfs_subgraph.cpp -std=c++17
 * 
 * Usage: ./bfs_subgraph <original_graph_file> <bfs_sequence_file> <k> <output_file>
 */

#include <bits/stdc++.h>
#include <unordered_set>
#include <unordered_map>

using namespace std;

struct Graph {
    int n;
    unordered_map<int, unordered_set<int>> adj;
    unordered_set<int> vertices;
    
    Graph() : n(0) {}
    
    void addEdge(int u, int v) {
        adj[u].insert(v);
        adj[v].insert(u);
        vertices.insert(u);
        vertices.insert(v);
        n = vertices.size();
    }
    
    bool hasVertex(int v) const {
        return vertices.find(v) != vertices.end();
    }
    
    bool hasEdge(int u, int v) const {
        if (!hasVertex(u) || !hasVertex(v)) return false;
        auto it = adj.find(u);
        if (it == adj.end()) return false;
        return it->second.find(v) != it->second.end();
    }
    
    vector<int> getVertices() const {
        return vector<int>(vertices.begin(), vertices.end());
    }
    
    int getEdgeCount() const {
        int count = 0;
        for (const auto& [u, neighbors] : adj) {
            count += neighbors.size();
        }
        return count / 2; // Each edge is counted twice
    }
};

// Read the original graph from edge list file
Graph readGraph(const string& filename) {
    Graph G;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open graph file " << filename << endl;
        exit(1);
    }
    
    string line;
    int edge_count = 0;
    
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        
        G.addEdge(u, v);
        edge_count++;
    }
    
    file.close();
    
    cout << "Original graph loaded: " << G.n << " nodes, " << edge_count << " edges" << endl;
    return G;
}

// Read BFS sequence from file
vector<int> readBFSSequence(const string& filename) {
    vector<int> bfs_sequence;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open BFS sequence file " << filename << endl;
        exit(1);
    }
    
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        try {
            int node = stoi(line);
            bfs_sequence.push_back(node);
        } catch (const exception& e) {
            cerr << "Warning: Invalid node in BFS sequence: " << line << endl;
        }
    }
    
    file.close();
    
    cout << "BFS sequence loaded: " << bfs_sequence.size() << " nodes" << endl;
    return bfs_sequence;
}

// Extract induced subgraph from the first k nodes in BFS sequence
Graph extractInducedSubgraph(const Graph& original_graph, const vector<int>& bfs_sequence, int k) {
    Graph subgraph;
    
    if (k <= 0 || k > bfs_sequence.size()) {
        cerr << "Error: Invalid k value. Must be between 1 and " << bfs_sequence.size() << endl;
        exit(1);
    }
    
    // Get the first k nodes from BFS sequence
    unordered_set<int> subgraph_nodes;
    for (int i = 0; i < k; i++) {
        subgraph_nodes.insert(bfs_sequence[i]);
    }
    
    cout << "Extracting induced subgraph from first " << k << " BFS nodes..." << endl;
    
    // Add all edges between nodes in the subgraph
    int edges_added = 0;
    for (int u : subgraph_nodes) {
        if (!original_graph.hasVertex(u)) {
            cerr << "Warning: Node " << u << " from BFS sequence not found in original graph" << endl;
            continue;
        }
        
        // Check all neighbors of u in the original graph
        auto it = original_graph.adj.find(u);
        if (it != original_graph.adj.end()) {
            for (int v : it->second) {
                // If v is also in our subgraph nodes, add the edge
                if (subgraph_nodes.find(v) != subgraph_nodes.end() && u < v) {
                    // Only add each edge once (u < v ensures this)
                    subgraph.addEdge(u, v);
                    edges_added++;
                }
            }
        }
    }
    
    cout << "Induced subgraph created: " << subgraph.n << " nodes, " << edges_added << " edges" << endl;
    
    // Calculate subgraph statistics
    double node_ratio = (double)subgraph.n / original_graph.n * 100.0;
    double edge_ratio = (double)edges_added / original_graph.getEdgeCount() * 100.0;
    
    cout << "Subgraph represents " << fixed << setprecision(2) 
         << node_ratio << "% of nodes and " << edge_ratio << "% of edges" << endl;
    
    return subgraph;
}

// Write subgraph to file in edge list format
void writeSubgraph(const Graph& subgraph, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open output file " << filename << endl;
        exit(1);
    }
    
    // Write header comments
    // file << "# Induced subgraph extracted from BFS sequence" << endl;
    // file << "# Generated on: " << __DATE__ << " " << __TIME__ << endl;
    // file << "# Nodes: " << subgraph.n << ", Edges: " << subgraph.getEdgeCount() << endl;
    // file << "#" << endl;
    
    // Write edges
    vector<pair<int, int>> edges;
    for (const auto& [u, neighbors] : subgraph.adj) {
        for (int v : neighbors) {
            if (u < v) { // Avoid duplicate edges
                edges.push_back({u, v});
            }
        }
    }
    
    // Sort edges for consistent output
    sort(edges.begin(), edges.end());
    
    for (const auto& [u, v] : edges) {
        file << u << " " << v << endl;
    }
    
    file.close();
    cout << "Subgraph written to: " << filename << endl;
}

// Generate statistics about the subgraph
void generateStatistics(const Graph& original_graph, const Graph& subgraph, 
                       const vector<int>& bfs_sequence, int k, const string& output_file) {
    string stats_file = output_file + ".stats";
    ofstream file(stats_file);
    
    if (!file.is_open()) {
        cerr << "Warning: Could not create statistics file " << stats_file << endl;
        return;
    }
    
    // file << "BFS Subgraph Extraction Statistics" << endl;
    // file << "===================================" << endl;
    // file << "Generated on: " << __DATE__ << " " << __TIME__ << endl;
    // file << endl;
    
    file << "Parameters:" << endl;
    file << "  k (subgraph size): " << k << endl;
    file << "  BFS sequence length: " << bfs_sequence.size() << endl;
    file << "  Starting node: " << bfs_sequence[0] << endl;
    file << endl;
    
    file << "Original Graph:" << endl;
    file << "  Nodes: " << original_graph.n << endl;
    file << "  Edges: " << original_graph.getEdgeCount() << endl;
    file << endl;
    
    file << "Extracted Subgraph:" << endl;
    file << "  Nodes: " << subgraph.n << endl;
    file << "  Edges: " << subgraph.getEdgeCount() << endl;
    file << "  Node coverage: " << fixed << setprecision(2) 
         << (double)subgraph.n / original_graph.n * 100.0 << "%" << endl;
    file << "  Edge coverage: " << fixed << setprecision(2) 
         << (double)subgraph.getEdgeCount() / original_graph.getEdgeCount() * 100.0 << "%" << endl;
    
    // Calculate density
    if (subgraph.n > 1) {
        double max_edges = (double)subgraph.n * (subgraph.n - 1) / 2.0;
        double density = (double)subgraph.getEdgeCount() / max_edges;
        file << "  Density: " << fixed << setprecision(4) << density << endl;
    }
    
    file.close();
    cout << "Statistics written to: " << stats_file << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <original_graph_file> <bfs_sequence_file> <k> <output_file>" << endl;
        cerr << "  original_graph_file: path to the original graph (edge list format)" << endl;
        cerr << "  bfs_sequence_file: path to the BFS traversal sequence" << endl;
        cerr << "  k: number of nodes to include in subgraph (first k from BFS)" << endl;
        cerr << "  output_file: path for the output subgraph" << endl;
        cerr << endl;
        cerr << "Example: " << argv[0] << " graph.txt bfs_sequence.txt 100 subgraph.txt" << endl;
        return 1;
    }
    
    string graph_file = argv[1];
    string bfs_file = argv[2];
    int k = stoi(argv[3]);
    string output_file = argv[4];
    
    cout << "BFS Subgraph Extraction" << endl;
    cout << "======================" << endl;
    cout << "Graph file: " << graph_file << endl;
    cout << "BFS sequence file: " << bfs_file << endl;
    cout << "Subgraph size (k): " << k << endl;
    cout << "Output file: " << output_file << endl;
    cout << endl;
    
    // Load original graph
    Graph original_graph = readGraph(graph_file);
    
    // Load BFS sequence
    vector<int> bfs_sequence = readBFSSequence(bfs_file);
    
    // Validate k
    if (k > bfs_sequence.size()) {
        cerr << "Error: k (" << k << ") is larger than BFS sequence length (" 
             << bfs_sequence.size() << ")" << endl;
        return 1;
    }
    
    // Extract induced subgraph
    Graph subgraph = extractInducedSubgraph(original_graph, bfs_sequence, k);
    
    // Write subgraph to file
    writeSubgraph(subgraph, output_file);
    
    // Generate statistics
    generateStatistics(original_graph, subgraph, bfs_sequence, k, output_file);
    
    cout << endl;
    cout << "Subgraph extraction completed successfully!" << endl;
    
    return 0;
} 