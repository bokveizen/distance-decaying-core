import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Dict
import random
from itertools import product

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_small_world_planted_subgraph(
    n_total: int = 1000,
    planted_size: int = 50,
    k_planted: int = 10,
    p_rewire_planted: float = 0.3,
    background_prob: float = 0.01,
    inter_planted_prob: float = 0.01
) -> Tuple[nx.Graph, List[int]]:
    """
    Generate a graph with a planted small-world subgraph.
    
    Parameters:
    -----------
    n_total : int
        Total number of nodes in the graph
    planted_size : int
        Size of the planted small-world subgraph
    k_planted : int
        Each node in planted subgraph connects to k nearest neighbors initially
    p_rewire_planted : float
        Probability of rewiring edges in the small-world subgraph
    background_prob : float
        Probability of edges in the background (non-planted) part
    inter_planted_prob : float
        Probability of edges between planted and non-planted nodes
    
    Returns:
    --------
    G : nx.Graph
        The generated graph
    planted_nodes : List[int]
        List of node indices that belong to the planted subgraph
    """
    
    # Select random nodes for the planted subgraph
    planted_nodes = random.sample(range(n_total), planted_size)
    non_planted_nodes = [i for i in range(n_total) if i not in planted_nodes]
    
    # Create the main graph
    G = nx.Graph()
    G.add_nodes_from(range(n_total))
    
    # Generate small-world structure for planted subgraph
    # Create a small-world graph and map it to our planted nodes
    planted_subgraph = nx.watts_strogatz_graph(planted_size, k_planted, p_rewire_planted)
    
    # Add edges from the planted subgraph to the main graph
    for edge in planted_subgraph.edges():
        u, v = planted_nodes[edge[0]], planted_nodes[edge[1]]
        G.add_edge(u, v)
    
    # Add background edges (sparse random graph for non-planted nodes)
    for i in range(len(non_planted_nodes)):
        for j in range(i + 1, len(non_planted_nodes)):
            if random.random() < background_prob:
                G.add_edge(non_planted_nodes[i], non_planted_nodes[j])
    
    # Add edges between planted and non-planted nodes (sparse)
    for planted_node in planted_nodes:
        for non_planted_node in non_planted_nodes:
            if random.random() < inter_planted_prob:
                G.add_edge(planted_node, non_planted_node)
    
    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['is_planted'] = node in planted_nodes
    
    return G, planted_nodes


def generate_sbm_planted_communities(
    n_total: int = 1000,
    n_communities: int = 1,
    community_sizes: Optional[List[int]] = None,
    p_within: float = 0.3,
    p_between: float = 0.01,
    background_prob: float = 0.005,    
) -> Tuple[nx.Graph, Dict[int, List[int]]]:
    """
    Generate a graph with planted SBM-like communities.
    
    Parameters:
    -----------
    n_total : int
        Total number of nodes in the graph
    n_communities : int
        Number of planted communities
    community_sizes : List[int], optional
        Sizes of each community. If None, communities are roughly equal
    p_within : float
        Base probability of edges within communities
    p_between : float
        Base probability of edges between different communities
    background_prob : float
        Probability of edges among non-community nodes    
    
    Returns:
    --------
    G : nx.Graph
        The generated graph
    communities : Dict[int, List[int]]
        Dictionary mapping community ID to list of node indices
    """
    
    # Determine community sizes
    if community_sizes is None:
        base_size = n_total // (n_communities + 2)  # Leave some nodes outside communities
        community_sizes = [base_size] * n_communities
    
    total_community_nodes = sum(community_sizes)
    if total_community_nodes > n_total:
        raise ValueError("Sum of community sizes exceeds total number of nodes")
    
    # Assign nodes to communities
    communities = {}
    available_nodes = list(range(n_total))
    random.shuffle(available_nodes)
    
    node_idx = 0
    for i in range(n_communities):
        community_nodes = available_nodes[node_idx:node_idx + community_sizes[i]]
        communities[i] = community_nodes
        node_idx += community_sizes[i]
    
    # Remaining nodes are not in any community
    non_community_nodes = available_nodes[node_idx:]
    
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(n_total))
    
    # Add edges within each community
    for comm_id, nodes in communities.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < p_within:
                    G.add_edge(nodes[i], nodes[j])
    
    # Add edges between different communities
    all_community_nodes = []
    for nodes in communities.values():
        all_community_nodes.extend(nodes)
    
    for comm_id1, nodes1 in communities.items():
        for comm_id2, nodes2 in communities.items():
            if comm_id1 < comm_id2:  # Avoid double counting
                for node1 in nodes1:
                    for node2 in nodes2:
                        if random.random() < p_between:
                            G.add_edge(node1, node2)
    
    # Add edges between community nodes and non-community nodes
    for comm_nodes in communities.values():
        for comm_node in comm_nodes:
            for non_comm_node in non_community_nodes:
                if random.random() < p_between:
                    G.add_edge(comm_node, non_comm_node)
    
    # Add background edges among non-community nodes
    for i in range(len(non_community_nodes)):
        for j in range(i + 1, len(non_community_nodes)):
            if random.random() < background_prob:
                G.add_edge(non_community_nodes[i], non_community_nodes[j])
    
    # Add node attributes
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = comm_id
    
    for node in G.nodes():
        G.nodes[node]['community'] = node_to_community.get(node, -1)  # -1 for non-community nodes
        G.nodes[node]['is_in_community'] = node in node_to_community
    
    return G, communities


from pathlib import Path

p_sw = Path("generated_graphs/small_world")
p_sw.mkdir(parents=True, exist_ok=True)

n_total = 200
planted_size_list = [15, 30, 45, 60]
p_planted = 0.2
p_rewire_planted = 0.2
p_background_list = [0.01, 0.03, 0.05]
n_generate = 10

for planted_size, p_background in product(planted_size_list, p_background_list):        
    background_prob = inter_planted_prob = p_background
    k_planted = int(planted_size * p_planted)

    config = f"{n_total}_{planted_size}_{k_planted}_{p_rewire_planted}_{p_background}"    
    p_sw_config = p_sw / config
    p_sw_config.mkdir(parents=True, exist_ok=True)    

    for i in range(n_generate):
        G_sw, planted_sw = generate_small_world_planted_subgraph(
            n_total=n_total,
            planted_size=planted_size,
            k_planted=k_planted,
            p_rewire_planted=p_rewire_planted,
            background_prob=background_prob,
            inter_planted_prob=inter_planted_prob,
        )
        
        # Save the graph
        nx.write_edgelist(G_sw, p_sw_config / f"graph_{i}.txt", data=False)
        # Save the planted nodes
        with open(p_sw_config / f"planted_nodes_{i}.txt", "w") as f:
            for node in planted_sw:
                f.write(f"{node}\n")

p_er = Path("generated_graphs/er")
p_er.mkdir(parents=True, exist_ok=True)

n_total = 200
planted_size_list = [15, 30, 45, 60]
p_planted = 0.2
p_background_list = [0.01, 0.03, 0.05]
n_generate = 10

for planted_size, p_background in product(planted_size_list, p_background_list):
    p_within = p_planted
    p_between = background_prob = p_background    

    config = f"{n_total}_{planted_size}_{p_within}_{p_between}_{p_background}"
    p_er_config = p_er / config
    p_er_config.mkdir(parents=True, exist_ok=True)

    for i in range(n_generate):
        G_er, planted_communities = generate_sbm_planted_communities(
            n_total=n_total,
            n_communities=1,
            community_sizes=[planted_size],
            p_within=p_within,
            p_between=p_between,
            background_prob=background_prob,
        )
        
        # Save the graph
        nx.write_edgelist(G_er, p_er_config / f"graph_{i}.txt", data=False)
        # Save the planted nodes
        with open(p_er_config / f"planted_nodes_{i}.txt", "w") as f:
            for node in planted_communities[0]:
                f.write(f"{node}\n")