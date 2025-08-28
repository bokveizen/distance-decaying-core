from pathlib import Path
from collections import defaultdict
import random

datasets = [
    "web-polblogs",
    "web-google",
    "soc-wiki-Vote",
    "web-edu",
    "web-BerkStan",
    "web-webbase-2001",
    "web-spam",
    "web-indochina-2004",
    "ca-CondMat",
    "soc-epinions",
]

p_metric = Path(f"../metric")
metric_list = [f"khcore_{d}" for d in range(1, 7)]
metric_list += ["poly_1", "poly_5", "poly_10"]
metric_list += ["exp_2", "exp_5", "exp_10"]

k_list = [5, 10, 20, 50, 100]

def get_max_k(centrality_dict, k_list, seed):
    # find the top k nodes by centrality
    # breaking ties by random shuffling
    centrality_list = list(centrality_dict.items())
    random.seed(seed)
    random.shuffle(centrality_list)  # shuffle first to randomize ties
    centrality_list.sort(key=lambda x: x[1], reverse=True)
    max_k_nodes_list = []
    for k in k_list:
        max_k_nodes = [v for v, _ in centrality_list[:k]]
        max_k_nodes_list.append(max_k_nodes)
    return max_k_nodes_list

p_landmark = Path("landmark")

for ds in datasets:
    p_landmark_ds = p_landmark / ds
    p_landmark_ds.mkdir(parents=True, exist_ok=True)    
    for metric in metric_list:
        for seed in range(10):
            f_metric = p_metric / ds / f"{metric}.txt"
            node2metric = defaultdict(float)
            with open(f_metric, "r") as f:
                lines = f.readlines()
                for line in lines:
                    node, value = line.strip().split()[:2]
                    node = int(node)
                    value = float(value)
                    node2metric[node] = value                                    
            max_k_nodes_list = get_max_k(node2metric, k_list, seed)
            for i, k in enumerate(k_list):
                max_k_nodes = max_k_nodes_list[i]
                p_res = p_landmark_ds / f"{k}" / metric
                p_res.mkdir(parents=True, exist_ok=True)
                with open(p_res / f"{seed}.txt", "w") as f:
                    for node in max_k_nodes:
                        f.write(f"{node}\n")