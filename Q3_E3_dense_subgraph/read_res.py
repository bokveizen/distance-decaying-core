from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import product

n_total = 200
planted_size_list = [15, 30, 45, 60]
p_planted = 0.2
p_rewire_planted = 0.2
p_background_list = [0.01, 0.03, 0.05]

er_config_list = []
for planted_size, p_background in product(planted_size_list, p_background_list):
    config = f"{n_total}_{planted_size}_{p_planted}_{p_background}_{p_background}"
    er_config_list.append(config)

small_world_config_list = []
for planted_size, p_background in product(planted_size_list, p_background_list):
    k_planted = int(planted_size * p_planted)    
    config = f"{n_total}_{planted_size}_{k_planted}_{p_planted}_{p_background}"
    small_world_config_list.append(config)

n_generate = 10

metric_list = [f"khcore_{d}" for d in range(1, 7)]
metric_list += ["poly_1", "poly_5", "poly_10"]
metric_list += ["exp_2", "exp_5", "exp_10"]

# ER
for config in er_config_list:
    config2auroc_list = defaultdict(list)
    for i_graph in range(n_generate):
        p_graph = f"generated_graphs/er/{config}/graph_{i_graph}.txt"
        p_planted_nodes = f"generated_graphs/er/{config}/planted_nodes_{i_graph}.txt"
        
        total_nodes = set()
        with open(p_graph, "r") as f:
            for line in f:
                u, v = line.strip().split()
                total_nodes.add(int(u))
                total_nodes.add(int(v))
        
        planted_nodes = set()
        with open(p_planted_nodes, "r") as f:
            for line in f:
                planted_nodes.add(int(line.strip()))
                
        total_nodes = sorted(list(total_nodes))
        y_gt = [bool(node in planted_nodes) for node in total_nodes]
        
        model2pred = dict()
        for metric in metric_list:
            centrality_dict = defaultdict(float)
            p_metric = f"metric/er/{config}/{i_graph}/{metric}.txt"
            with open(p_metric, "r") as f:
                for line in f:
                    node, centrality = line.strip().split()
                    centrality_dict[int(node)] = float(centrality)
            model2pred[metric] = [centrality_dict[node] for node in total_nodes]
            auroc = roc_auc_score(y_gt, model2pred[metric])
            config2auroc_list[config].append(auroc)
    
        n_total, planted_size, p_planted, p_background, p_background = config.split("_")    
        metric2auroc_avg = {k: np.mean(v) for k, v in config2auroc_list.items()}
        print_info = [str(planted_size), str(p_background)] + [f"{metric2auroc_avg[k]}" for k in metric_list]
        print(f" ".join(print_info))
