import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

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

# 10 p values for each model
model2p_list = {
    "IC": ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.10"],
    "LT": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
    "SIR": ["0.01", "0.03", "0.05", "0.08", "0.12", "0.17", "0.25", "0.35", "0.50", "0.75"]
}

p_metric = Path(f"../metric")
metric_list = [f"khcore_{d}" for d in range(1, 7)]
metric_list += ["poly_1", "poly_5", "poly_10"]
metric_list += ["exp_2", "exp_5", "exp_10"]

for model in ["IC", "LT", "SIR"]:
    print(f"model: {model}")
    p_list = model2p_list[model]
    p_simu_res = Path(f"{model}_simu")    

    for ds in datasets:
        metric2rank_cor_list = defaultdict(list)
        for p in p_list:
            node2influence = defaultdict(float)
            # read the ground truth influences
            f_simu_res = p_simu_res / ds / f"{p}.txt"
            with open(f_simu_res, "r") as f:
                lines = f.readlines()[1:]  # skip the header
                for line in lines:
                    node, mean_influence = line.strip().split(",")[:2]
                    node = int(node)
                    mean_influence = float(mean_influence)
                    node2influence[node] = mean_influence
            
            all_nodes = list(node2influence.keys())
            all_influence = [node2influence[node] for node in all_nodes]
            
            for metric in metric_list:
                f_metric = p_metric / ds / f"{metric}.txt"
                node2metric = defaultdict(float)
                with open(f_metric, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        node, value = line.strip().split()[:2]
                        node = int(node)
                        value = float(value)
                        node2metric[node] = value
            
                all_metric = [node2metric[node] for node in all_nodes]
                
                cor, _ = spearmanr(all_influence, all_metric)
                metric2rank_cor_list[metric].append(cor)
        
        metric2rank_cor_list = {k: np.mean(v) for k, v in metric2rank_cor_list.items()}
        print_info = [ds] + [f"{metric2rank_cor_list[k]}" for k in metric_list]
        print(" ".join(print_info))
    print("-" * 100)
    print()