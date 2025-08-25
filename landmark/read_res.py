import numpy as np
from pathlib import Path
from collections import defaultdict

dataset_list = [
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

metric_list = [f"khcore_{d}" for d in range(1, 7)]
metric_list += ["poly_1", "poly_5", "poly_10"]
metric_list += ["exp_2", "exp_5", "exp_10"]

k_list = [5, 10, 20, 50, 100]

p_error = Path("landmark_errors")

for ds in dataset_list:
    metric2error_list = defaultdict(list)
    for k in k_list:
        for metric in metric_list:
            p_error_k = p_error / ds / str(k) / metric
            for i in range(10):
                p_error_k_i = p_error_k / f"{i}.txt"
                with open(p_error_k_i, "r") as f:
                    line = f.readline()
                error_avg, error_std = line.strip().split()
                metric2error_list[metric].append(float(error_avg))
    metric2error_list = {k: np.mean(v) for k, v in metric2error_list.items()}
    print_info = [ds] + [f"{metric2error_list[k]}" for k in metric_list]
    print(" ".join(print_info))

print("-" * 100)
print()

for k in k_list:
    print(f"k = {k}")
    for ds in dataset_list:
        metric2error_list = defaultdict(list)
        for metric in metric_list:
            p_error_k = p_error / ds / str(k) / metric
            for i in range(10):
                p_error_k_i = p_error_k / f"{i}.txt"
                with open(p_error_k_i, "r") as f:
                    line = f.readline()
                error_avg, error_std = line.strip().split()
                metric2error_list[metric].append(float(error_avg))
        metric2error_list = {k: np.mean(v) for k, v in metric2error_list.items()}
        print_info = [ds] + [f"{metric2error_list[k]}" for k in metric_list]
        print(" ".join(print_info))
    print("-" * 100)
    print()
