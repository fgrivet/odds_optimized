import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
import pickle
from math import comb

from scripts.declare_methods import METHOD_PARAMS


if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams["svg.fonttype"] = 'none'

    p_list = [1, 2, 3, 4, 5, 6, 7, 8]
    sdp_list = [comb(p+6, p)*comb(p+6, p) for p in p_list]
    datasets = [np.random.uniform(-1, 1, p * 1000).reshape(1000, p) for p in p_list]
    duration_list = []
    for i, data in enumerate(datasets):
        if os.path.exists(f"temp/time_evolution/{i}"):
            with open(f"temp/time_evolution/{i}/temp.pkl", "rb") as f:
                duration_list.append(pickle.load(f))
        else:
            print(f"Dimension {p_list[i]}")
            train = data[:500, :]
            test = data[500:, :]
            model = METHOD_PARAMS["experiment"]["DyCF"].copy()
            model.fit(train)
            start = time.time()
            scores = model.eval_update(test)
            end = time.time()
            os.makedirs(f"temp/time_evolution/{i}")
            with open(f"temp/time_evolution/{i}/temp.pkl", "wb") as f:
                pickle.dump(end - start, f)
            duration_list.append(end - start)

    df = pd.DataFrame()
    df["Dimension"] = p_list
    df["Duration"] = duration_list
    df["Matrix Size"] = sdp_list

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    sns.lineplot(x="Dimension", y="Duration", data=df, ax=ax1, marker="o", color=sns.color_palette()[0])
    sns.lineplot(x="Dimension", y="Matrix Size", data=df, ax=ax2, marker="o", color=sns.color_palette()[1])
    ax1.set_ylabel('Duration (seconds)', color=sns.color_palette()[0])
    ax2.set_ylabel(r'Matrix size $s_d(p)$x$s_d(p)$', color=sns.color_palette()[1])
    ax1.set_yticks(np.linspace(0, ax1.get_ybound()[1] + 1, 6))
    ax2.set_yticks(np.linspace(0, ax2.get_ybound()[1] + 1, 6))
    plt.savefig("Fig17.eps", bbox_inches='tight')
    plt.close()
