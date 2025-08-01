from odds_optimized.statistics import DyCF, KDE
from odds_optimized.utils import load_dataset, roc_auc_score, average_precision_score
from odds_optimized.plotter import LevelsetPlotter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_context("paper")

    cols = ["AUROC", "AP"]
    table = []

    data = load_dataset("../data/synthetics/two_disks/two_disks.csv")
    labels = data[:, -1]
    data = data[:, :-1]

    # Christoffel Function
    cf = DyCF(d=6, regularization="none")
    start = time.time()
    cf.fit(data)
    print(f"CF fit: {time.time() - start}s")
    cf_lp = LevelsetPlotter(cf)

    # Multivariate KDE
    kde = KDE(threshold=0.1, win_size=data.shape[0])
    start = time.time()
    kde.fit(data)
    print(f"KDE fit: {time.time() - start}s")
    mkde_lp = LevelsetPlotter(kde)

    # AUROC & AP
    start = time.time()
    cf_scores = cf.decision_function(data)
    cf_auroc = roc_auc_score(labels, cf_scores)
    cf_ap = average_precision_score(labels, cf_scores)
    print(f"CF score & metrics: {time.time() - start}s")
    res = [cf_auroc, cf_ap]
    table.append(['%1.9f' % v for v in res])
    start = time.time()
    mkde_scores = kde.decision_function(data)
    mkde_auroc = roc_auc_score(labels, mkde_scores)
    mkde_ap = average_precision_score(labels, mkde_scores)
    print(f"KDE score & metrics: {time.time() - start}s")
    res = [mkde_auroc, mkde_ap]
    table.append(['%1.9f' % v for v in res])

    start = time.time()
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    cf_lp.plot_in_ax(data, ax, percentiles=[10, 20, 30, 50, 75], colors=["gold", "green", "aqua", "fuchsia", "indigo"], linewidths=1.8)
    cf_lp.plot_in_ax(data, ax, levels=[cf.d ** (3 * cf.p / 2)], colors="red", linewidths=3.2)
    print(f"CF plot: {time.time() - start}s")
    plt.savefig("Fig1a.eps", bbox_inches='tight', format="eps")
    plt.close()

    start = time.time()
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    mkde_lp.plot_in_ax(data, ax, percentiles=[10, 20, 30, 50, 75], colors=["gold", "green", "aqua", "fuchsia", "indigo"], linewidths=1.8)
    print(f"KDE plot: {time.time() - start}s")
    plt.savefig("Fig1b.eps", bbox_inches='tight', format="eps")
    plt.close()

    """ Save AUROC & AP results """
    df = pd.DataFrame(data=table, columns=cols, index=["CF", "MKDE"])
    df.to_csv(f"cf_vs_kde.csv")
