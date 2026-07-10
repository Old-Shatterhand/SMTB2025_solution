from pathlib import Path
import pickle

from typing_extensions import Literal

import matplotlib

from src.viz.constants import CLASS_METRIC, DATASET2TASK, DATASETS, LAYERS, MODELS, REG_METRIC, TASK_METRICS
from src.viz.fig0 import plot_fig0
from src.viz.fig1 import plot_fig1, plot_full_fig1_left, plot_full_fig1_right
from src.viz.fig2 import plot_fig2, plot_full_fig2
from src.viz.fig3_old import plot_fig3_old
from src.viz.fig3 import plot_fig3, plot_full_fig3_left, plot_full_fig3_right  #, plot_full_fig3
from src.viz.fig4 import plot_fig4
from src.viz.fig5 import plot_fig5
from src.viz.utils import compute_performance, compute_scope_performance, read_metric, read_pca_metric, read_scope_metric


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def collect_metric(metric: Literal["zero", "pc@95", "var@10", "5dvol", "id", "no", "knn"]):
    if (p := Path(f"data_{metric}_{CLASS_METRIC}_{REG_METRIC}.pkl")).exists():
        print(f"Found {metric} data in {p.absolute()}")
        return
    
    data = {}
    for model in MODELS:
        data[model] = {}
        for dataset in DATASETS:
            print(metric, model, dataset)
            try:
                if dataset == "scope_40_208":
                    if metric == "knn":
                        data[model][dataset + "_fold"] = [compute_scope_performance(BASE, model, dataset, layer, algo=metric, metric=TASK_METRICS[DATASET2TASK[dataset]], level="fold", min_x=10) for layer in range(LAYERS[model] + 1)]
                        data[model][dataset + "_superfamily"] = [compute_scope_performance(BASE, model, dataset, layer, algo=metric, metric=TASK_METRICS[DATASET2TASK[dataset]], level="superfamily", min_x=10) for layer in range(LAYERS[model] + 1)]
                    else:
                        data[model][dataset + "_fold"] = [read_scope_metric(BASE, model, layer, metric, f"{metric}_fold_min10.csv") for layer in range(LAYERS[model] + 1)]
                        data[model][dataset + "_superfamily"] = [read_scope_metric(BASE, model, layer, metric, f"{metric}_superfamily_min10.csv") for layer in range(LAYERS[model] + 1)]
                else:
                    if metric in {"knn", "lr"}:
                        data[model][dataset] = [compute_performance(BASE, model, dataset, layer, algo=metric, metric=TASK_METRICS[DATASET2TASK[dataset]], aa=False, n_classes=0, task=DATASET2TASK[dataset]) for layer in range(LAYERS[model] + 1)]
                    elif metric in {"zero", "pc@95", "var@10", "5dvol"}:
                        data[model][dataset] = [read_pca_metric(BASE, model, dataset, layer, metric=metric, aa=False) for layer in range(LAYERS[model] + 1)]
                    else:
                        data[model][dataset] = [read_metric(BASE, model, dataset, layer, metric=metric, aa=False) for layer in range(LAYERS[model] + 1)]
            except Exception as e:
                print("\r", model, dataset, "Error:", e)
                data[model][dataset] = 0
        try:
            data[model]["scope_40_208_3ssp"] = [compute_performance(BASE, model, "scope_40_208", layer, algo="knn", metric=CLASS_METRIC, aa=True, n_classes=3, task="multi-class") for layer in range(LAYERS[model] + 1)]
        except Exception as e:
            print("\r", model, "scope_40_208_3ssp", "Error:", e)
            data[model]["scope_40_208_3ssp"] = 0
        try:
            data[model]["scope_40_208_8ssp"] = [compute_performance(BASE, model, "scope_40_208", layer, algo="knn", metric=CLASS_METRIC, aa=True, n_classes=8, task="multi-class") for layer in range(LAYERS[model] + 1)]
        except Exception as e:
            print("\r", model, "scope_40_208_8ssp", "Error:", e)
            data[model]["scope_40_208_8ssp"] = 0
        try:
            data[model]["binding"] = [compute_performance(BASE, model, "binding", layer, algo="knn", metric=CLASS_METRIC, aa=True, n_classes=2, task="binary") for layer in range(LAYERS[model] + 1)]
        except Exception as e:
            print("\r", model, "binding", "Error:", e)
            data[model]["binding"] = 0
    
    with open(p, "wb") as f:
        pickle.dump(data, f)
        print(f"Saved data to {p.absolute()}")
    return


if __name__ == "__main__":
    matplotlib.rc('font', **{'size': 11})
    models = ["esm_t33", "esm_t36", "esmc_600m", "ankh_large", "prott5", "prostt5", "progen2_medium", "progen2_large", "protgpt2"]

    # collect_metric("knn")

    plot_fig0()

    # plot_fig1(models)
    # plot_full_fig1_left()
    # plot_full_fig1_right()

    # plot_fig2(models)
    # plot_full_fig2()

    # plot_fig3(models)
    # plot_full_fig3_left()
    # plot_full_fig3_right()

    # plot_fig4()
    # plot_fig5()

    # plot_fig3_old(models)
