from pathlib import Path
import pickle

from typing_extensions import Literal

import matplotlib

from src.viz.constants import CLASS_METRIC, DATASET2TASK, LAYERS, MODELS, REG_METRIC, WP_DATASETS
from src.viz.new_fig1 import plot_heatmap, plot_new_fig1
from src.viz.new_fig2 import plot_new_fig2
from src.viz.new_fig3 import plot_new_fig3
from src.viz.new_fig4 import plot_new_fig4
# from src.viz.fig1 import plot_fig1, plot_fig1_smtb, plot_full_fig1_perfs
from src.viz.fig2 import plot_fig2, plot_full_fig2_top, plot_full_fig2_bot, plot_fig2_grouped
from src.viz.fig3 import plot_fig3, plot_full_fig3_left, plot_full_fig3_right
from src.viz.fig4 import plot_fig4, plot_fig4_smtb
from src.viz.fig5 import plot_fig5_space
from src.viz.fig6 import plot_fig6, plot_fig6_smtb
from src.viz.fig7 import plot_fig7
from src.viz.utils import compute_performance, compute_scope_performance, read_metric, read_pca_metric, read_scope_metric


BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"


def collect_metric(algo: Literal["zero", "pc@95", "var@10", "5dvol", "ids", "noverlap", "knn", "lr"], class_metric: str, reg_metric: str, force: bool = False):
    if (p := Path(f"data_{algo}_{class_metric}_{reg_metric}.pkl")).exists() and not force:
        print(f"Found {algo} data in {p.absolute()}")
        return

    task_metrics = {
        "regression": reg_metric,
        "binary": class_metric,
        "multi-label": class_metric,
        "multi-class": class_metric,
    }
    
    data = {}
    for model in MODELS:
        data[model] = {}
        for dataset in WP_DATASETS + ["scope_40_208_3ssp", "scope_40_208_8ssp", "binding"]:
            if dataset in {"scope_40_208_3ssp", "scope_40_208_8ssp", "binding"} and algo in {"knn", "lr"}:
                continue
            print(algo, model, dataset)
            try:
                # if "esm_t6" in data and "scope_40_208_3ssp" in data["esm_t6"]:
                #     print(data["esm_t6"]["scope_40_208_3ssp"])
                if dataset == "scope_40_208":
                    if algo in {"knn", "lr"}:
                        data[model][dataset + "_fold"] = [compute_scope_performance(BASE, model, dataset, layer, algo=algo, metric=task_metrics[DATASET2TASK[dataset]], level="fold", min_x=10) for layer in range(LAYERS[model] + 1)]
                        data[model][dataset + "_superfamily"] = [compute_scope_performance(BASE, model, dataset, layer, algo=algo, metric=task_metrics[DATASET2TASK[dataset]], level="superfamily", min_x=10) for layer in range(LAYERS[model] + 1)]
                    elif algo in {"zero", "pc@95", "var@10", "5dvol"}:
                        data[model][dataset + "_fold"] = [read_pca_metric(BASE, model, "scope_40_208", layer, algo, "pca_fold_min10.pkl") for layer in range(LAYERS[model] + 1)]
                        data[model][dataset + "_superfamily"] = [read_pca_metric(BASE, model, "scope_40_208", layer, algo, "pca_superfamily_min10.pkl") for layer in range(LAYERS[model] + 1)]
                    else:
                        data[model][dataset + "_fold"] = [read_scope_metric(BASE, model, layer, algo, f"{algo}_fold_min10.csv") for layer in range(LAYERS[model] + 1)]
                        data[model][dataset + "_superfamily"] = [read_scope_metric(BASE, model, layer, algo, f"{algo}_superfamily_min10.csv") for layer in range(LAYERS[model] + 1)]
                else:
                    if algo in {"knn", "lr"}:
                        data[model][dataset] = [compute_performance(BASE, model, dataset, layer, algo=algo, metric=task_metrics[DATASET2TASK[dataset]], aa=dataset not in WP_DATASETS, n_classes=0, task=DATASET2TASK[dataset]) for layer in range(LAYERS[model] + 1)]
                    elif algo in {"zero", "pc@95", "var@10", "5dvol"}:
                        if dataset.startswith("scope_40_208"):
                            data[model][dataset] = [read_pca_metric(BASE, model, "scope_40_208", layer, metric=algo, aa=True) for layer in range(LAYERS[model] + 1)]
                        else:
                            data[model][dataset] = [read_pca_metric(BASE, model, dataset, layer, metric=algo, aa=dataset not in WP_DATASETS) for layer in range(LAYERS[model] + 1)]
                    else:
                        if dataset.startswith("scope_40_208"):
                            data[model][dataset] = [read_metric(BASE, model, "scope_40_208", layer, metric=algo, aa=True) for layer in range(LAYERS[model] + 1)]
                        else:
                            data[model][dataset] = [read_metric(BASE, model, dataset, layer, metric=algo, aa=dataset not in WP_DATASETS) for layer in range(LAYERS[model] + 1)]
            except Exception as e:
                print("\r", model, dataset, "Error:", e)
                data[model][dataset] = 0
        if algo in {"knn", "lr"}:
            try:
                data[model]["scope_40_208_3ssp"] = [compute_performance(BASE, model, "scope_40_208", layer, algo=algo, metric=class_metric, aa=True, n_classes=3, task="multi-class") for layer in range(LAYERS[model] + 1)]
            except Exception as e:
                print("\r", model, "scope_40_208_3ssp", "Error:", e)
                data[model]["scope_40_208_3ssp"] = 0
            try:
                data[model]["scope_40_208_8ssp"] = [compute_performance(BASE, model, "scope_40_208", layer, algo=algo, metric=class_metric, aa=True, n_classes=8, task="multi-class") for layer in range(LAYERS[model] + 1)]
            except Exception as e:
                print("\r", model, "scope_40_208_8ssp", "Error:", e)
                data[model]["scope_40_208_8ssp"] = 0
            try:
                data[model]["binding"] = [compute_performance(BASE, model, "binding", layer, algo=algo, metric=class_metric, aa=True, n_classes=2, task="binary") for layer in range(LAYERS[model] + 1)]
            except Exception as e:
                print("\r", model, "binding", "Error:", e)
                data[model]["binding"] = 0
    
    with open(p, "wb") as f:
        pickle.dump(data, f)
        print(f"Saved data to {p.absolute()}")
    return


if __name__ == "__main__":
    matplotlib.rc('font', **{'size': 13})
    # models = ["esm_t33", "esm_t36", "esmc_600m", "ankh_large", "prott5", "prostt5", "progen2_medium", "progen2_large", "protgpt2"]

    # collect_metric("knn", class_metric=CLASS_METRIC, reg_metric="spearman")
    # collect_metric("knn", class_metric=CLASS_METRIC, reg_metric="pearson")
    # collect_metric("knn", class_metric=CLASS_METRIC, reg_metric="r2")
    # collect_metric("lr", class_metric=CLASS_METRIC, reg_metric="spearman")
    # collect_metric("lr", class_metric=CLASS_METRIC, reg_metric="pearson")
    # collect_metric("lr", class_metric=CLASS_METRIC, reg_metric="r2", force=True)
    # collect_metric("ids", class_metric="mcc", reg_metric="pearson", force=True)
    # collect_metric("noverlap", class_metric="mcc", reg_metric="pearson", force=True)
    # collect_metric("pc@95", class_metric="mcc", reg_metric="pearson")
    # collect_metric("var@10", class_metric="mcc", reg_metric="pearson", force=True)

    full_models = list(filter(lambda x: not x.startswith("ankh"), MODELS))

    # plot_new_fig1(full_models, "lr", class_metric=CLASS_METRIC, reg_metric="pearson")
    plot_heatmap(full_models, "knn", class_metric=CLASS_METRIC, reg_metric="pearson")
    # plot_new_fig2(full_models, "lr", class_metric=CLASS_METRIC, reg_metric="pearson")
    # plot_new_fig3()
    # plot_new_fig4(full_models)
    
    # plot_fig1(full_models, "knn", class_metric=CLASS_METRIC, reg_metric="pearson")
    # plot_full_fig1_perfs(full_models, "knn", class_metric=CLASS_METRIC, reg_metric="pearson")
    # plot_full_fig1_perfs(["proteinbert"], "lr", class_metric=CLASS_METRIC, reg_metric="pearson")
    
    # plot_fig2_grouped(full_models)
    # plot_full_fig2_top(full_models)
    # plot_full_fig2_bot(full_models)

    # plot_fig3(full_models, "lr", class_metric=CLASS_METRIC, reg_metric="pearson")
    # plot_full_fig3_left(full_models, "knn", "mcc", "pearson")
    # plot_full_fig3_right()
    
    # plot_fig4()
    # plot_fig5(full_models, "lr", class_metric=CLASS_METRIC, reg_metric=REG_METRIC)
    # plot_fig5_space()
    # plot_fig6()

    # plot_fig1_smtb(full_models, "lr", class_metric=CLASS_METRIC, reg_metric=REG_METRIC)
    # plot_fig4_smtb()
    # plot_fig6_smtb()
