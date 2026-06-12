from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.viz.constants import LAYERS
from src.viz.utils_general import compute_performance, plot_metric, set_subplot_label
from src.viz.utils_scope import plot_scope_minx_metric



BASE = Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "SMTB"
models = ["esm_t33", "esm_t36", "esmc_600m", "ankh_large", "prott5", "prostt5", "progen2_medium", "progen2_large", "protgpt2"]
# print([compute_performance(BASE, "protgpt2", "deeploc2_bin", layer, algo="knn", metric="mcc", aa=False, n_classes=0, task="binary") for layer in range(LAYERS["protgpt2"] + 1)])
# exit(0)

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 4, figure=fig)
axs = []
for i in range(3):
    axs.append([])
    for j in range(4):
        axs[-1].append(fig.add_subplot(gs[i, j]))
        set_subplot_label(axs[-1][-1], fig, label=f"{chr(ord('A') + i * 4 + j)}")

for m, metric in enumerate(["ids", "noverlap", "var@10"]):
    for d, dataset in enumerate(["fluorescence", "stability", "deeploc2"]):
        plot_metric(axs[m][d], BASE, dataset, model_prefix="", metric=metric, models=models, relative=True, legend=False, title=False)
        if m != 2:
            axs[m][3].set_xlabel("")
    plot_scope_minx_metric(axs[m][3], BASE, metric, "fold", model_prefix="", models=models, relative=True, legend=False, title=False)
    if m != 2:
        axs[m][3].set_xlabel("")

for d, dataset in enumerate(["Fluorescence", "Stability", "DeepLoc2", "SCOPe40 2.08 Fold"]):
    axs[0][d].set_title(dataset)

for m, metric in enumerate(["2NN ID", "Neighborhood Overlap", "Variance @ 10"]):
    axs[m][0].set_ylabel(metric)
    for d in range(1, 4):
        axs[m][d].set_ylabel("")

handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), bbox_transform=fig.transFigure, ncol=(len(models) + 1) // 2)  # -0.08

plt.tight_layout(rect=[0, 0.075, 1, 1])
plt.savefig("layer_metrics.pdf", dpi=300)
plt.show()
