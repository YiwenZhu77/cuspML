#!/usr/bin/env python
"""R045: regenerate Fig 9 (model comparison) with the grid-tuned controlled MLP (1.26).

Standalone, no training: the five bar values are the established results reported in the
manuscript (Newell/Ridge/GBR/XGBoost, Table & baseline-ladder) plus the tuned MLP from
r044_fixes.json. Login-safe (no XGBoost fit). Replaces paper/figures/fig09.

CLAIM : XGBoost 0.97 lowest; tuned MLP 1.26 (best of 12 configs) still worse -> tree wins.
INPUTS: values inline (from paper text + src/kernels/cuspmap_mvp/bundles/r044_fixes.json)
OUTPUT: paper/figures/fig09_model_comparison.{png,pdf}
RUN   : python src/scripts/mvp/r045_fig09.py
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight'})
FIG = "/glade/work/yizhu/cuspML/paper/figures/fig09_model_comparison"

names = ["Linear\n(Newell CF)", "Ridge\n74 features", "GBR\n300-d5",
         "XGBoost\n1000-d8", "MLP\n(tuned)"]
maes  = [1.80, 1.41, 1.11, 0.97, 1.26]
colors = ["#bdbdbd", "#d9b38c", "#74c476", "#b30000", "#9ecae1"]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(names, maes, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
for b, v in zip(bars, maes):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}°", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
bars[3].set_edgecolor("#b30000"); bars[3].set_linewidth(2.0)
ax.set_ylabel("Mean Absolute Error (°MLAT)", fontsize=12)
ax.set_title("Equatorward Boundary Prediction: Model Comparison", fontsize=13)
ax.set_ylim(0, max(maes)*1.18)
ax.tick_params(axis="both", labelsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG + ".png"); fig.savefig(FIG + ".pdf")
print("saved", FIG)
