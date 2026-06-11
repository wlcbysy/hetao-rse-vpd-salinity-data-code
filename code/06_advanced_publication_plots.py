import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor

from figure_common import (
    FEATURES,
    load_master_data,
    read_threshold_file,
    salinity_quantiles,
    thresholds_from_file,
)


def main():
    print("========== Generating Figure 2: global drivers and exposure space ==========")
    df, paths = load_master_data(__file__)
    fig_dir = paths["figures"]
    os.makedirs(fig_dir, exist_ok=True)

    X = df[FEATURES]
    y = df["GPP"]
    q33, q66 = salinity_quantiles(df)
    threshold_meta = read_threshold_file(paths["thresholds"])
    q33 = threshold_meta.get("NDSI_Q33", q33)
    q66 = threshold_meta.get("NDSI_Q66", q66)
    thresholds = thresholds_from_file(paths["thresholds"])

    print(f"Effective sample size: {len(X)} pixel-year observations")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
    )
    rf_model.fit(X, y)
    X_sample = shap.utils.sample(X, min(3000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1.1

    fig = plt.figure(figsize=(15, 6.4), dpi=400)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.34)

    ax_bees = fig.add_subplot(gs[0, 0])
    plt.sca(ax_bees)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=FEATURES,
        show=False,
        alpha=0.45,
        cmap="viridis",
        plot_size=None,
    )
    ax_bees.set_xlabel("SHAP value (impact on GPP)", fontweight="bold")
    ax_bees.text(-0.1, 1.05, "(a)", transform=ax_bees.transAxes, size=16, weight="bold")
    if len(fig.axes) > 2:
        fig.axes[-1].set_ylabel("Feature value", fontsize=11, fontweight="bold")

    ax_biv = fig.add_subplot(gs[0, 1])
    sns.kdeplot(
        x=X_sample["VPD"],
        y=X_sample["NDSI"],
        ax=ax_biv,
        cmap="Reds",
        fill=True,
        thresh=0.04,
        levels=12,
        alpha=0.86,
    )
    ax_biv.scatter(
        X_sample["VPD"],
        X_sample["NDSI"],
        s=2,
        color="black",
        alpha=0.08,
        linewidths=0,
    )
    ax_biv.axhline(q33, color="#2c7fb8", linestyle="--", linewidth=1.2, label="NDSI 33%")
    ax_biv.axhline(q66, color="#d95f0e", linestyle="--", linewidth=1.2, label="NDSI 66%")
    for group, color in [
        ("Low Salinity", "#1b9e77"),
        ("Medium Salinity", "#7570b3"),
        ("High Salinity", "#d95f02"),
    ]:
        ax_biv.axvline(thresholds[group], color=color, linestyle=":", linewidth=1.1)
    ax_biv.set_xlabel("VPD (kPa)", fontweight="bold")
    ax_biv.set_ylabel("NDSI (higher = saltier)", fontweight="bold")
    ax_biv.text(-0.1, 1.05, "(b)", transform=ax_biv.transAxes, size=16, weight="bold")
    ax_biv.text(
        0.985,
        q33,
        "33%",
        transform=ax_biv.get_yaxis_transform(),
        color="#2c7fb8",
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.0),
    )
    ax_biv.text(
        0.985,
        q66,
        "66%",
        transform=ax_biv.get_yaxis_transform(),
        color="#d95f0e",
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.0),
    )
    ax_biv.grid(True, linestyle=":", alpha=0.45)
    sns.despine(ax=ax_biv, top=True, right=True)

    out_fig = os.path.join(fig_dir, "05_Paper_Summary_Beeswarm.png")
    final_fig = os.path.join(fig_dir, "Final_Fig2_Drivers.png")
    fig.savefig(out_fig, dpi=500, bbox_inches="tight")
    fig.savefig(final_fig, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_fig}")
    print(f"Saved: {final_fig}")


if __name__ == "__main__":
    main()
