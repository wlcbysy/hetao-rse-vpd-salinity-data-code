import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from sklearn.ensemble import RandomForestRegressor

from figure_common import (
    FEATURES,
    load_master_data,
    salinity_group,
    salinity_quantiles,
    write_threshold_file,
)


def lowess_curve(x, y, frac=0.15):
    z = sm.nonparametric.lowess(y, x, frac=frac, it=0)
    return z[np.argsort(z[:, 0])]


def zero_crossing_to_gain(x, y):
    """Return the first LOWESS crossing from negative to non-negative SHAP."""
    z = lowess_curve(x, y)
    for i in range(1, len(z)):
        y0, y1 = z[i - 1, 1], z[i, 1]
        if y0 <= 0 < y1:
            x0, x1 = z[i - 1, 0], z[i, 0]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float(z[np.argmin(np.abs(z[:, 1])), 0])


def smooth_2d_mean(x, y, values, bins_x, bins_y, sigma=1.2):
    sum_grid = binned_statistic_2d(
        x, y, values, statistic="sum", bins=[bins_x, bins_y]
    ).statistic.T
    count_grid = binned_statistic_2d(
        x, y, values, statistic="count", bins=[bins_x, bins_y]
    ).statistic.T
    sum_grid = np.nan_to_num(sum_grid, nan=0.0)
    count_grid = np.nan_to_num(count_grid, nan=0.0)
    smooth_sum = gaussian_filter(sum_grid, sigma=sigma)
    smooth_count = gaussian_filter(count_grid, sigma=sigma)
    mean_grid = np.divide(
        smooth_sum,
        smooth_count,
        out=np.full_like(smooth_sum, np.nan, dtype=float),
        where=smooth_count > 0,
    )
    mean_grid[smooth_count < 1.5] = np.nan
    return mean_grid, smooth_count


def main():
    print("========== Safe Operating Space: unified threshold analysis ==========")
    df, paths = load_master_data(__file__)
    fig_dir = paths["figures"]
    os.makedirs(fig_dir, exist_ok=True)

    X = df[FEATURES]
    y = df["GPP"]
    q33, q66 = salinity_quantiles(df)

    print("Fitting Random Forest and calculating SHAP on a fixed sample...")
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
    )
    rf_model.fit(X, y)
    X_sample = shap.utils.sample(X, min(3000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    feature_index = {feature: FEATURES.index(feature) for feature in FEATURES}

    vpd = X_sample["VPD"].to_numpy()
    ndsi = X_sample["NDSI"].to_numpy()
    groups = salinity_group(ndsi, q33, q66)
    vpd_shap = shap_values[:, feature_index["VPD"]]
    compound_shap = vpd_shap + shap_values[:, feature_index["NDSI"]]

    thresholds = {}
    curves = {}
    for group in ["Low Salinity", "Medium Salinity", "High Salinity"]:
        mask = groups == group
        thresholds[group] = zero_crossing_to_gain(vpd[mask], vpd_shap[mask])
        curves[group] = lowess_curve(vpd[mask], vpd_shap[mask])
        print(f"{group}: VPD zero-response threshold = {thresholds[group]:.3f} kPa")

    write_threshold_file(
        paths["thresholds"], q33, q66, thresholds, sample_size=len(X_sample)
    )
    print("Threshold file written; releasing model objects...", flush=True)
    del rf_model, explainer, X, y, shap_values
    gc.collect()
    print("Model objects released; starting plotting...", flush=True)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    fig = plt.figure(figsize=(16.5, 5.8), dpi=260)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.22, 1], wspace=0.52)
    print("Figure canvas ready.", flush=True)

    # Panel A: data support in VPD-NDSI space.
    ax_a = fig.add_subplot(gs[0, 0])
    hb = ax_a.hexbin(
        vpd,
        ndsi,
        gridsize=45,
        mincnt=1,
        cmap="Greys",
        linewidths=0,
        bins="log",
    )
    ax_a.axhline(q33, color="#2c7fb8", linestyle="--", linewidth=1.3)
    ax_a.axhline(q66, color="#d95f0e", linestyle="--", linewidth=1.3)
    ax_a.text(0.02, 0.96, "(a)", transform=ax_a.transAxes, fontsize=16, weight="bold")
    ax_a.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_a.set_ylabel("NDSI (higher = saltier)", fontsize=12, weight="bold")
    cbar_a = fig.colorbar(hb, ax=ax_a, pad=0.02)
    cbar_a.set_label("Pixel count (log)", fontsize=10, weight="bold")
    print("Panel A complete.", flush=True)

    # Panel B: compound SHAP safe operating space.
    ax_b = fig.add_subplot(gs[0, 1])
    bins_vpd = np.linspace(vpd.min(), vpd.max(), 58)
    bins_ndsi = np.linspace(ndsi.min(), ndsi.max(), 58)
    z_grid, count_grid = smooth_2d_mean(vpd, ndsi, compound_shap, bins_vpd, bins_ndsi)
    x_centers = (bins_vpd[:-1] + bins_vpd[1:]) / 2
    y_centers = (bins_ndsi[:-1] + bins_ndsi[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
    max_abs = np.nanmax(np.abs(z_grid))
    levels = np.linspace(-max_abs, max_abs, 21)
    contour = ax_b.contourf(
        x_mesh,
        y_mesh,
        z_grid,
        levels=levels,
        cmap="RdYlGn",
        norm=mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
        extend="both",
    )
    ax_b.contour(
        x_mesh,
        y_mesh,
        z_grid,
        levels=[0],
        colors="black",
        linewidths=2.4,
    )
    ax_b.contour(
        x_mesh,
        y_mesh,
        count_grid,
        levels=[2],
        colors="#777777",
        linewidths=0.7,
        linestyles=":",
    )
    ax_b.scatter(vpd, ndsi, c="black", s=1.2, alpha=0.035, linewidths=0)
    ax_b.text(0.02, 0.96, "(b)", transform=ax_b.transAxes, fontsize=16, weight="bold")
    ax_b.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_b.set_ylabel("NDSI (higher = saltier)", fontsize=12, weight="bold")
    cbar_b = fig.colorbar(contour, ax=ax_b, orientation="horizontal", pad=0.16, fraction=0.08)
    cbar_b.set_label("Compound SHAP effect on GPP", fontsize=9, weight="bold")
    print("Panel B complete.", flush=True)

    # Panel C: salinity-specific VPD zero-response thresholds.
    ax_c = fig.add_subplot(gs[0, 2])
    palette = {
        "Low Salinity": "#1b9e77",
        "Medium Salinity": "#7570b3",
        "High Salinity": "#d95f02",
    }
    labels = {
        "Low Salinity": "Low salinity",
        "Medium Salinity": "Medium salinity",
        "High Salinity": "High salinity",
    }
    for group, curve in curves.items():
        ax_c.plot(curve[:, 0], curve[:, 1], color=palette[group], linewidth=2.4, label=labels[group])
        ax_c.axvline(thresholds[group], color=palette[group], linestyle="--", linewidth=1.3)
        ax_c.text(
            thresholds[group],
            ax_c.get_ylim()[0] if ax_c.get_ylim()[0] < 0 else -0.006,
            f"{thresholds[group]:.2f}",
            color=palette[group],
            fontsize=9,
            rotation=90,
            va="bottom",
            ha="right",
        )
    ax_c.axhline(0, color="#555555", linestyle=":", linewidth=1.2)
    ax_c.text(0.02, 0.96, "(c)", transform=ax_c.transAxes, fontsize=16, weight="bold")
    ax_c.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_c.set_ylabel("SHAP value of VPD", fontsize=12, weight="bold")
    ax_c.legend(
        frameon=False,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    print("Panel C complete.", flush=True)

    for ax in [ax_a, ax_b, ax_c]:
        sns.despine(ax=ax, top=True, right=True)
        ax.tick_params(labelsize=10)

    out_fig = os.path.join(fig_dir, "08_Advanced_Safe_Operating_Space.png")
    final_fig = os.path.join(fig_dir, "Final_Fig4_Safe_Operating_Space.png")
    fig.savefig(out_fig, bbox_inches="tight", dpi=350)
    fig.savefig(final_fig, bbox_inches="tight", dpi=350)
    plt.close(fig)
    print(f"Saved: {out_fig}")
    print(f"Saved: {final_fig}")


if __name__ == "__main__":
    main()
