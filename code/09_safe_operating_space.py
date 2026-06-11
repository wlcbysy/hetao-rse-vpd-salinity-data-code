import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import binned_statistic_2d, gaussian_kde
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


def bootstrap_thresholds(vpd, vpd_shap, groups, point_thresholds, n_boot=180, random_state=42):
    """Estimate sampling uncertainty for salinity-specific zero-response thresholds."""
    rng = np.random.default_rng(random_state)
    records = []
    sample_records = []
    for group in ["Low Salinity", "Medium Salinity", "High Salinity"]:
        idx = np.flatnonzero(groups == group)
        boot_values = []
        for replicate in range(n_boot):
            sample_idx = rng.choice(idx, size=len(idx), replace=True)
            try:
                threshold = zero_crossing_to_gain(vpd[sample_idx], vpd_shap[sample_idx])
                boot_values.append(threshold)
                sample_records.append(
                    {
                        "salinity_group": group,
                        "replicate": replicate + 1,
                        "threshold_kpa": threshold,
                    }
                )
            except (ValueError, IndexError, FloatingPointError):
                continue

        boot_values = np.array(boot_values, dtype=float)
        boot_values = boot_values[np.isfinite(boot_values)]
        records.append(
            {
                "salinity_group": group,
                "point_threshold_kpa": point_thresholds[group],
                "bootstrap_median_kpa": float(np.nanmedian(boot_values)),
                "bootstrap_ci_low_kpa": float(np.nanpercentile(boot_values, 2.5)),
                "bootstrap_ci_high_kpa": float(np.nanpercentile(boot_values, 97.5)),
                "successful_bootstraps": int(len(boot_values)),
            }
        )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(sample_records)


def boundary_crossings_from_grid(z_grid, x_centers, y_centers):
    """Extract one zero-response boundary y-value for each VPD column."""
    boundary = np.full(len(x_centers), np.nan, dtype=float)
    for j in range(len(x_centers)):
        z = z_grid[:, j]
        valid = np.isfinite(z)
        if valid.sum() < 3:
            continue
        z_valid = z[valid]
        y_valid = y_centers[valid]
        crossings = np.flatnonzero((z_valid[:-1] >= 0) & (z_valid[1:] < 0))
        if len(crossings) == 0:
            crossings = np.flatnonzero((z_valid[:-1] <= 0) & (z_valid[1:] > 0))
        if len(crossings) == 0:
            continue
        k = crossings[len(crossings) // 2]
        z0, z1 = z_valid[k], z_valid[k + 1]
        y0, y1 = y_valid[k], y_valid[k + 1]
        if z1 == z0:
            boundary[j] = (y0 + y1) / 2
        else:
            boundary[j] = y0 + (0 - z0) * (y1 - y0) / (z1 - z0)
    return boundary


def smooth_nan_line(values, sigma=1.1):
    out = np.array(values, dtype=float)
    finite = np.isfinite(out)
    if finite.sum() < 4:
        return out
    idx = np.arange(len(out))
    filled = np.interp(idx, idx[finite], out[finite])
    smoothed = gaussian_filter1d(filled, sigma=sigma)
    smoothed[~finite] = np.nan
    return smoothed


def bootstrap_boundary_envelopes(
    vpd,
    ndsi,
    compound_shap,
    bins_vpd,
    bins_ndsi,
    x_centers,
    y_centers,
    n_boot=140,
    random_state=7,
):
    """Return bootstrap zero-response boundary samples and summary envelopes."""
    rng = np.random.default_rng(random_state)
    n = len(vpd)
    boundaries = []

    for i in range(n_boot):
        sample_idx = rng.integers(0, n, n)
        boot_grid, _ = smooth_2d_mean(
            vpd[sample_idx],
            ndsi[sample_idx],
            compound_shap[sample_idx],
            bins_vpd,
            bins_ndsi,
            sigma=1.2,
            min_support=1.5,
        )
        boundaries.append(boundary_crossings_from_grid(boot_grid, x_centers, y_centers))
        if (i + 1) % 35 == 0:
            print(f"  Bootstrap boundary replicate {i + 1}/{n_boot}", flush=True)

    boundary_samples = np.vstack(boundaries)
    valid_count = np.isfinite(boundary_samples).sum(axis=0)
    enough = valid_count >= max(20, n_boot * 0.45)
    summary = {
        "median": np.full(len(x_centers), np.nan, dtype=float),
        "p25": np.full(len(x_centers), np.nan, dtype=float),
        "p75": np.full(len(x_centers), np.nan, dtype=float),
        "p025": np.full(len(x_centers), np.nan, dtype=float),
        "p975": np.full(len(x_centers), np.nan, dtype=float),
    }
    for key, q in [("median", 50), ("p25", 25), ("p75", 75), ("p025", 2.5), ("p975", 97.5)]:
        vals = np.full(len(x_centers), np.nan, dtype=float)
        for j, ok in enumerate(enough):
            if ok:
                vals[j] = np.nanpercentile(boundary_samples[:, j], q)
        summary[key] = smooth_nan_line(vals)
    return boundary_samples, summary


def smooth_2d_mean(x, y, values, bins_x, bins_y, sigma=1.2, min_support=1.5):
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
    mean_grid[smooth_count < min_support] = np.nan
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
    for group in ["Low Salinity", "Medium Salinity", "High Salinity"]:
        mask = groups == group
        thresholds[group] = zero_crossing_to_gain(vpd[mask], vpd_shap[mask])
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

    bins_vpd = np.linspace(vpd.min(), vpd.max(), 58)
    bins_ndsi = np.linspace(ndsi.min(), ndsi.max(), 58)
    z_grid, count_grid = smooth_2d_mean(
        vpd, ndsi, compound_shap, bins_vpd, bins_ndsi, min_support=1.5
    )
    z_grid_soft, _ = smooth_2d_mean(
        vpd, ndsi, compound_shap, bins_vpd, bins_ndsi, sigma=1.7, min_support=0.08
    )
    x_centers = (bins_vpd[:-1] + bins_vpd[1:]) / 2
    y_centers = (bins_ndsi[:-1] + bins_ndsi[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
    max_abs = np.nanmax(np.abs(z_grid))
    levels = np.linspace(-max_abs, max_abs, 21)

    print("Running bootstrap stability diagnostics...", flush=True)
    baseline_boundary = smooth_nan_line(boundary_crossings_from_grid(z_grid, x_centers, y_centers))
    boundary_samples, boundary_summary = bootstrap_boundary_envelopes(
        vpd, ndsi, compound_shap, bins_vpd, bins_ndsi, x_centers, y_centers
    )
    threshold_bootstrap, threshold_bootstrap_samples = bootstrap_thresholds(vpd, vpd_shap, groups, thresholds)
    threshold_bootstrap_path = os.path.join(fig_dir, "fig4_threshold_bootstrap.csv")
    threshold_samples_path = os.path.join(fig_dir, "fig4_threshold_bootstrap_samples.csv")
    threshold_bootstrap.to_csv(threshold_bootstrap_path, index=False)
    threshold_bootstrap_samples.to_csv(threshold_samples_path, index=False)
    boundary_path = os.path.join(fig_dir, "fig4_boundary_bootstrap_envelope.csv")
    pd.DataFrame(
        {
            "vpd_kpa": x_centers,
            "mean_boundary_ndsi": baseline_boundary,
            "bootstrap_median_ndsi": boundary_summary["median"],
            "bootstrap_p25_ndsi": boundary_summary["p25"],
            "bootstrap_p75_ndsi": boundary_summary["p75"],
            "bootstrap_p025_ndsi": boundary_summary["p025"],
            "bootstrap_p975_ndsi": boundary_summary["p975"],
        }
    ).to_csv(boundary_path, index=False)
    print(f"Bootstrap threshold table written: {threshold_bootstrap_path}", flush=True)
    print(f"Bootstrap threshold samples written: {threshold_samples_path}", flush=True)
    print(f"Bootstrap boundary envelope written: {boundary_path}", flush=True)

    focus_xlim = (1.48, 2.18)
    focus_ylim = (-0.45, -0.05)

    fig = plt.figure(figsize=(13.2, 10.2), dpi=260)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.34, hspace=0.34)
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
    ax_a.set_title("Observed VPD-salinity support", fontsize=13, weight="bold", pad=8)
    ax_a.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_a.set_ylabel("NDSI (higher = saltier)", fontsize=12, weight="bold")
    cbar_a = fig.colorbar(hb, ax=ax_a, pad=0.02)
    cbar_a.set_label("Pixel count (log)", fontsize=10, weight="bold")
    print("Panel A complete.", flush=True)

    # Panel B: mean compound SHAP safe operating space.
    ax_b = fig.add_subplot(gs[0, 1])
    safe_space_cmap = mcolors.LinearSegmentedColormap.from_list(
        "safe_space",
        ["#8c510a", "#d8b365", "#f7f7f7", "#80cdc1", "#01665e"],
        N=256,
    )
    contour = ax_b.contourf(
        x_mesh,
        y_mesh,
        z_grid_soft,
        levels=levels,
        cmap=safe_space_cmap,
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
    low_support = np.where(count_grid <= 1.5, 1.0, np.nan)
    ax_b.contourf(
        x_mesh,
        y_mesh,
        low_support,
        levels=[0.5, 1.5],
        colors=["white"],
        alpha=0.42,
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
    ax_b.contour(
        x_mesh,
        y_mesh,
        z_grid,
        levels=[0],
        colors="black",
        linewidths=2.4,
    )
    ax_b.scatter(vpd, ndsi, c="black", s=1.0, alpha=0.025, linewidths=0, rasterized=True)
    ax_b.annotate(
        "Negative compound\ncontribution",
        xy=(1.63, -0.22),
        xytext=(1.52, -0.075),
        arrowprops=dict(arrowstyle="-", color="#5f3b08", linewidth=1.0),
        fontsize=9,
        color="#5f3b08",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2.5),
    )
    ax_b.annotate(
        "Positive / near-neutral\ncontribution",
        xy=(1.94, -0.36),
        xytext=(1.86, -0.42),
        arrowprops=dict(arrowstyle="-", color="#005f56", linewidth=1.0),
        fontsize=9,
        color="#005f56",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2.5),
    )
    ax_b.text(
        0.98,
        0.95,
        "Dotted line: observed support",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    ax_b.text(0.02, 0.96, "(b)", transform=ax_b.transAxes, fontsize=16, weight="bold")
    ax_b.set_title("Zoomed decision surface", fontsize=13, weight="bold", pad=8)
    ax_b.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_b.set_ylabel("NDSI (higher = saltier)", fontsize=12, weight="bold")
    ax_b.set_xlim(*focus_xlim)
    ax_b.set_ylim(*focus_ylim)
    cbar_b = fig.colorbar(contour, ax=ax_b, pad=0.02)
    cbar_b.set_label("Mean compound SHAP effect on GPP", fontsize=9, weight="bold")
    print("Panel B complete.", flush=True)

    # Panel C: bootstrap uncertainty envelope around the zero-response boundary.
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.fill_between(
        x_centers,
        boundary_summary["p025"],
        boundary_summary["p975"],
        color="#d0d0d0",
        alpha=0.52,
        linewidth=0,
        label="95% bootstrap envelope",
    )
    ax_c.fill_between(
        x_centers,
        boundary_summary["p25"],
        boundary_summary["p75"],
        color="#737373",
        alpha=0.36,
        linewidth=0,
        label="50% bootstrap envelope",
    )
    ax_c.plot(
        x_centers,
        boundary_summary["median"],
        color="#111111",
        linewidth=2.5,
        label="Bootstrap median boundary",
    )
    ax_c.plot(
        x_centers,
        baseline_boundary,
        color="#00796b",
        linewidth=1.6,
        linestyle="--",
        label="Original mean boundary",
    )
    ax_c.axhline(q33, color="#2c7fb8", linestyle=":", linewidth=1.0, alpha=0.75)
    ax_c.axhline(q66, color="#d95f0e", linestyle=":", linewidth=1.0, alpha=0.75)
    ax_c.text(focus_xlim[0] + 0.01, q33 - 0.006, "NDSI 33%", fontsize=8.2, color="#2c7fb8", va="top")
    ax_c.text(focus_xlim[0] + 0.01, q66 + 0.006, "NDSI 66%", fontsize=8.2, color="#d95f0e", va="bottom")
    ax_c.text(0.02, 0.96, "(c)", transform=ax_c.transAxes, fontsize=16, weight="bold")
    ax_c.set_title("Boundary fan chart", fontsize=13, weight="bold", pad=8)
    ax_c.set_xlabel("VPD (kPa)", fontsize=12, weight="bold")
    ax_c.set_ylabel("NDSI boundary (higher = saltier)", fontsize=12, weight="bold")
    ax_c.set_xlim(*focus_xlim)
    ax_c.set_ylim(*focus_ylim)
    ax_c.grid(True, axis="both", linestyle=":", alpha=0.28)
    ax_c.legend(frameon=False, fontsize=8.3, loc="lower right")
    print("Panel C complete.", flush=True)

    # Panel D: salinity-specific VPD zero-response threshold distributions.
    ax_d = fig.add_subplot(gs[1, 1])
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
    ordered_groups = ["Low Salinity", "Medium Salinity", "High Salinity"]
    y_pos = np.arange(len(ordered_groups))[::-1]
    for y_i, group in zip(y_pos, ordered_groups):
        row = threshold_bootstrap[threshold_bootstrap["salinity_group"] == group].iloc[0]
        values = threshold_bootstrap_samples.loc[
            threshold_bootstrap_samples["salinity_group"] == group, "threshold_kpa"
        ].to_numpy()
        x_support = np.linspace(values.min() - 0.008, values.max() + 0.008, 240)
        kde = gaussian_kde(values)
        density = kde(x_support)
        density = density / density.max() * 0.22
        ax_d.fill_between(
            x_support,
            y_i - density,
            y_i + density,
            color=palette[group],
            alpha=0.28,
            linewidth=0,
        )
        ax_d.plot(x_support, y_i + density, color=palette[group], linewidth=1.4)
        ax_d.plot(x_support, y_i - density, color=palette[group], linewidth=1.4)
        ax_d.hlines(
            y_i,
            row["bootstrap_ci_low_kpa"],
            row["bootstrap_ci_high_kpa"],
            color=palette[group],
            linewidth=2.4,
            alpha=0.95,
        )
        ax_d.plot(
            row["bootstrap_median_kpa"],
            y_i,
            marker="o",
            markersize=6.8,
            color=palette[group],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax_d.plot(
            row["point_threshold_kpa"],
            y_i,
            marker="D",
            markersize=5.3,
            color="white",
            markeredgecolor=palette[group],
            markeredgewidth=1.4,
        )
        ax_d.text(
            row["bootstrap_ci_high_kpa"] + 0.005,
            y_i,
            f"{row['bootstrap_median_kpa']:.3f} kPa",
            va="center",
            fontsize=9,
            color="#222222",
        )

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([labels[group] for group in ordered_groups], fontsize=10)
    x_min = threshold_bootstrap["bootstrap_ci_low_kpa"].min() - 0.04
    x_max = threshold_bootstrap["bootstrap_ci_high_kpa"].max() + 0.075
    ax_d.set_xlim(x_min, x_max)
    for threshold in thresholds.values():
        ax_d.axvline(threshold, color="#d9d9d9", linestyle=":", linewidth=0.8, zorder=0)
    ax_d.grid(axis="x", linestyle=":", alpha=0.45)
    ax_d.text(0.02, 0.96, "(d)", transform=ax_d.transAxes, fontsize=16, weight="bold")
    ax_d.set_title("Threshold bootstrap distributions", fontsize=13, weight="bold", pad=8)
    ax_d.set_xlabel("VPD zero-response threshold (kPa)", fontsize=12, weight="bold")
    ax_d.set_ylabel("")
    median_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="#333333",
        markerfacecolor="#333333",
        linewidth=0,
        label="Bootstrap median",
    )
    point_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="#333333",
        markerfacecolor="white",
        linewidth=0,
        label="Original estimate",
    )
    ci_handle = plt.Line2D([0], [0], color="#333333", linewidth=4, alpha=0.65, label="95% bootstrap interval")
    ax_d.legend(
        handles=[ci_handle, median_handle, point_handle],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    print("Panel D complete.", flush=True)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
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
