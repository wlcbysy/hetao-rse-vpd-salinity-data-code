import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_common import (
    LEGACY_HARDCODED_THRESHOLDS_FOR_SENSITIVITY,
    apply_risk_thresholds,
    load_master_data,
    read_threshold_file,
    salinity_quantiles,
    thresholds_from_file,
)


RISK_ORDER = [
    "Within zero-response boundary",
    "0-0.20 kPa above boundary",
    ">0.20 kPa above boundary",
]
RISK_COLORS = {
    "Within zero-response boundary": "#2c7fb8",
    "0-0.20 kPa above boundary": "#fdae61",
    ">0.20 kPa above boundary": "#d7191c",
}


def classify_margin(margin):
    return pd.cut(
        margin,
        bins=[-np.inf, 0, 0.20, np.inf],
        labels=RISK_ORDER,
        include_lowest=True,
    )


def prepare_risk_data(df, q33, q66, thresholds):
    out = apply_risk_thresholds(df, q33, q66, thresholds)
    out["Risk_Class"] = classify_margin(out["Risk_Margin_kPa"])
    return out


def plot_temporal(ax, risk_df):
    yearly = (
        pd.crosstab(risk_df["Year"], risk_df["Risk_Class"], normalize="index")
        .reindex(columns=RISK_ORDER, fill_value=0)
        * 100
    )
    bottom = np.zeros(len(yearly))
    x = yearly.index.astype(int).to_numpy()
    year_min = int(x.min())
    year_max = int(x.max())
    for label in RISK_ORDER:
        values = yearly[label].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.65,
            color=RISK_COLORS[label],
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        bottom += values

    ax2 = ax.twinx()
    mean_margin = risk_df.groupby("Year")["Risk_Margin_kPa"].mean().reindex(yearly.index)
    ax2.plot(
        x,
        mean_margin.to_numpy(),
        color="black",
        marker="o",
        linewidth=2.0,
        label="Mean VPD margin",
    )
    ax2.set_ylabel("Mean VPD margin (kPa)", fontsize=11, weight="bold")
    ax2.tick_params(labelsize=10)
    ax2.spines["top"].set_visible(False)

    ax.set_ylim(0, 100)
    tick_step = max(1, int(np.ceil(len(x) / 8)))
    ticks = list(x[::tick_step])
    if ticks[-1] != year_max:
        ticks.append(year_max)
    ax.set_xticks(ticks)
    ax.set_xlabel("Available year", fontsize=12, weight="bold")
    ax.set_ylabel("Cropland area proportion (%)", fontsize=12, weight="bold")
    ax.text(
        0.015,
        0.96,
        "(a)",
        transform=ax.transAxes,
        fontsize=16,
        weight="bold",
        va="top",
        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.4),
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    return yearly, mean_margin


def plot_spatial(ax, risk_df):
    df_2023 = risk_df[risk_df["Year"] == risk_df["Year"].max()].copy()
    map_year = int(df_2023["Year"].iloc[0])
    margin = df_2023["Risk_Margin_kPa"].to_numpy()
    vmax = float(np.nanpercentile(np.abs(margin), 98))
    vmax = max(vmax, 0.05)
    hb = ax.hexbin(
        df_2023["Lon"],
        df_2023["Lat"],
        C=margin,
        reduce_C_function=np.mean,
        gridsize=130,
        mincnt=1,
        cmap="RdYlBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        linewidths=0,
    )
    ax.scatter(
        df_2023["Lon"],
        df_2023["Lat"],
        s=0.15,
        c="black",
        alpha=0.05,
        linewidths=0,
        rasterized=True,
    )
    ax.annotate(
        "N",
        xy=(0.96, 0.96),
        xytext=(0.96, 0.86),
        arrowprops=dict(facecolor="black", width=2.5, headwidth=9),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.set_xlabel("Longitude (°E)", fontsize=12, weight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=12, weight="bold")
    ax.text(0.01, 0.97, "(b)", transform=ax.transAxes, fontsize=16, weight="bold", va="top")
    ax.grid(True, linestyle=":", alpha=0.45)
    safe_patch = mpatches.Patch(color=RISK_COLORS[RISK_ORDER[0]], label="≤ boundary")
    moderate_patch = mpatches.Patch(color=RISK_COLORS[RISK_ORDER[1]], label="0-0.20 kPa above")
    high_patch = mpatches.Patch(color=RISK_COLORS[RISK_ORDER[2]], label=">0.20 kPa above")
    ax.legend(
        handles=[safe_patch, moderate_patch, high_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    return hb


def write_sensitivity(fig_dir, df, q33, q66, thresholds):
    records = []
    for name, thresh in [
        ("old_hardcoded_thresholds", LEGACY_HARDCODED_THRESHOLDS_FOR_SENSITIVITY),
        ("unified_zero_response_thresholds", thresholds),
    ]:
        risk = prepare_risk_data(df, q33, q66, thresh)
        for year, group in risk.groupby("Year"):
            records.append(
                {
                    "threshold_set": name,
                    "year": int(year),
                    "exceedance_area_percent": group["Risk_Flag"].mean() * 100,
                    "mean_vpd_margin_kpa": group["Risk_Margin_kPa"].mean(),
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(fig_dir, "threshold_sensitivity.csv"), index=False)


def main():
    print("========== Spatiotemporal risk mapping with unified thresholds ==========")
    df, paths = load_master_data(
        __file__,
        required_columns=["Year", "Lon", "Lat", "NDSI", "VPD"],
    )
    fig_dir = paths["figures"]
    os.makedirs(fig_dir, exist_ok=True)

    threshold_meta = read_threshold_file(paths["thresholds"])
    thresholds = thresholds_from_file(paths["thresholds"])
    q33, q66 = salinity_quantiles(df)
    q33 = threshold_meta.get("NDSI_Q33", q33)
    q66 = threshold_meta.get("NDSI_Q66", q66)
    print(f"Using NDSI thresholds: q33={q33:.3f}, q66={q66:.3f}")
    print("Using VPD zero-response thresholds:", thresholds)

    risk_df = prepare_risk_data(df, q33, q66, thresholds)
    write_sensitivity(fig_dir, df, q33, q66, thresholds)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1.1

    fig = plt.figure(figsize=(12, 12), dpi=400)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.25], hspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    plot_temporal(ax_a, risk_df)
    hb = plot_spatial(ax_b, risk_df)
    cbar = fig.colorbar(hb, ax=ax_b, pad=0.015)
    cbar.set_label(
        "VPD margin above salinity-specific zero-response boundary (kPa)",
        fontsize=10,
        weight="bold",
    )
    for ax in [ax_a, ax_b]:
        ax.tick_params(labelsize=10)
        sns.despine(ax=ax, top=True, right=True)

    final_path = os.path.join(fig_dir, "Final_Fig5_Spatiotemporal_Risk.png")
    fig.savefig(final_path, bbox_inches="tight", dpi=500)
    plt.close(fig)

    fig_a, ax_a_only = plt.subplots(figsize=(10, 5.8), dpi=350)
    plot_temporal(ax_a_only, risk_df)
    sns.despine(ax=ax_a_only, top=True, right=True)
    fig_a.savefig(os.path.join(fig_dir, "09_Temporal_Risk_Trend.png"), bbox_inches="tight", dpi=400)
    plt.close(fig_a)

    fig_b, ax_b_only = plt.subplots(figsize=(10, 7), dpi=350)
    hb_only = plot_spatial(ax_b_only, risk_df)
    cbar_only = fig_b.colorbar(hb_only, ax=ax_b_only, pad=0.015)
    cbar_only.set_label("VPD margin above boundary (kPa)", fontsize=10, weight="bold")
    sns.despine(ax=ax_b_only, top=True, right=True)
    fig_b.savefig(os.path.join(fig_dir, "10_Spatial_Risk_Map_2023.png"), bbox_inches="tight", dpi=400)
    plt.close(fig_b)

    print(f"Saved: {final_path}")
    print("Saved: figures/09_Temporal_Risk_Trend.png")
    print("Saved: figures/10_Spatial_Risk_Map_2023.png")
    print("Saved: figures/threshold_sensitivity.csv")


if __name__ == "__main__":
    main()
