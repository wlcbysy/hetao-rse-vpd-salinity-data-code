import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from figure_common import (
    FEATURES,
    apply_risk_thresholds,
    load_master_data,
    read_threshold_file,
    salinity_group,
    salinity_quantiles,
    thresholds_from_file,
)


def lowess_curve(x, y, frac=0.12):
    z = sm.nonparametric.lowess(y, x, frac=frac, it=0)
    return z[np.argsort(z[:, 0])]


def zero_crossings(curve):
    crossings = []
    for i in range(1, len(curve)):
        y0, y1 = curve[i - 1, 1], curve[i, 1]
        if y0 == 0 or y1 == 0 or y0 * y1 < 0:
            x0, x1 = curve[i - 1, 0], curve[i, 0]
            crossings.append(float(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)))
    return crossings


def draw_dependence(
    ax,
    x_data,
    shap_data,
    c_data,
    x_label,
    y_label,
    c_label,
    letter,
    vlines=None,
):
    scatter = ax.scatter(
        x_data,
        shap_data,
        c=c_data,
        cmap="Spectral_r",
        s=5,
        alpha=0.45,
        edgecolors="none",
        zorder=2,
    )
    sns.kdeplot(
        x=x_data,
        y=shap_data,
        ax=ax,
        levels=5,
        color="#444444",
        alpha=0.35,
        linewidths=0.7,
        zorder=3,
    )
    curve = lowess_curve(x_data, shap_data)
    ax.plot(curve[:, 0], curve[:, 1], color="#d7191c", linewidth=2.2, zorder=4)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=1)

    crossings = zero_crossings(curve)
    if crossings:
        cross = crossings[len(crossings) // 2]
        ax.axvline(cross, color="#222222", linestyle=":", linewidth=1.1)
        ax.text(
            cross,
            0.96,
            f"{cross:.2f}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            fontsize=8,
            va="top",
            ha="right",
        )

    if vlines:
        for x, label, color in vlines:
            ax.axvline(x, color=color, linestyle="--", linewidth=1.1)
            ax.text(
                x,
                0.04,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                fontsize=8,
                color=color,
                va="bottom",
                ha="left",
            )

    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.text(-0.14, 1.13, letter, transform=ax.transAxes, fontsize=15, fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)

    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.38, pad=0.06, sharex=ax)
    ax_histy = divider.append_axes("right", 0.38, pad=0.06, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    sns.kdeplot(x=x_data, ax=ax_histx, fill=True, color="gray", alpha=0.25)
    sns.kdeplot(y=shap_data, ax=ax_histy, fill=True, color="gray", alpha=0.25)
    ax_histx.axis("off")
    ax_histy.axis("off")

    cax = inset_axes(ax, width="34%", height="5%", loc="lower right", borderpad=1.3)
    cbar = plt.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_label(c_label, fontsize=8, fontweight="bold")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=7)


def draw_model_performance(ax, y_test, y_pred, r2, rmse):
    ax.scatter(y_test, y_pred, s=7, alpha=0.35, color="#3182bd", edgecolors="none")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=1.4)
    ax.text(
        0.05,
        0.92,
        f"R² = {r2:.3f}\nRMSE = {rmse:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="#BBBBBB"),
    )
    ax.set_xlabel("Observed GPP", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted GPP", fontsize=11, fontweight="bold")
    ax.text(-0.14, 1.13, "(a)", transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax, top=True, right=True)


def draw_grouped_vpd(ax, X_sample, vpd_shap, q33, q66, thresholds):
    vpd = X_sample["VPD"].to_numpy()
    ndsi = X_sample["NDSI"].to_numpy()
    groups = salinity_group(ndsi, q33, q66)
    palette = {
        "Low Salinity": "#1b9e77",
        "Medium Salinity": "#7570b3",
        "High Salinity": "#d95f02",
    }
    for group in ["Low Salinity", "Medium Salinity", "High Salinity"]:
        mask = groups == group
        ax.scatter(vpd[mask], vpd_shap[mask], s=5, alpha=0.12, color=palette[group], edgecolors="none")
        curve = lowess_curve(vpd[mask], vpd_shap[mask])
        ax.plot(curve[:, 0], curve[:, 1], color=palette[group], linewidth=2.4, label=group.replace(" Salinity", ""))
        ax.axvline(thresholds[group], color=palette[group], linestyle="--", linewidth=1.1)
        ax.text(
            thresholds[group],
            0.04,
            f"{thresholds[group]:.2f}",
            transform=ax.get_xaxis_transform(),
            color=palette[group],
            rotation=90,
            fontsize=8,
            va="bottom",
            ha="left",
        )
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.2)
    ax.set_xlabel("VPD (kPa)", fontsize=11, fontweight="bold")
    ax.set_ylabel("SHAP value of VPD", fontsize=11, fontweight="bold")
    ax.text(-0.14, 1.13, "(b)", transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.2,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.25,
    )
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax, top=True, right=True)


def main():
    print("========== Generating revised six-panel mechanism figure ==========")
    df, paths = load_master_data(__file__)
    fig_dir = paths["figures"]
    os.makedirs(fig_dir, exist_ok=True)

    X = df[FEATURES]
    y = df["GPP"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    validation_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
    )
    validation_model.fit(X_train, y_train)
    y_pred = validation_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
    )
    rf_model.fit(X, y)
    X_sample = shap.utils.sample(X, min(3000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)
    feature_index = {feature: FEATURES.index(feature) for feature in FEATURES}

    threshold_meta = read_threshold_file(paths["thresholds"])
    q33, q66 = salinity_quantiles(df)
    q33 = threshold_meta.get("NDSI_Q33", q33)
    q66 = threshold_meta.get("NDSI_Q66", q66)
    thresholds = thresholds_from_file(paths["thresholds"])
    risk_df = apply_risk_thresholds(df, q33, q66, thresholds)

    with open(os.path.join(fig_dir, "model_performance.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test R2: {r2:.6f}\n")
        f.write(f"Test RMSE: {rmse:.6f}\n")
        f.write(f"Training N: {len(X_train)}\n")
        f.write(f"Test N: {len(X_test)}\n")
        f.write(f"Unified exceedance area percent: {risk_df['Risk_Flag'].mean() * 100:.3f}\n")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2), dpi=400)
    plt.subplots_adjust(wspace=0.34, hspace=0.38)

    draw_model_performance(axes[0, 0], y_test, y_pred, r2, rmse)
    draw_grouped_vpd(
        axes[0, 1],
        X_sample,
        shap_values[:, feature_index["VPD"]],
        q33,
        q66,
        thresholds,
    )
    draw_dependence(
        axes[0, 2],
        X_sample["SM"].to_numpy(),
        shap_values[:, feature_index["SM"]],
        X_sample["LST"].to_numpy(),
        "Soil moisture (m³/m³)",
        "SHAP value of SM",
        "LST (°C)",
        "(c)",
    )
    draw_dependence(
        axes[1, 0],
        X_sample["LST"].to_numpy(),
        shap_values[:, feature_index["LST"]],
        X_sample["SM"].to_numpy(),
        "Land surface temperature (°C)",
        "SHAP value of LST",
        "SM (m³/m³)",
        "(d)",
    )
    draw_dependence(
        axes[1, 1],
        X_sample["Tmax"].to_numpy(),
        shap_values[:, feature_index["Tmax"]],
        X_sample["VPD"].to_numpy(),
        "Maximum air temperature (°C)",
        "SHAP value of Tmax",
        "VPD (kPa)",
        "(e)",
    )
    draw_dependence(
        axes[1, 2],
        X_sample["NDSI"].to_numpy(),
        shap_values[:, feature_index["NDSI"]],
        X_sample["VPD"].to_numpy(),
        "NDSI (higher = saltier)",
        "SHAP value of NDSI",
        "VPD (kPa)",
        "(f)",
        vlines=[(q33, "33%", "#2c7fb8"), (q66, "66%", "#d95f0e")],
    )

    out_fig = os.path.join(fig_dir, "07_Premium_6Panel_Main_Figure.png")
    final_fig = os.path.join(fig_dir, "Final_Fig3_6Panel_Mechanisms.png")
    fig.savefig(out_fig, bbox_inches="tight", dpi=500)
    fig.savefig(final_fig, bbox_inches="tight", dpi=500)
    plt.close(fig)
    print(f"Saved: {out_fig}")
    print(f"Saved: {final_fig}")
    print("Saved: figures/model_performance.txt")


if __name__ == "__main__":
    main()
