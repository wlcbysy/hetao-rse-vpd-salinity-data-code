import pandas as pd

from project_layout import ANALYSIS_OUTPUTS_DIR, MASTER_DATA, ensure_output_dirs


OUTPUT_CSV = ANALYSIS_OUTPUTS_DIR / "nirv_margin_triangulation.csv"
CORRELATION_CSV = ANALYSIS_OUTPUTS_DIR / "nirv_margin_triangulation_correlations.csv"
SUMMARY_MD = ANALYSIS_OUTPUTS_DIR / "nirv_margin_triangulation_summary.md"


def zscore(series):
    std = series.std(ddof=0)
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def main():
    ensure_output_dirs()
    master = pd.read_csv(MASTER_DATA, usecols=["Year", "NIRv"]).dropna()
    sensitivity = pd.read_csv(ANALYSIS_OUTPUTS_DIR / "threshold_sensitivity.csv")
    unified = sensitivity[
        sensitivity["threshold_set"] == "unified_zero_response_thresholds"
    ][["year", "exceedance_area_percent", "mean_vpd_margin_kpa"]].rename(
        columns={"year": "Year"}
    )

    annual = (
        master.groupby("Year")["NIRv"]
        .agg(pixel_count="count", nirv_mean="mean", nirv_median="median", nirv_std="std")
        .reset_index()
    )
    annual["nirv_mean_z"] = zscore(annual["nirv_mean"])
    annual["nirv_median_z"] = zscore(annual["nirv_median"])
    out = annual.merge(unified, on="Year", how="inner").sort_values("Year")

    correlations = pd.DataFrame(
        [
            {
                "metric_a": "nirv_mean_z",
                "metric_b": "exceedance_area_percent",
                "pearson_r": out["nirv_mean_z"].corr(out["exceedance_area_percent"]),
            },
            {
                "metric_a": "nirv_mean_z",
                "metric_b": "mean_vpd_margin_kpa",
                "pearson_r": out["nirv_mean_z"].corr(out["mean_vpd_margin_kpa"]),
            },
            {
                "metric_a": "exceedance_area_percent",
                "metric_b": "mean_vpd_margin_kpa",
                "pearson_r": out["exceedance_area_percent"].corr(out["mean_vpd_margin_kpa"]),
            },
        ]
    )

    out.to_csv(OUTPUT_CSV, index=False)
    correlations.to_csv(CORRELATION_CSV, index=False)

    corr_exceedance = correlations.loc[
        correlations["metric_b"] == "exceedance_area_percent", "pearson_r"
    ].iloc[0]
    corr_margin = correlations.loc[
        correlations["metric_b"] == "mean_vpd_margin_kpa", "pearson_r"
    ].iloc[0]
    SUMMARY_MD.write_text(
        "\n".join(
            [
                "# NIRv Margin Triangulation Summary",
                "",
                "Generated from `Hetao_Master_Dataset_2000_2023.csv` and "
                "`threshold_sensitivity.csv`.",
                "",
                "## Purpose",
                "",
                "This ancillary check compares annual mean NIRv with unified VPD-margin "
                "exposure metrics. It tests whether high-margin years are also years with "
                "low optical vegetation signal. It is not a yield, flux or field-salinity "
                "validation.",
                "",
                "## Results",
                "",
                f"- Years compared: {int(out['Year'].min())}-{int(out['Year'].max())}",
                f"- Annual NIRv mean z-score vs exceedance area: r = {corr_exceedance:.3f}",
                f"- Annual NIRv mean z-score vs mean VPD margin: r = {corr_margin:.3f}",
                "",
                "## Interpretation",
                "",
                "The weak positive associations indicate that high VPD-margin years were "
                "not necessarily low-NIRv years. The VPD-margin layer should therefore be "
                "interpreted as atmospheric-demand and NDSI-background exposure screening, "
                "not as a direct estimate of canopy decline, yield loss or economic damage.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {CORRELATION_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
