# Figure QA Checklist

Generated on 2026-06-11 for the revised 5-main-figure package after supplementing the local 2000-2023 covariate record.

## Unified Data and Thresholds

- Expanded master table: `data/Hetao_Master_Dataset_2000_2023.csv`
- Backup of previous 2021-2023 table: `data/Hetao_Master_Dataset_2021_2023_MOD17_backup.csv`
- Build audit: `data/Hetao_Master_Dataset_2000_2023_Audit.md`
- Available covariate years in the current master table: 2000-2023
- Row count: 355,944 pixel-year observations, 14,831 per year
- True GPP coverage: 2021-2023 only, 14,423 valid MOD17A2H GPP rows per year
- 2000-2020 local yearly TIFFs contain LST, NIRv, Tmax, SM, and VPD, but no GPP band
- NDSI salinity quantiles from the GPP-observed SHAP sample: q33 = -0.316382, q66 = -0.268388
- Unified VPD zero-response thresholds:
  - Low salinity: 1.747825 kPa
  - Medium salinity: 1.757473 kPa
  - High salinity: 1.762781 kPa
- Threshold source of truth: `figures/threshold_results.txt`

## Final Figures

| Figure | File | Main checks |
| :--- | :--- | :--- |
| Figure 1 | `Final_Fig1_StudyArea_Concept.png` | Uses 2023 true GPP and NDSI maps; adds scale bar, 2023 summer label, pixel-density labels, and NDSI quantile lines. |
| Figure 2 | `Final_Fig2_Drivers.png` | Uses only rows with true GPP for SHAP; shows global SHAP distribution plus VPD-NDSI exposure space. |
| Figure 3 | `Final_Fig3_6Panel_Mechanisms.png` | Uses only rows with true GPP; adds model validation, salinity-specific VPD response curves, SM/LST/Tmax/NDSI response panels, and local threshold markers. |
| Figure 4 | `Final_Fig4_Safe_Operating_Space.png` | Uses true-GPP SHAP sample for observed density, annotated compound SHAP safe-space response, bootstrap boundary uncertainty envelopes, and salinity-specific bootstrap threshold distributions. |
| Figure 5 | `Final_Fig5_Spatiotemporal_Risk.png` | Uses 2000-2023 VPD/NDSI covariates for long-term exposure projection and 2023 continuous VPD-margin map. |

## Sensitivity Check

The legacy hard-coded thresholds remained over-saturated across the expanded 2000-2023 record.

| Threshold set | Min annual exceedance | Mean annual exceedance | Max annual exceedance |
| :--- | ---: | ---: | ---: |
| Old hard-coded thresholds | 37.60% | 94.04% | 100.00% |
| Unified zero-response thresholds | 0.00% | 31.41% | 97.07% |

The revised Figure 5 therefore uses continuous VPD margin and three risk classes rather than a binary all-danger map.

## Manuscript Sync

- Figure numbering is consolidated to Figure 1-Figure 5.
- Current manuscript should describe two evidence layers: true-GPP SHAP modeling for 2021-2023 and VPD/NDSI exposure projection for 2000-2023.
- Table 3 uses VPD zero-response threshold language, not absolute disaster-threshold language.
