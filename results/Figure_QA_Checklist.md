# Figure QA Checklist

Generated on 2026-06-24 after GPP product-QC correction, MOD17A2H unit conversion, spatial-block validation and full figure regeneration.

## Unified Data and Thresholds

- Expanded master table: `01_Data/02_Standardized_Analysis_Input/Hetao_Master_Dataset_2000_2023.csv`
- Build audit: `04_Tables/Data_Audit/Hetao_Master_Dataset_2000_2023_Audit.md`
- Available covariate years in the current master table: 2000-2023
- Row count: 355,944 pixel-year observations, 14,831 retained cropland pixels per year
- QC-passed observed GPP coverage: 2021 and 2023 only, 28,846 total pixel-years
- 2022 local GPP status: all-zero GPP band excluded from response modeling
- GPP unit handling: scaled 8-day MOD17A2H kg C m-2 composites converted to daily g C m-2 day-1 by multiplying by 125.0
- 2000-2020 local yearly TIFFs contain LST, NIRv, Tmax, SM and VPD, but no GPP band
- NDSI quantiles from the QC-passed observed-GPP SHAP sample: q33 = -0.316382, q66 = -0.268389
- Unified VPD zero-response thresholds:
  - Low NDSI: 1.945913 kPa
  - Medium NDSI: 1.959654 kPa
  - High NDSI: 1.963876 kPa
- Threshold source of truth: `04_Tables/Analysis_Outputs/threshold_results.txt`
- Model-performance source of truth: `04_Tables/Analysis_Outputs/model_performance.txt`

## Model Validation

- Random held-out R2: 0.809378
- Random held-out RMSE: 0.299988 g C m-2 day-1
- Spatial-block CV R2: 0.593309
- Spatial-block CV RMSE: 0.432721 g C m-2 day-1
- Spatial-block folds: 16

## Final Figures

| Figure | File | Main checks |
| :--- | :--- | :--- |
| Figure 1 | `Final_Fig1_StudyArea_Concept.png` | Uses 2023 QC-passed GPP and NDSI maps; adds scale bar, pixel-density labels and NDSI quantile lines. |
| Figure 2 | `Final_Fig2_Drivers.png` | Uses the QC-passed observed-GPP subset for SHAP; shows global SHAP distribution plus VPD-NDSI exposure space. |
| Figure 3 | `Final_Fig3_6Panel_Mechanisms.png` | Reports random and spatial-block validation; shows NDSI-background VPD response curves, SM/LST/Tmax/NDSI response panels and local threshold markers. |
| Figure 4 | `Final_Fig4_Safe_Operating_Space.png` | Uses the fixed SHAP sample for observed density, compound SHAP response, bootstrap boundary uncertainty envelopes and NDSI-stratified bootstrap threshold distributions. |
| Figure 5 | `Final_Fig5_Spatiotemporal_Risk.png` | Uses 2000-2023 VPD/NDSI covariates for long-term exposure projection and the 2023 continuous VPD-margin map. |

## Sensitivity Check

The old hard-coded threshold set remained over-saturated across the expanded 2000-2023 record, while the unified SHAP-derived thresholds produced a conservative exposure-margin series.

| Threshold set | Min annual exceedance | Mean annual exceedance | Max annual exceedance |
| :--- | ---: | ---: | ---: |
| Old hard-coded thresholds | 37.60% | 94.04% | 100.00% |
| Unified zero-response thresholds | 0.00% | 8.29% | 52.32% |

The revised Figure 5 therefore uses continuous VPD margin and three exposure classes rather than a binary all-danger map.

## Manuscript Sync

- Figure numbering is consolidated to Figure 1-Figure 5.
- Current manuscript describes two evidence layers: QC-passed observed-GPP SHAP modeling for 2021 and 2023, and VPD/NDSI exposure projection for 2000-2023.
- Table 3 uses NDSI-background VPD zero-response threshold language, not absolute salinity or disaster-threshold language.
