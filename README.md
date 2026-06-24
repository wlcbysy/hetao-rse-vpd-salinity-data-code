# Hetao VPD-Salinity Remote-Sensing Data and Code

This repository contains the processed data, analysis scripts and final figure
outputs for the manuscript:

**Monitoring vapor pressure deficit-salinity exposure margins for irrigated
cropland productivity in the Hetao Irrigation District, China**

## Contents

- `data/`: processed pixel-year tables and aligned raster products used in the
  analysis.
- `data/timeseries/`: annual eco-meteorological raster stacks from 2000 to 2023.
- `code/`: Google Earth Engine export scripts, local preprocessing scripts,
  model interpretation scripts and figure-generation scripts.
- `figures/final/`: final manuscript figures.
- `results/`: model-performance, threshold and sensitivity-summary outputs used
  as numeric sources for the manuscript, including the ancillary NIRv
  triangulation check.

## Evidence Boundary

The processed table `data/Hetao_Master_Dataset_2000_2023.csv` contains 355,944
pixel-year observations from 2000 to 2023. Annual GPP layers were screened before
response modeling. The 2022 local MOD17A2H GPP band is all zero and is excluded
from response modeling as an invalid product layer. QC-passed observed GPP is
therefore limited to 2021 and 2023, yielding 28,846 observed-GPP pixel-years.
GPP values were converted from scaled 8-day kg C m-2 composites to daily
g C m-2 day-1 using a factor of 125.0. The random forest and SHAP response
analysis use only the QC-passed observed-GPP subset. The full 2000 to 2023
covariate record is used only for VPD/NDSI exposure projection after the
response boundary is derived.

## Reproducibility Notes

The main scripts are intended to be run from the repository root in numeric
order after the required Google Earth Engine exports or processed local data are
available:

1. `code/04_data_preprocessing.py`
2. `code/05_model_shap_analysis.py`
3. `code/08_six_panel_main_figure.py`
4. `code/09_safe_operating_space.py`
5. `code/10_spatiotemporal_risk_mapping.py`
6. `code/13_nirv_margin_triangulation.py`
7. `code/12_plot_fig1_study_area.py`
8. `code/11_compile_final_figures.py`

Earlier scripts (`01` to `03`) document the Google Earth Engine export route.

## Key Outputs

- Random forest random-test R2: `0.809`
- Random forest random-test RMSE: `0.300 g C m-2 day-1`
- Spatial-block cross-validation R2: `0.593`
- Spatial-block cross-validation RMSE: `0.433 g C m-2 day-1`
- NDSI-background VPD zero-response thresholds: `1.946`, `1.960` and
  `1.964 kPa`
- Mean annual threshold exceedance, 2000 to 2023: `8.3%` of cropland pixels
- NIRv mean z-score correlations with unified exposure metrics: `r = 0.242`
  for exceedance area and `r = 0.346` for mean VPD margin. This check supports
  interpreting VPD-margin maps as exposure screening rather than direct
  canopy-decline or yield-loss estimates.

The exact values are recorded in:

- `results/model_performance.txt`
- `results/threshold_results.txt`
- `results/threshold_sensitivity.csv`
- `results/nirv_margin_triangulation.csv`
- `results/nirv_margin_triangulation_correlations.csv`
- `results/nirv_margin_triangulation_summary.md`

## Data Sources

Raw satellite, reanalysis and land-cover products were obtained from NASA LP
DAAC, Copernicus/ECMWF, Sentinel-2 and ESA WorldCover. This repository stores
processed and aligned derivatives used for the manuscript analysis.

## Citation

Please cite the associated manuscript when using this data/code package. The
public GitHub repository is available at:

https://github.com/wlcbysy/hetao-rse-vpd-salinity-data-code

A permanent archive DOI can be added after the GitHub repository is archived in a
research data repository such as Zenodo.
