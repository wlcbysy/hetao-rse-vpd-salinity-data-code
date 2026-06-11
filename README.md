# Hetao VPD-Salinity Remote-Sensing Data and Code

This repository contains the processed data, analysis scripts and final figure
outputs for the manuscript:

**A remote-sensing safe operating space for vapor pressure deficit-salinity
compound stress in the Hetao Irrigation District, China**

## Contents

- `data/`: processed pixel-year tables and aligned raster products used in the
  analysis.
- `data/timeseries/`: annual eco-meteorological raster stacks from 2000 to 2023.
- `code/`: Google Earth Engine export scripts, local preprocessing scripts,
  model interpretation scripts and figure-generation scripts.
- `figures/final/`: final manuscript figures.
- `results/`: model-performance, threshold and sensitivity-summary outputs used
  as numeric sources for the manuscript.

## Evidence Boundary

The processed table `data/Hetao_Master_Dataset_2000_2023.csv` contains 355,944
pixel-year observations from 2000 to 2023. True MOD17A2H GPP values are present
only for 2021 to 2023, yielding 43,269 observed-GPP samples. The random forest
and SHAP response analysis therefore use only the observed-GPP subset. The full
2000 to 2023 covariate record is used only for VPD/salinity exposure projection
after the response boundary is derived.

## Reproducibility Notes

The main scripts are intended to be run from the repository root in numeric
order after the required Google Earth Engine exports or processed local data are
available:

1. `code/04_data_preprocessing.py`
2. `code/05_model_shap_analysis.py`
3. `code/08_six_panel_main_figure.py`
4. `code/09_safe_operating_space.py`
5. `code/10_spatiotemporal_risk_mapping.py`
6. `code/12_plot_fig1_study_area.py`
7. `code/11_compile_final_figures.py`

Earlier scripts (`01` to `03`) document the Google Earth Engine export route.

## Key Outputs

- Random forest test R2: `0.949`
- Random forest test RMSE: `0.00234 g C m-2 day-1`
- Salinity-stratified VPD zero-response thresholds: `1.748`, `1.757` and
  `1.763 kPa`
- Mean annual threshold exceedance, 2000 to 2023: `31.4%` of cropland pixels

The exact values are recorded in:

- `results/model_performance.txt`
- `results/threshold_results.txt`
- `results/threshold_sensitivity.csv`

## Data Sources

Raw satellite, reanalysis and land-cover products were obtained from NASA LP
DAAC, Copernicus/ECMWF, Sentinel-2 and ESA WorldCover. This repository stores
processed and aligned derivatives used for the manuscript analysis.

## Citation

Please cite the associated manuscript when using this data/code package. A
permanent archive DOI can be added after the GitHub repository is archived in a
research data repository such as Zenodo.
