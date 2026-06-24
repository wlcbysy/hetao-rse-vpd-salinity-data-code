from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
STANDARD_INPUT_DIR = DATA_DIR
SPATIAL_DATA_DIR = DATA_DIR
DERIVED_RASTER_DIR = DATA_DIR
TIMESERIES_RASTER_DIR = DATA_DIR / "timeseries"

MASTER_DATA = STANDARD_INPUT_DIR / "Hetao_Master_Dataset_2000_2023.csv"
DATA_AUDIT = DATA_DIR / "Hetao_Master_Dataset_2000_2023_Audit.md"

FIGURES_DIR = ROOT / "figures"
MAIN_FIGURES_DIR = FIGURES_DIR / "final"
DIAGNOSTIC_FIGURES_DIR = FIGURES_DIR / "diagnostic"

TABLES_DIR = ROOT / "results"
ANALYSIS_OUTPUTS_DIR = TABLES_DIR
THRESHOLD_RESULTS = ANALYSIS_OUTPUTS_DIR / "threshold_results.txt"
THRESHOLD_SENSITIVITY = ANALYSIS_OUTPUTS_DIR / "threshold_sensitivity.csv"
MODEL_PERFORMANCE = ANALYSIS_OUTPUTS_DIR / "model_performance.txt"

MANUSCRIPT_DIR = ROOT / "manuscript_outputs"
MANUSCRIPT_BUILD_DIR = MANUSCRIPT_DIR / "Build_Scripts"
MANUSCRIPT_QA_DIR = MANUSCRIPT_DIR / "QA_Reports"
RENDERED_QA_DIR = MANUSCRIPT_DIR / "Rendered_QA"

SUBMISSION_PACKAGE_DIR = ROOT / "submission_outputs"
IRRIGATION_SCIENCE_SUBMISSION_DIR = (
    SUBMISSION_PACKAGE_DIR / "Irrigation_Science_submission_package_final"
)
IRRIGATION_SCIENCE_RENDER_DIR = RENDERED_QA_DIR / "Irrigation_Science_submission_package_final"

REPOSITORY_DEPOSITION_DIR = ROOT


def ensure_output_dirs():
    for path in [
        STANDARD_INPUT_DIR,
        DERIVED_RASTER_DIR,
        TIMESERIES_RASTER_DIR,
        DATA_AUDIT.parent,
        MAIN_FIGURES_DIR,
        DIAGNOSTIC_FIGURES_DIR,
        ANALYSIS_OUTPUTS_DIR,
        MANUSCRIPT_DIR,
        MANUSCRIPT_QA_DIR,
        RENDERED_QA_DIR,
        SUBMISSION_PACKAGE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
