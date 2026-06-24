import os

from PIL import Image

from figure_common import read_threshold_file
from project_layout import MAIN_FIGURES_DIR, THRESHOLD_RESULTS


FINAL_FIGURES = [
    ("Figure 1", "Final_Fig1_StudyArea_Concept.png"),
    ("Figure 2", "Final_Fig2_Drivers.png"),
    ("Figure 3", "Final_Fig3_6Panel_Mechanisms.png"),
    ("Figure 4", "Final_Fig4_Safe_Operating_Space.png"),
    ("Figure 5", "Final_Fig5_Spatiotemporal_Risk.png"),
]


def inspect_image(fig_dir, label, filename):
    path = os.path.join(fig_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} is missing: {path}")
    with Image.open(path) as img:
        width, height = img.size
    if width < 1600 or height < 1200:
        raise ValueError(f"{label} is too small for submission use: {width} x {height}")
    print(f"{label}: {filename} | {width} x {height}px")


def main():
    fig_dir = str(MAIN_FIGURES_DIR)
    threshold_path = str(THRESHOLD_RESULTS)

    print("========== Final figure QA: 5-main-figure package ==========")
    for label, filename in FINAL_FIGURES:
        inspect_image(fig_dir, label, filename)

    thresholds = read_threshold_file(threshold_path)
    required = [
        "NDSI_Q33",
        "NDSI_Q66",
        "Low Salinity Threshold",
        "Medium Salinity Threshold",
        "High Salinity Threshold",
    ]
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise ValueError(f"Unified threshold file is incomplete: {missing}")

    print("Unified thresholds:")
    print(f" - NDSI q33 / q66: {thresholds['NDSI_Q33']:.3f}, {thresholds['NDSI_Q66']:.3f}")
    print(f" - Low-NDSI VPD threshold: {thresholds['Low Salinity Threshold']:.3f} kPa")
    print(f" - Medium-NDSI VPD threshold: {thresholds['Medium Salinity Threshold']:.3f} kPa")
    print(f" - High-NDSI VPD threshold: {thresholds['High Salinity Threshold']:.3f} kPa")
    print("========== Final figure QA passed ==========")


if __name__ == "__main__":
    main()
