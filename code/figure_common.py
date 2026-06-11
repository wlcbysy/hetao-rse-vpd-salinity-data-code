import os
import re

import numpy as np
import pandas as pd


FEATURES = ["VPD", "NDSI", "SM", "LST", "Tmax"]
FEATURE_LABELS = {
    "VPD": "VPD",
    "NDSI": "NDSI",
    "SM": "Soil moisture",
    "LST": "Land surface temperature",
    "Tmax": "Maximum air temperature",
}
LEGACY_HARDCODED_THRESHOLDS_FOR_SENSITIVITY = {
    "Low Salinity": 1.247,
    "Medium Salinity": 1.320,
    "High Salinity": 1.381,
}


def project_paths(current_file):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
    return {
        "base": base_dir,
        "data": os.path.join(base_dir, "data", "Hetao_Master_Dataset_2000_2023.csv"),
        "figures": os.path.join(base_dir, "figures"),
        "thresholds": os.path.join(base_dir, "figures", "threshold_results.txt"),
    }


def load_master_data(current_file, required_columns=None):
    paths = project_paths(current_file)
    df = pd.read_csv(paths["data"]).replace([np.inf, -np.inf], np.nan)
    if required_columns is None:
        required_columns = FEATURES + ["GPP"]
    df = df.dropna(subset=required_columns)
    return df, paths


def salinity_quantiles(df):
    return float(df["NDSI"].quantile(0.33)), float(df["NDSI"].quantile(0.66))


def salinity_group(ndsi, q33, q66):
    return np.select(
        [ndsi < q33, (ndsi >= q33) & (ndsi <= q66), ndsi > q66],
        ["Low Salinity", "Medium Salinity", "High Salinity"],
        default="Medium Salinity",
    )


def write_threshold_file(path, q33, q66, thresholds, sample_size):
    lines = [
        "# Unified thresholds for figure generation",
        f"NDSI_Q33: {q33:.6f}",
        f"NDSI_Q66: {q66:.6f}",
        f"Low Salinity Threshold: {thresholds['Low Salinity']:.6f}",
        f"Medium Salinity Threshold: {thresholds['Medium Salinity']:.6f}",
        f"High Salinity Threshold: {thresholds['High Salinity']:.6f}",
        f"SHAP_Sample_Size: {sample_size}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_threshold_file(path):
    values = {}
    if not os.path.exists(path):
        return values
    pattern = re.compile(r"^([^:#]+):\s*([-+]?\d+(?:\.\d+)?)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                values[match.group(1).strip()] = float(match.group(2))
    return values


def thresholds_from_file(path):
    values = read_threshold_file(path)
    required = [
        "Low Salinity Threshold",
        "Medium Salinity Threshold",
        "High Salinity Threshold",
    ]
    if not all(key in values for key in required):
        missing = [key for key in required if key not in values]
        raise ValueError(
            f"Unified threshold file is missing required entries: {missing}. "
            "Run code/09_safe_operating_space.py first."
        )
    return {
        "Low Salinity": values["Low Salinity Threshold"],
        "Medium Salinity": values["Medium Salinity Threshold"],
        "High Salinity": values["High Salinity Threshold"],
    }


def apply_risk_thresholds(df, q33, q66, thresholds):
    groups = salinity_group(df["NDSI"].to_numpy(), q33, q66)
    group_thresholds = np.select(
        [
            groups == "Low Salinity",
            groups == "Medium Salinity",
            groups == "High Salinity",
        ],
        [
            thresholds["Low Salinity"],
            thresholds["Medium Salinity"],
            thresholds["High Salinity"],
        ],
        default=thresholds["Medium Salinity"],
    )
    out = df.copy()
    out["Salinity_Group"] = groups
    out["VPD_Limit_kPa"] = group_thresholds
    out["Risk_Margin_kPa"] = out["VPD"] - out["VPD_Limit_kPa"]
    out["Risk_Flag"] = (out["Risk_Margin_kPa"] > 0).astype(int)
    return out
