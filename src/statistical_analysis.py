"""Inference on paired region-level changes, plus descriptive associations."""

import numpy as np
import pandas as pd
from scipy import stats

from .data_cleaning import METRICS


def paired_statistics(contrasts, label, *, seed=2020, draws=20000):
    differences = np.asarray(contrasts, dtype=float)
    if differences.ndim != 1 or len(differences) < 3 or not np.isfinite(differences).all():
        raise ValueError("Paired inference requires at least three finite regional contrasts")
    n = len(differences)
    mean = differences.mean()
    sd = differences.std(ddof=1)
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, size=(draws, n), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    up = int((differences > 0).sum())
    down = int((differences < 0).sum())
    trimmed = (differences.sum() - differences) / (n - 1)
    return {
        "contrast": label, "regions": n, "mean_pp": mean, "median_pp": np.median(differences),
        "sd_pp": sd, "paired_cohens_dz": mean / sd if sd else np.nan,
        "bootstrap_95_low_pp": low, "bootstrap_95_high_pp": high,
        "bootstrap_draws": draws, "positive": up, "negative": down, "ties": n - up - down,
        "exact_sign_p": stats.binomtest(up, up + down).pvalue if up + down else 1.0,
        "shapiro_p": stats.shapiro(differences).pvalue if sd else np.nan,
        "leave_one_out_min_pp": trimmed.min(), "leave_one_out_max_pp": trimmed.max(),
    }


def statistics_tables(regional, area, covid):
    shock = covid["covid_by_region"].query("period == 'Shock'")
    area_shock = covid["covid_by_region_area"].query("period == 'Shock'").pivot(
        index="region", columns="area", values="unemployment_change_pp")
    estimates = pd.DataFrame([
        paired_statistics(shock["unemployment_change_pp"], "Regional shock minus baseline"),
        paired_statistics(area_shock["Urban"] - area_shock["Rural"], "Urban minus rural shock change"),
    ])
    tables = {"paired_inference": estimates}
    for name, frame in [("regional", regional), ("area", area)]:
        # Within-unit Pearson removes unit means; shared time shocks remain in the residuals.
        keys = ["region"] + (["area"] if name == "area" else [])
        centered = frame[METRICS] - frame.groupby(keys)[METRICS].transform("mean")
        for method, matrix in [("pearson", frame[METRICS].corr()), ("within_unit_pearson", centered.corr()),
                               ("spearman", frame[METRICS].corr(method="spearman"))]:
            tables[f"{name}_{method}"] = matrix.rename_axis("metric").reset_index()
    return tables

