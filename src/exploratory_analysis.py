"""Descriptive summaries for each source without pooling observations."""

import pandas as pd

from .data_cleaning import METRICS


def describe(frame):
    summary = frame[METRICS].describe().T.rename_axis("metric")
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary.reset_index()


def regional_summary(frame, groups):
    return (frame.groupby(groups)["unemployment_rate"]
            .agg(n="count", mean="mean", median="median", sd="std", minimum="min", maximum="max")
            .sort_values("mean", ascending=False).reset_index())


def exploratory_tables(area, regional):
    stats = pd.concat([describe(frame).assign(source=name) for name, frame in
                       [("area", area), ("regional", regional)]], ignore_index=True)
    peaks = pd.concat([frame.nlargest(10, "unemployment_rate").assign(source=name)
                       for name, frame in [("area", area), ("regional", regional)]], ignore_index=True)
    table = regional_summary(regional, "region")
    return {"descriptive_statistics": stats, "regional_summary": table,
            "area_summary": regional_summary(area, "area"),
            "region_area_summary": regional_summary(area, ["region", "area"]),
            "largest_observations": peaks}

