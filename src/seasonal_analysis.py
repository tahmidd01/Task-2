"""Coverage-aware monthly summaries and descriptive seasonality checks."""

import pandas as pd

from .covid_analysis import balanced_panel
from .data_cleaning import METRICS


def temporal_tables(area, regional):
    # Fix both the region and area composition across the historical trend.
    months = pd.date_range(area["month"].min(), area["month"].max(), freq="MS")
    fixed = balanced_panel(area, months, both_areas=True)
    monthly = fixed.groupby("month")[METRICS].mean().reindex(months).rename_axis("month")
    monthly["regions"] = fixed.groupby("month")["region"].nunique()
    monthly["rolling_3m_rate"] = monthly["unemployment_rate"].rolling(3, min_periods=3).mean()
    monthly["change_pp"] = monthly["unemployment_rate"].diff()
    monthly = monthly.reset_index()
    monthly["year"] = monthly["month"].dt.year
    monthly["calendar_month"] = monthly["month"].dt.month
    calendar = monthly.groupby("calendar_month")["unemployment_rate"].agg(
        mean="mean", observed_years="count").reset_index()
    years = monthly.groupby("year")["unemployment_rate"].agg(mean="mean", observed_months="count").reset_index()
    heatmap = monthly.pivot(index="year", columns="calendar_month", values="unemployment_rate")
    matched = heatmap.T.dropna().reset_index()
    if {2019, 2020}.issubset(matched.columns):
        matched["change_pp"] = matched[2020] - matched[2019]
    expected = pd.MultiIndex.from_product(
        [sorted(area["region"].unique()), months, ["Rural", "Urban"]],
        names=["region", "month", "area"],
    )
    missing_cells = expected.difference(pd.MultiIndex.from_frame(area[["region", "month", "area"]]))
    regional_months = pd.date_range(regional["month"].min(), regional["month"].max(), freq="MS")
    expected_regional = pd.MultiIndex.from_product(
        [sorted(regional["region"].unique()), regional_months], names=["region", "month"],
    )
    missing_regional = expected_regional.difference(pd.MultiIndex.from_frame(regional[["region", "month"]]))
    return {"historical_monthly": monthly, "historical_years": years, "calendar_months": calendar,
            "calendar_heatmap": heatmap.reset_index(), "same_month_year_comparison": matched,
            "missing_region_month_areas": missing_cells.to_frame(index=False),
            "missing_regional_months": missing_regional.to_frame(index=False),
            "regional_heatmap": regional.pivot(index="region", columns="month", values="unemployment_rate").reset_index()}

