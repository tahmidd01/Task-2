"""Fixed panels and within-region pandemic contrasts."""

import numpy as np
import pandas as pd

from .data_cleaning import METRICS

PERIODS = ["Baseline", "Transition", "Shock", "Reopening"]
PERIOD_LABELS = {
    "Baseline": "Jan-Feb 2020", "Transition": "Mar 2020",
    "Shock": "Apr-May 2020", "Reopening": "Jun onward",
}


def classify_period(months):
    months = pd.to_datetime(months).dt.to_period("M")
    labels = pd.Series("Historical", index=months.index, dtype="string").mask(months.isna())
    labels.loc[months.between(pd.Period("2020-01"), pd.Period("2020-02"))] = "Baseline"
    labels.loc[months.eq(pd.Period("2020-03"))] = "Transition"
    labels.loc[months.between(pd.Period("2020-04"), pd.Period("2020-05"))] = "Shock"
    labels.loc[months.between(pd.Period("2020-06"), pd.Period("2020-12"))] = "Reopening"
    labels.loc[months.ge(pd.Period("2021-01"))] = "Outside study"
    return labels


def balanced_panel(frame, months, *, both_areas=False, metrics=METRICS):
    """Select regions with usable data at every requested month (and both areas when needed)."""
    panel = frame.loc[frame["month"].isin(months)].dropna(subset=metrics).copy()
    required = len(months)
    if both_areas:
        required *= 2
        panel = panel.loc[panel["area"].isin(["Rural", "Urban"])]
    counts = panel.groupby("region")["month"].transform("size")
    return panel.loc[counts.eq(required)].copy()


def monthly_panel(panel, groups):
    return (panel.groupby(groups)[METRICS].mean().join(
        panel.groupby(groups)["region"].nunique().rename("regions")).reset_index())


def period_table(panel, groups):
    averages = panel.groupby([*groups, "period"])[METRICS].mean()
    pandemic = panel.loc[panel["period"].isin(["Shock", "Reopening"])]
    pandemic = pandemic.groupby(groups)[METRICS].mean().assign(period="Pandemic")
    averages = pd.concat([averages.reset_index(), pandemic.reset_index()], ignore_index=True)
    baseline = averages.loc[averages["period"].eq("Baseline"), [*groups, *METRICS]]
    averages = averages.merge(baseline, on=groups, suffixes=("", "_baseline"), validate="many_to_one")
    averages["unemployment_change_pp"] = averages["unemployment_rate"] - averages["unemployment_rate_baseline"]
    averages["unemployment_relative_pct"] = 100 * averages["unemployment_change_pp"] / averages[
        "unemployment_rate_baseline"].replace(0, np.nan)
    averages["participation_change_pp"] = averages["participation_rate"] - averages["participation_rate_baseline"]
    averages["employment_change_people"] = averages["employed"] - averages["employed_baseline"]
    averages["employment_relative_pct"] = 100 * averages["employment_change_people"] / averages[
        "employed_baseline"].replace(0, np.nan)
    order = [*PERIODS, "Pandemic"]
    averages["period"] = pd.Categorical(averages["period"], order, ordered=True)
    return averages.sort_values([*groups, "period"]).reset_index(drop=True)


def covid_tables(area, regional):
    long_months = pd.date_range("2020-01-01", regional["month"].max(), freq="MS")
    area_months = pd.date_range("2020-01-01", area["month"].max(), freq="MS")
    long = balanced_panel(regional, long_months).assign(group="Regional panel")
    urban_rural = balanced_panel(area, area_months, both_areas=True)
    if long.empty or urban_rural.empty:
        raise ValueError("The fixed comparison panels require complete regional and area coverage")
    for frame in [long, urban_rural]:
        frame["period"] = classify_period(frame["month"])
    core = balanced_panel(regional, pd.date_range("2020-01-01", "2020-05-01", freq="MS"))
    core["period"] = classify_period(core["month"])
    area_history = balanced_panel(area, pd.date_range(area["month"].min(), area["month"].max(), freq="MS"),
                                  both_areas=True)
    area_history = area_history.assign(period=classify_period(area_history["month"]))
    area_history = area_history.loc[~area_history["period"].eq("Historical")].copy()
    return {
        "monthly_regional_panel": monthly_panel(long, ["month", "period"]),
        "monthly_area_panel": monthly_panel(urban_rural, ["month", "period", "area"]),
        "covid_periods": period_table(long, ["group"]),
        "covid_by_region": period_table(long, ["region"]),
        "covid_by_area": period_table(urban_rural, ["area"]),
        "covid_by_region_area": period_table(urban_rural, ["region", "area"]),
        "covid_sensitivity": pd.concat([
            period_table(core.assign(panel="Jan-May complete"), ["panel"]),
            period_table(long.assign(panel="Jan-Oct complete"), ["panel"]),
        ], ignore_index=True),
        "area_cohort_sensitivity": period_table(area_history, ["area"]),
        "regional_panel_members": long[["region"]].drop_duplicates(),
        "area_panel_members": urban_rural[["region"]].drop_duplicates(),
        "regional_panel_observations": long.drop(columns="group"),
        "area_panel_observations": urban_rural,
    }

