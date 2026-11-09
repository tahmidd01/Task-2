"""Consistent, print-ready charts with explicit denominators and coverage."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

BLUE = "#155575"
TEAL = "#087F83"
ORANGE = "#AD4B29"
INK = "#172A38"
GRAY = "#455965"
SHADE = "#FAE7DA"
LABELS = {"unemployment_rate": "Unemployment", "employed": "Estimated employed",
          "participation_rate": "Labour participation"}
REGIONAL = "Kaggle / gokulrajkmv  |  Region-wide estimates, Jan–Oct 2020"
AREA = "Kaggle / gokulrajkmv  |  Region × area estimates, May 2019–Jun 2020"


def setup():
    sns.set_theme(style="whitegrid", palette=[BLUE, TEAL, ORANGE], rc={
        "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": GRAY, "ytick.color": GRAY, "axes.edgecolor": "#C8D5DB",
        "grid.color": "#DBE4E8", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "legend.frameon": False, "font.size": 10,
    })


def title(fig, heading, subtitle, source):
    fig.text(.06, .955, heading, fontsize=17, weight="bold", va="top")
    fig.text(.06, .889, subtitle, fontsize=10, color=GRAY, va="top")
    fig.text(.06, .018, source, fontsize=8, color=GRAY, va="bottom")


def date_axis(ax, interval=1):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15, interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_xlabel("Observation month")
    ax.margins(x=.035)


def rate_axis(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Unemployment rate")
    ax.set_ylim(bottom=0)


def lockdown(ax):
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-01"), color=SHADE, zorder=0)
    ax.axvline(pd.Timestamp("2020-03-25"), color=ORANGE, linestyle=":", lw=1.3)


def charts(t, area, regional, folder: Path):
    setup()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(name, fig):
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        paths[name] = path

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.3), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=.09, right=.96, bottom=.16, top=.79, wspace=.30)
    data = regional["unemployment_rate"]
    sns.histplot(data, bins=np.arange(0, 85, 5), ax=axes[0], color=BLUE, edgecolor="white")
    axes[0].axvline(data.median(), color=ORANGE, linestyle="--", label=f"Median {data.median():.2f}%")
    axes[0].legend()
    axes[0].set(ylabel="Region-month observations", xlabel="Unemployment rate (%)", xlim=(0, 80))
    sns.boxplot(x=data, ax=axes[1], color=TEAL, width=.4, fliersize=4)
    axes[1].set(xlabel="Unemployment rate (%)", ylabel="", xlim=(0, 80), yticks=[])
    title(fig, "Unemployment estimates have a long right tail",
          f"{len(data)} region-month observations. Dots beyond the boxplot whiskers are retained.", REGIONAL)
    save("01_distribution", fig)

    hist = t["historical_monthly"]
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    fig.subplots_adjust(left=.095, right=.96, bottom=.16, top=.80)
    lockdown(ax)
    ax.plot(hist.month + pd.Timedelta(days=14), hist.unemployment_rate, color=BLUE, marker="o", lw=2, label="Monthly mean")
    ax.plot(hist.month + pd.Timedelta(days=14), hist.rolling_3m_rate, color=ORANGE, linestyle="--", lw=2, label="Trailing 3 months")
    ax.legend(loc="upper left", ncols=2)
    date_axis(ax, 2)
    rate_axis(ax)
    title(fig, "The historical panel shows a sharp break in spring 2020",
          f"Equal weight for both areas in each of {hist.regions.iloc[0]} complete regions. Shading marks Apr–May estimates.", AREA)
    save("02_historical_trend", fig)

    month = t["monthly_regional_panel"]
    periods = t["covid_periods"].set_index("period")
    baseline = periods.at["Baseline", "unemployment_rate"]
    peak = month.loc[month["unemployment_rate"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.7), gridspec_kw={"width_ratios": [2.25, 1]})
    fig.subplots_adjust(left=.075, right=.965, bottom=.18, top=.79, wspace=.29)
    lockdown(axes[0])
    axes[0].plot(month.month + pd.Timedelta(days=14), month.unemployment_rate, lw=2.5, color=BLUE, marker="o", label="Monthly regional mean")
    axes[0].axhline(baseline, color=TEAL, linestyle="--", label="Jan–Feb baseline")
    axes[0].annotate(f"{peak.unemployment_rate:.2f}%", (peak.month + pd.Timedelta(days=14), peak.unemployment_rate),
                     xytext=(10, 5), textcoords="offset points", weight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    date_axis(axes[0], 2)
    rate_axis(axes[0])
    picks = periods.loc[["Baseline", "Shock", "Reopening"], "unemployment_rate"]
    bars = axes[1].bar(["Jan–Feb", "Apr–May", "Jun–Oct"], picks, color=[BLUE, ORANGE, TEAL], width=.63)
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    rate_axis(axes[1])
    axes[1].set(xlabel="Comparison window", ylabel="")
    for ax in axes:
        ax.set_ylim(0, max(28, peak.unemployment_rate + 5))
    title(fig, "Unemployment surged in April and May, then receded",
          f"Unweighted means for {month.regions.iloc[0]} regions. March is a transition month. Dotted line marks 25 March.", REGIONAL)
    save("03_lockdown_and_reopening", fig)

    ranking = t["regional_summary"].sort_values("mean")
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.205, right=.95, bottom=.125, top=.815)
    colors = [TEAL if i < 5 else ORANGE if i >= len(ranking)-5 else "#AEC3CC" for i in range(len(ranking))]
    ax.barh(ranking.region, ranking["mean"], color=colors, height=.72)
    ax.set(xlabel="Mean unemployment rate (%)", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    title(fig, "Average unemployment varied widely across regions",
          "Observed-month means. The five lowest and highest are highlighted; coverage differs for two regions.", REGIONAL)
    save("04_regional_ranking", fig)

    changes = t["covid_by_region"].query("period == 'Shock'").sort_values("unemployment_change_pp")
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.205, right=.95, bottom=.125, top=.815)
    ax.barh(changes.region, changes.unemployment_change_pp,
            color=np.where(changes.unemployment_change_pp >= 0, BLUE, TEAL), height=.72)
    ax.axvline(0, color=INK, lw=1)
    ax.set(xlabel="Change from Jan–Feb to Apr–May (percentage points)", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    title(fig, "The lockdown-period increase was uneven",
          "Each bar compares two monthly averages for the same region. Regions with missing comparison months are omitted.", REGIONAL)
    save("05_regional_shock", fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.45))
    fig.subplots_adjust(left=.10, right=.96, bottom=.16, top=.80)
    lockdown(ax)
    by_area = t["monthly_area_panel"]
    for label, color, marker in [("Rural", TEAL, "s"), ("Urban", BLUE, "o")]:
        subset = by_area.loc[by_area.area.eq(label)]
        ax.plot(subset.month + pd.Timedelta(days=14), subset.unemployment_rate, label=label, lw=2.25, marker=marker, color=color)
    ax.legend(loc="upper left")
    rate_axis(ax)
    date_axis(ax)
    title(fig, "Unemployment rose in both rural and urban areas",
          f"Equal-weight means over the same {by_area.regions.iloc[0]} regions for each area, Jan–Jun 2020.", AREA)
    save("06_urban_rural", fig)

    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    fig.subplots_adjust(left=.105, right=.96, bottom=.26, top=.70)
    calendar = t["calendar_heatmap"].set_index("year")
    calendar.columns = [pd.Timestamp(2020, col, 1).strftime("%b") for col in calendar.columns]
    ax.set_facecolor("#E9EEF0")
    sns.heatmap(calendar, ax=ax, cmap="YlGnBu", vmin=0, vmax=30, annot=True, fmt=".1f",
                cbar_kws={"label": "Rate (%)"}, linewidths=.7, linecolor="white", annot_kws={"size": 9})
    ax.set(xlabel="Calendar month", ylabel="Year")
    ax.tick_params(axis="y", rotation=0)
    title(fig, "Only May and June can be compared across years",
          "Fixed region-by-area panel. Gray cells mark months outside the observation window.", AREA)
    save("07_calendar_heatmap", fig)

    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    fig.subplots_adjust(left=.20, right=.95, bottom=.13, top=.82)
    grid = t["regional_heatmap"].set_index("region").copy()
    grid.columns = [col.strftime("%b") for col in grid.columns]
    ax.set_facecolor("#DFE5E8")
    sns.heatmap(grid, ax=ax, cmap="YlGnBu", vmin=0, vmax=max(80, np.nanmax(grid)), yticklabels=True,
                linewidths=.45, linecolor="white", cbar_kws={"label": "Rate (%)"})
    ax.set(xlabel="Month in 2020", ylabel="")
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    title(fig, "The spring shock varied in timing and intensity",
          "Each cell shows an observed region-month estimate. Gray cells indicate unavailable observations.", REGIONAL)
    save("08_regional_heatmap", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.15))
    fig.subplots_adjust(left=.16, right=.93, bottom=.24, top=.77, wspace=.65)
    for ax, key, heading in zip(axes, ["regional_pearson", "regional_within_unit_pearson"],
                               ["Pooled region-months", "After removing region means"]):
        matrix = t[key].set_index("metric").rename(index=LABELS, columns=LABELS)
        sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                    cbar=ax is axes[1], cbar_kws={"label": "Pearson r"}, square=True,
                    linewidths=1, linecolor="white", annot_kws={"size": 11})
        ax.set(title=heading, xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    title(fig, "Labour-market correlations change within regions",
          "Centering each metric on its regional mean reduces differences due to region size and persistent levels.", REGIONAL)
    save("09_correlations", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3))
    fig.subplots_adjust(left=.09, right=.96, bottom=.17, top=.78, wspace=.34)
    for ax, metric, scale, label in [
        (axes[0], "participation_rate", 1, "Labour participation rate (%)"),
        (axes[1], "employed", 1e6, "Mean employed per region (millions)"),
    ]:
        lockdown(ax)
        ax.plot(month.month + pd.Timedelta(days=14), month[metric] / scale, color=BLUE, marker="o", lw=2.25, label="Monthly mean")
        ax.axhline(periods.at["Baseline", metric] / scale, linestyle="--", color=TEAL, label="Jan–Feb baseline")
        ax.set(ylabel=label, ylim=(0, max(month[metric]) / scale * 1.16))
        date_axis(ax, 3)
    axes[0].legend(loc="lower right", fontsize=8)
    title(fig, "Participation and employment lagged the unemployment rebound",
          f"Unweighted estimates for {month.regions.iloc[0]} matched regions. Employment is an average count per region.", REGIONAL)
    save("10_participation_employment", fig)
    return paths

