"""Regression checks against the official release and its generated deliverables."""

import hashlib
import shutil
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
import pytest

from main import analyze, prepare, run
from src.covid_analysis import classify_period
from src.data_cleaning import clean
from src.data_loader import FILES, load_raw
from src.statistical_analysis import paired_statistics

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def data():
    return prepare(RAW)


@pytest.fixture(scope="module")
def tables(data):
    return analyze(*data[:4])


def test_load_and_verify_original_files():
    raw, meta = load_raw(RAW)
    assert raw["area"].shape == (768, 7)
    assert raw["regional"].shape == (267, 9)
    assert sorted(meta["files"]) == sorted(FILES.values())
    for name, item in meta["files"].items():
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == item["sha256"]


def test_loader_rejects_changed_original(tmp_path):
    shutil.copytree(RAW, tmp_path / "raw")
    damaged = tmp_path / "raw" / FILES["regional"]
    damaged.write_bytes(damaged.read_bytes().replace(b"5.48", b"5.49", 1))
    with pytest.raises(ValueError, match="Integrity check failed"):
        load_raw(tmp_path / "raw")


def test_required_column_identification(data):
    raw = data[0]["area"]
    with pytest.raises(ValueError, match="unemployment_rate"):
        clean(raw.drop(columns=" Estimated Unemployment Rate (%)"), "area")


def test_date_parsing_and_coordinates(data):
    area, regional = data[1]["area"], data[1]["regional"]
    assert area.date.min() == pd.Timestamp("2019-05-31")
    assert area.date.max() == pd.Timestamp("2020-06-30")
    assert regional.date.max() == pd.Timestamp("2020-10-31")
    sample = regional.query("region == 'Andhra Pradesh'").iloc[0]
    assert (sample.latitude, sample.longitude) == (15.9129, 79.74)
    assert regional.frequency.eq("Monthly").all()


def test_missing_and_blank_handling(data):
    raw, cleaned, audit, *_ = data
    assert len(cleaned["area"]) == 740
    assert len(cleaned["regional"]) == 267
    assert audit.query("check == 'blank_rows_removed'")["count"].sum() == 28
    sample = raw["area"].iloc[:2].copy()
    sample.loc[sample.index[0], " Estimated Employed"] = np.nan
    result, _ = clean(sample, "area")
    assert len(result) == 2
    assert result["employed"].isna().sum() == 1


def test_exact_duplicate_and_conflicting_key(data):
    sample = data[0]["regional"].iloc[[0]].copy()
    duplicate = pd.concat([sample, sample], ignore_index=True)
    result, audit = clean(duplicate, "regional")
    assert len(result) == 1
    assert audit.query("check == 'exact_duplicates_removed'")["count"].item() == 1
    duplicate.loc[1, " Estimated Unemployment Rate (%)"] = 42.0
    with pytest.raises(ValueError, match="conflicting observations"):
        clean(duplicate, "regional")


def test_percentage_bounds_and_outliers(data):
    raw, frames, *_ = data
    sample = raw["area"].iloc[:3].copy()
    sample.loc[0, " Estimated Unemployment Rate (%)"] = 101
    sample.loc[1, " Estimated Labour Participation Rate (%)"] = -1
    sample.loc[2, " Estimated Employed"] = -10
    cleaned, _ = clean(sample, "area")
    for metric in ["unemployment_rate", "participation_rate", "employed"]:
        assert cleaned[metric].isna().sum() == 1
    area = frames["area"]
    assert area.loc[area.iqr_outlier, "unemployment_rate"].max() == 76.74
    assert area["iqr_outlier"].sum() == 35


def test_period_classification():
    dates = pd.Series(["2019-12-31", "2020-02-29", "2020-03-25", "2020-04-30",
                       "2020-05-31", "2020-06-01", "2021-01-01", None])
    assert classify_period(dates).fillna("missing").tolist() == [
        "Historical", "Baseline", "Transition", "Shock", "Shock", "Reopening",
        "Outside study", "missing",
    ]


def test_regional_aggregation_has_fixed_members_and_known_values(tables):
    panel = tables["regional_panel_observations"]
    assert panel.region.nunique() == 25
    assert panel.groupby("region").size().eq(10).all()
    periods = tables["covid_periods"].set_index("period")
    assert periods.at["Baseline", "unemployment_rate"] == pytest.approx(8.7634)
    assert periods.at["Shock", "unemployment_change_pp"] == pytest.approx(14.4408)
    assert periods.at["Pandemic", "unemployment_rate"] == pytest.approx(13.4308)
    assert tables["monthly_regional_panel"].iloc[-1].unemployment_rate == pytest.approx(7.9876)
    assert len(tables["source_overlap"]) == 158


def test_area_and_temporal_panels(tables):
    area = tables["area_panel_observations"]
    assert area.groupby(["region", "area"]).size().eq(6).all()
    assert area.region.nunique() == 20
    changes = tables["covid_by_area"].query("period == 'Shock'").set_index("area")
    assert changes.at["Urban", "unemployment_change_pp"] == pytest.approx(15.724)
    assert changes.at["Rural", "unemployment_change_pp"] == pytest.approx(14.31275)
    years = tables["same_month_year_comparison"]
    assert years.calendar_month.tolist() == [5, 6]


def test_paired_calculations():
    result = paired_statistics([1, 2, 3, 4], "example", seed=9, draws=1000)
    assert result["mean_pp"] == 2.5
    assert result["exact_sign_p"] == .125
    assert result["paired_cohens_dz"] == pytest.approx(2.5 / np.std([1, 2, 3, 4], ddof=1))
    assert result["bootstrap_95_low_pp"] <= 2.5 <= result["bootstrap_95_high_pp"]
    assert result == paired_statistics([1, 2, 3, 4], "example", seed=9, draws=1000)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    shutil.copytree(RAW, root / "data" / "raw")
    before = {p.name: p.read_bytes() for p in (root / "data" / "raw").iterdir()}
    assert run(root).is_file()
    after = {p.name: p.read_bytes() for p in (root / "data" / "raw").iterdir()}
    assert before == after
    return root


def test_output_tables_and_log(generated):
    assert len(pd.read_csv(generated / "data/processed/cleaned_unemployment_data.csv")) == 267
    assert len(pd.read_csv(generated / "data/processed/cleaned_unemployment_by_area.csv")) == 740
    assert (generated / "data/processed/data_quality_summary.csv").is_file()
    assert (generated / "outputs/tables/covid_periods.csv").is_file()
    assert "Completed with 10 charts" in (generated / "outputs/logs/analysis.log").read_text(encoding="utf-8")


def test_chart_creation(generated):
    figures = list((generated / "outputs/figures").glob("*.png"))
    assert len(figures) == 10
    for path in figures:
        with Image.open(path) as image:
            assert image.width >= 3000
            assert image.info["dpi"][0] == pytest.approx(300, abs=.1)
            image.verify()


def test_report_creation(generated):
    report = PdfReader(generated / "outputs/reports/Unemployment_Analysis_Report.pdf")
    assert len(report.pages) == 9
    text = "\n".join(page.extract_text() for page in report.pages)
    for snippet in ["23.20%", "14.44 percentage points", "Seasonality".lower(), "References"]:
        assert snippet in text


def test_executed_notebook():
    notebook = nbformat.read(ROOT / "notebooks/Unemployment_Analysis.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 10

