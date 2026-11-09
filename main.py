"""Run the complete analysis from the project root or any working directory."""

import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd

from src.covid_analysis import covid_tables
from src.data_cleaning import clean, missingness_by_key, overlap
from src.data_loader import FILES, load_raw
from src.exploratory_analysis import exploratory_tables
from src.report_generator import make_report
from src.seasonal_analysis import temporal_tables
from src.statistical_analysis import statistics_tables
from src.visualization import charts


def prepare(raw_folder):
    raw, metadata = load_raw(raw_folder)
    cleaned = {}
    audits = []
    inventory = []
    for grain, frame in raw.items():
        cleaned[grain], audit = clean(frame, grain)
        audits.append(audit)
        usable = cleaned[grain]
        inventory.append({
            "source": grain, "filename": FILES[grain], "raw_rows": len(frame),
            "raw_columns": len(frame.columns), "cleaned_rows": len(usable),
            "regions": usable["region"].nunique(), "first_month": usable["month"].min(),
            "last_month": usable["month"].max(),
        })
    quality = pd.concat(audits, ignore_index=True)
    input_rows = quality.source.map({grain: len(frame) for grain, frame in raw.items()})
    quality["pct_of_raw_rows"] = (100 * quality["count"] / input_rows).where(quality.check.eq("raw_missing"))
    return raw, cleaned, quality, pd.DataFrame(inventory), metadata


def analyze(raw, cleaned, quality, inventory):
    area, regional = cleaned["area"], cleaned["regional"]
    logging.info("Auditing source overlap and missingness; computing EDA")
    tables = exploratory_tables(area, regional)
    bins = np.arange(0, 105, 5)
    counts, edges = np.histogram(regional["unemployment_rate"].dropna(), bins)
    tables.update({
        "source_inventory": inventory, "data_quality": quality,
        "source_overlap": overlap(area, regional),
        "regional_histogram": pd.DataFrame({"lower_pct": edges[:-1], "upper_pct": edges[1:], "observations": counts}),
        **{f"nonblank_missing_{grain}": missingness_by_key(frame, grain) for grain, frame in raw.items()},
    })
    logging.info("Constructing matched COVID-19 and urban/rural comparisons")
    covid = covid_tables(area, regional)
    tables.update(covid)
    logging.info("Measuring time trends, year overlap, coverage and correlations")
    tables.update(temporal_tables(area, regional))
    tables.update(statistics_tables(regional, area, covid))
    return tables


def run(root=ROOT):
    log_folder = root / "outputs" / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_folder / "analysis.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logging.info("Verifying original Kaggle CSVs and cleaning by observation grain")
        raw, cleaned, quality, inventory, metadata = prepare(root / "data" / "raw")
        processed = root / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        cleaned["regional"].to_csv(processed / "cleaned_unemployment_data.csv", index=False)
        cleaned["area"].to_csv(processed / "cleaned_unemployment_by_area.csv", index=False)
        quality.to_csv(processed / "data_quality_summary.csv", index=False)
        tables = analyze(raw, cleaned, quality, inventory)
        logging.info("Exporting %s analysis tables", len(tables))
        table_folder = root / "outputs" / "tables"
        table_folder.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(table_folder / f"{name}.csv", index=False)
        logging.info("Rendering 300-DPI figures")
        figures = charts(tables, cleaned["area"], cleaned["regional"], root / "outputs" / "figures")
        report = root / "outputs" / "reports" / "Unemployment_Analysis_Report.pdf"
        logging.info("Building the academic report")
        make_report(report, tables, figures, metadata)
        logging.info("Completed with %s charts", len(figures))
        return report
    except Exception:
        logging.exception("Analysis failed")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebook", action="store_true", help="Also execute and save the project notebook")
    args = parser.parse_args()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.getLogger().addHandler(console)
    print(f"Report saved to {run()}", flush=True)
    if args.execute_notebook:
        from src.notebook_runner import execute_notebook

        execute_notebook(ROOT)

