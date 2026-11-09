"""Clean each source at its native grain and retain an explicit audit."""

import numpy as np
import pandas as pd

FIELDS = {
    "Region": "region", "Region.1": "macro_region", "Date": "date",
    "Frequency": "frequency", "Estimated Unemployment Rate (%)": "unemployment_rate",
    "Estimated Employed": "employed", "Estimated Labour Participation Rate (%)": "participation_rate",
    "Area": "area",
}
METRICS = ["unemployment_rate", "employed", "participation_rate"]
EXTRAS = {"area": ["area"], "regional": ["macro_region", "latitude", "longitude"]}


def normalize_columns(raw):
    return raw.rename(columns=lambda c: FIELDS.get(c.strip(), c.strip().lower()))


def clean(raw, grain):
    frame = normalize_columns(raw).replace(r"^\s*$", np.nan, regex=True).copy()
    # Accept both the raw header order and a pandas-mangled second Region header.
    required = {"region", "date", "frequency", *METRICS, *EXTRAS[grain]}
    missing = required.difference(frame.columns)
    if missing or frame.columns.has_duplicates:
        raise ValueError(f"{grain}: missing {sorted(missing)} or ambiguous columns {list(frame.columns)}")
    blanks = frame.isna().all(axis=1)
    exact = frame.duplicated() & ~blanks
    audit = [
        ("input_rows", "all", len(frame), "Count includes empty CSV records"),
        ("blank_rows_removed", "all", int(blanks.sum()), "All fields empty"),
        ("exact_duplicates_removed", "all", int(exact.sum()), "Nonblank exact duplicates"),
    ]
    for column, count in frame.isna().sum().items():
        audit.append(("raw_missing", column, count, "Includes blank records"))
    frame = frame.loc[~(blanks | exact)].copy()
    for column, count in frame.isna().sum().items():
        audit.append(("nonblank_missing", column, count, "Excludes blank and duplicate records"))

    for field in ["region", "frequency", *[c for c in EXTRAS[grain] if c not in {"latitude", "longitude"}]]:
        frame[field] = frame[field].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    frame["frequency"] = frame["frequency"].str.casefold().map({"m": "Monthly", "monthly": "Monthly"})
    frame["date"] = pd.to_datetime(frame["date"].astype("string").str.strip(), format="%d-%m-%Y", errors="coerce")
    if grain == "area":
        frame["area"] = frame["area"].str.title().where(lambda s: s.isin(["Rural", "Urban"]))
    else:
        # Source headings are reversed: e.g. Andhra Pradesh is (15.9129 N, 79.7400 E).
        frame = frame.rename(columns={"longitude": "latitude", "latitude": "longitude"})
        audit.append(("coordinate_headers_swapped", "latitude,longitude", len(frame),
                      "Source longitude contains northing in degrees; source latitude contains easting"))

    bounds = {"unemployment_rate": (0, 100), "employed": (0, np.inf),
              "participation_rate": (0, 100), "latitude": (-90, 90), "longitude": (-180, 180)}
    for column in bounds.keys() & set(frame.columns):
        numbers = pd.to_numeric(frame[column], errors="coerce")
        valid = np.isfinite(numbers) & numbers.between(*bounds[column])
        audit.append(("invalid_to_missing", column, int((frame[column].notna() & ~valid).sum()),
                      "Retain the row; omit missing metric from applicable calculations"))
        frame[column] = numbers.where(valid)

    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    key = ["region", "month"] + (["area"] if grain == "area" else [])
    key_valid = frame[key + ["frequency"]].notna().all(axis=1) & frame["region"].str.contains(r"[A-Za-z]", na=False)
    audit.append(("invalid_key_rows_removed", "all", int((~key_valid).sum()),
                  "Requires valid region, monthly date, frequency and, where applicable, area"))
    frame = frame.loc[key_valid].copy()
    if frame.duplicated(key).any():
        raise ValueError(f"{grain}: conflicting observations for {key}; inspect before aggregating")

    q1, q3 = frame["unemployment_rate"].quantile([.25, .75])
    spread = q3 - q1
    values = frame["unemployment_rate"]
    frame["iqr_outlier"] = (values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)
    audit.extend([
        ("conflicting_keys", "all", 0, "Native observation key is unique"),
        ("iqr_flags_retained", "unemployment_rate", int(frame["iqr_outlier"].sum()),
         f"1.5 IQR fences {q1 - 1.5 * spread:.2f} to {q3 + 1.5 * spread:.2f}; retained"),
        ("output_rows", "all", len(frame), "No imputation"),
    ])
    summary = pd.DataFrame(audit, columns=["check", "field", "count", "decision"])
    summary.insert(0, "source", grain)
    return frame.sort_values(key).reset_index(drop=True), summary


def overlap(area, regional):
    """Diagnose overlap only. The naive area mean is not a state unemployment estimate."""
    keys = ["region", "month"]
    components = area.groupby(keys).agg(
        area_count=("area", "size"), naive_area_rate=("unemployment_rate", "mean"),
        sum_area_employed=("employed", lambda s: s.sum(min_count=2)),
    ).reset_index()
    matched = regional[keys + METRICS].merge(components, on=keys, validate="one_to_one")
    complete = matched["area_count"].eq(2)
    matched["naive_area_rate"] = matched["naive_area_rate"].where(complete)
    matched["rate_gap_pp"] = matched["unemployment_rate"] - matched["naive_area_rate"]
    matched["employment_gap"] = matched["employed"] - matched["sum_area_employed"]
    return matched


def missingness_by_key(raw, grain):
    frame = normalize_columns(raw).replace(r"^\s*$", np.nan, regex=True)
    frame = frame.loc[frame.notna().any(axis=1)]
    keys = ["region", "date"] + (["area"] if grain == "area" else [])
    fields = [column for column in frame.columns if column not in keys]
    labels = frame[keys].fillna("(missing)").apply(lambda s: s.astype(str).str.strip())
    counts = frame[fields].isna().astype(int).join(labels).groupby(keys).sum()
    return counts.loc[counts.any(axis=1)].reset_index()

