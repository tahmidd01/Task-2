# Unemployment in India
## The 2020 lockdown and labour-market rebound

This Python project examines monthly unemployment, employment and labour-force participation in India. It describes regional differences, compares the first full lockdown months with a fixed pre-lockdown baseline, tests whether urban and rural changes differed, and assesses the evidence for seasonal patterns.

### Data

The project downloaded both original CSVs on 9 November 2026 through the official Kaggle public API. The publisher is Gokulraj K M V. The [Kaggle dataset page](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india) is the primary source for attribution. The original file sizes, raw shapes and SHA-256 checksums are recorded in `data/raw/dataset_metadata.json`.

| Original file | Raw dimensions | Usable rows | Dates | Grain |
| --- | ---: | ---: | --- | --- |
| Unemployment in India.csv | 768 × 7 | 740 | May 2019–June 2020 | Region × month × urban/rural |
| Unemployment_Rate_upto_11_2020.csv | 267 × 9 | 267 | January–October 2020 | Region × month |

The area file covers 28 regions and the region-wide file covers 27. Cleaning removed 28 entirely empty records, normalized labels and dates, and corrected reversed coordinate headers in the regional file. The two files had no duplicate observation keys or missing values in the retained rows. All 45 unemployment observations flagged by the 1.5-IQR rule were retained. The coverage audits show absent panel cells separately from missing fields in existing rows.

There are 158 overlapping region-month keys, including 148 with both area records. The region-wide rate differs from the simple rural/urban average in 144 of those 148 cases at a tolerance of 0.011 percentage points. Labour-force weights are needed to aggregate area rates. The analysis keeps the two grains separate. See [the data dictionary](data/DATA_DICTIONARY.md) for fields, keys and definitions.

### Study design

- The region-wide comparison uses 25 regions observed in every month from January to October 2020. Jammu & Kashmir lacks April, and Sikkim lacks both baseline months. The same 25 qualify when completeness is restricted to January–May.
- The area comparison uses the same 20 regions for both urban and rural series from January to June. The historical trend also requires both areas in all 14 months, and the same 20 regions qualify.
- Baseline is January–February 2020. The nationwide lockdown began on 25 March, making March a transition observation. April–May are the first two full lockdown months. Reopening begins in June and runs through the last available observation in each source. Pandemic-period means pool April onward, with equal weight per available month of the fixed cohort. [Lockdown order announcement](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1607997&lang=2&reg=48) · [June reopening guidelines](https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=1627965&lang=2&reg=48)
- Equal-weight means describe the included regions. Percentage-point change is the comparison value minus baseline. Relative change divides that difference by baseline and multiplies by 100. Employment is a mean count per region or area.
- Inference collapses repeated monthly observations to a paired change per region. It uses a two-sided exact sign test, a 20,000-draw seeded percentile bootstrap, paired Cohen’s dz and leave-one-region-out means. Correlations are calculated on pooled and within-region data.

### Main findings

| Measure | Result |
| --- | --- |
| Baseline and initial shock | 8.76% in January–February and 23.20% in April–May, an increase of 14.44 percentage points or 164.8% relative |
| Peak and subsequent months | The monthly mean peaked at 23.37% in May. April–October averaged 13.43%, June–October averaged 9.52%, and October reached 7.99%. |
| Participation and employment | April–May participation fell by 7.22 percentage points and mean employment fell 27.5% per region. October remained 2.89 points and 2.7% below baseline, respectively. |
| Geographic differences | Haryana had the highest observed-month mean (27.48%) and Meghalaya the lowest (3.87%). Puducherry had the largest paired increase (+65.85 points) and the highest single region-wide estimate (75.85%, April). |
| Urban and rural changes | Urban +15.72 points and rural +14.31 points. The difference was +1.41 points, with a 95% region-bootstrap interval of −2.42 to +5.14 points. |
| Paired regional uncertainty | Unemployment increased in 23 of 25 regions. The exploratory bootstrap interval for the mean change was 8.92 to 21.09 points. |
| Seasonality | May and June are the only calendar months that repeat. They increased by 17.60 and 3.53 points year over year in the fixed area panel. The short span and shared disruption limit recurring-pattern inference. |

The report proposes piloting targeted regional employment support, evaluating area-specific assistance, and including participation and estimated employment in recovery dashboards. The release lacks population weights, worker-level detail and a comparison group. A shared national shock can correlate regional changes, and the short post-shock period supports a short-run assessment. Annual seasonality, seasonal decomposition and forecasting require longer coverage.

### Run on Windows

Python 3.11.9 was used for this run. Install the pinned dependencies into an isolated virtual environment. The pipeline uses Pandas, NumPy, SciPy, Matplotlib, Seaborn and ReportLab. Jupyter executes and displays the notebook, and pytest verifies the pipeline.

```powershell
Set-Location 'D:\Task 2'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --execute-notebook
.\.venv\Scripts\python.exe -m pytest -q
```

Use `main.py` without the optional flag to refresh cleaned data, tables, charts, report and logs. The existing validated originals are reused on every run. If the CSVs are missing and automatic download fails, download the ZIP from the dataset page and extract both original CSVs into `data/raw/`. The supplied public release downloaded without credentials.

To view or run the notebook interactively, launch Jupyter from the project root and select the notebook in the `notebooks` directory. The notebook also works in VS Code with its Jupyter extension and the project virtual environment selected.

```powershell
.\.venv\Scripts\python.exe -m notebook
```

### Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Original CSVs and acquisition manifest |
| `data/processed/` | Cleaned region-wide data, cleaned area data and cleaning audit |
| `data/DATA_DICTIONARY.md` | Raw and processed fields, keys and interpretation |
| `notebooks/Unemployment_Analysis.ipynb` | Executed narrative notebook with tables and figures |
| `outputs/reports/Unemployment_Analysis_Report.pdf` | Academic report |
| `outputs/figures/` | Ten publication figures at 300 DPI |
| `outputs/tables/` | Chart data, panel memberships, matched contrasts, correlations and audits |
| `outputs/logs/analysis.log` | Pipeline stages and run status |
| `src/`, `main.py`, `tests/` | Modular implementation, command-line entry point and regression tests |

| Figure | What to look for |
| --- | --- |
| 01 Distribution | Long right tail in the histogram and boxplot |
| 02 Historical trend | Monthly movement and the lagging three-month mean |
| 03 Lockdown and reopening | Baseline, May peak and decline after the first shock |
| 04 Regional ranking | Highest and lowest observed-month averages |
| 05 Regional shock | Variation in matched changes and regions with declines |
| 06 Urban/rural | April–May surge in both areas of the matched panel |
| 07 Calendar heatmap | Only two months available for comparison across years |
| 08 Regional heatmap | Geographic concentration and gaps in coverage |
| 09 Correlations | Differences between pooled and within-region associations |
| 10 Participation and employment | Continued shortfalls after unemployment approached baseline |

