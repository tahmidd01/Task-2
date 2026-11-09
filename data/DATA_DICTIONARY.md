# Data dictionary

The original Kaggle CSVs are preserved in `raw/`. Each processed CSV retains its source grain.

| Grain | Native key | Observed coverage |
| --- | --- | --- |
| Region by area | region, month, area | 28 regions, May 2019–June 2020 |
| Region-wide | region, month | 27 regions, January–October 2020 |

| Original field | Cleaned field | Definition and units |
| --- | --- | --- |
| Region (first occurrence) | `region` | State or Union Territory label as recorded in the source. Historical labels are preserved. |
| Date | `date` | Source observation date. Parsed with day-month-year order and exported as ISO date. |
| Derived from Date | `month` | First calendar day of the observation month, stored as an ISO date. Used as the monthly key. |
| Frequency | `frequency` | Observation cadence. Monthly and M are normalized to Monthly. |
| Estimated Unemployment Rate (%) | `unemployment_rate` | Estimated unemployed people as a percentage of the labour force, on a 0–100 scale. |
| Estimated Employed | `employed` | Estimated count of employed people in the observed region or area. |
| Estimated Labour Participation Rate (%) | `participation_rate` | Estimated labour force as a percentage of the relevant reference population. The release does not specify the population universe or underlying survey weights. |
| Area | `area` | Observed Rural or Urban category. Available only in the area file. |
| Region (second occurrence, Region.1 in pandas) | `macro_region` | Regional grouping such as North, South or Northeast. Available only in the region-wide file. |
| longitude | `latitude` | Source labels are transposed. This column has values about 10.85–33.78 degrees north and is geographic latitude. Available only in the region-wide file. |
| latitude | `longitude` | Source labels are transposed. This column has values about 71.19–92.94 degrees east and is geographic longitude. Available only in the region-wide file. The release gives no coordinate reference-point definition. |
| Derived from unemployment rate | `iqr_outlier` | True if outside Q1 minus 1.5 × IQR or Q3 plus 1.5 × IQR, computed separately per file. Flags retain the observation. |
| Derived from month (analysis tables) | `period` | Baseline, Transition, Shock, Reopening, or pooled Pandemic as defined in the study design. Pandemic overlaps Shock and Reopening. |
| Derived from baseline and comparison (analysis tables) | `unemployment_change_pp`, `participation_change_pp` | Comparison minus baseline in percentage points. |
| Derived from baseline and comparison (analysis tables) | `unemployment_relative_pct`, `employment_relative_pct` | Change divided by baseline times 100, expressed as a percent. Undefined at a zero baseline. |
| Derived from baseline and comparison (analysis tables) | `employment_change_people` | Difference in mean estimated employed counts. |

The cleaning audit gives the source, validation check, field, count and decision. Raw missing percentages use all raw records as the denominator. Coverage audits list absent combinations separately. A blank in an analytical table represents an unavailable estimate or a derived statistic that cannot be computed for the observation. The cross-source overlap table is a comparability diagnostic. Its simple area mean is unsuitable as a regional unemployment estimate because the area labour-force denominators are unavailable.

