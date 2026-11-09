"""Assemble the academic report from validated analysis tables."""

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .data_loader import SOURCE

LOCKDOWN = "https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1607997&lang=2&reg=48"
REOPENING = "https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=1627965&lang=2&reg=48"
INK = "#172A38"
BLUE = "#155575"
TEAL = "#087F83"
GRAY = "#455965"


def findings(t):
    periods = t["covid_periods"].set_index("period")
    base, shock, recovery, pandemic = (periods.loc[key] for key in ["Baseline", "Shock", "Reopening", "Pandemic"])
    months = t["monthly_regional_panel"]
    peak = months.loc[months.unemployment_rate.idxmax()]
    latest = months.iloc[-1]
    regions = t["covid_by_region"].query("period == 'Shock'").sort_values("unemployment_change_pp", ascending=False)
    largest = regions.iloc[0]
    largest_three = ", ".join(f"{row.region} ({row.unemployment_change_pp:+.2f} pp)" for row in regions.head(3).itertuples())
    rank = t["regional_summary"]
    variable = rank.loc[rank.sd.idxmax()]
    by_area = t["covid_by_area"].query("period == 'Shock'").set_index("area")
    test, gap = (row for row in t["paired_inference"].itertuples())
    maximum = t["largest_observations"].query("source == 'regional'").iloc[0]
    area_max = t["largest_observations"].query("source == 'area'").iloc[0]
    year_match = t["same_month_year_comparison"].set_index("calendar_month")
    ds = t["descriptive_statistics"].query("source == 'regional'").set_index("metric")
    rate = ds.loc["unemployment_rate"]
    rho = t["regional_pearson"].set_index("metric").loc["unemployment_rate"]
    centered = t["regional_within_unit_pearson"].set_index("metric").loc["unemployment_rate"]
    overlap = t["source_overlap"]
    both = overlap.loc[overlap.area_count.eq(2)]
    mismatches = both.rate_gap_pp.abs().gt(.011).sum()
    largest_jump = t["historical_monthly"].change_pp.max()
    return {
        "shock": f"The unweighted unemployment mean across {int(peak.regions)} matched regions rose from "
        f"{base.unemployment_rate:.2f}% in January-February to {shock.unemployment_rate:.2f}% in April-May "
        f"({shock.unemployment_change_pp:+.2f} percentage points, {shock.unemployment_relative_pct:+.1f}% relative). "
        f"The monthly mean peaked in {peak.month:%B} at {peak.unemployment_rate:.2f}%. Across April-October, "
        f"the mean was {pandemic.unemployment_rate:.2f}% ({pandemic.unemployment_change_pp:+.2f} pp; "
        f"{pandemic.unemployment_relative_pct:+.1f}% relative to baseline).",
        "recovery": f"The June-October mean fell to {recovery.unemployment_rate:.2f}% "
        f"({recovery.unemployment_change_pp:+.2f} pp from baseline), reaching {latest.unemployment_rate:.2f}% "
        f"in {latest.month:%B}. In that final month participation remained "
        f"{base.participation_rate - latest.participation_rate:.2f} pp below baseline and mean estimated employment "
        f"was {100 * (1 - latest.employed / base.employed):.1f}% lower.",
        "labour": f"During April-May, mean participation fell from {base.participation_rate:.2f}% to "
        f"{shock.participation_rate:.2f}% ({shock.participation_change_pp:+.2f} pp). Mean estimated employment "
        f"fell from {base.employed / 1e6:.2f} million to {shock.employed / 1e6:.2f} million per region "
        f"({shock.employment_relative_pct:+.1f}%).",
        "distribution": f"Across all {int(rate['count'])} region-month estimates, unemployment averaged {rate['mean']:.2f}% "
        f"with a median of {rate['50%']:.2f}% and SD of {rate['std']:.2f} pp. Values ranged from "
        f"{rate['min']:.2f}% to {rate['max']:.2f}%, with the middle half between "
        f"{rate['25%']:.2f}% and {rate['75%']:.2f}% (IQR {rate['iqr']:.2f} pp).",
        "region": f"{rank.iloc[0].region} had the highest observed-month mean ({rank.iloc[0]['mean']:.2f}%), "
        f"{rank.iloc[-1].region} the lowest ({rank.iloc[-1]['mean']:.2f}%), and {variable.region} the largest "
        f"within-region SD ({variable.sd:.2f} pp). The largest paired increases were {largest_three}. "
        f"The highest individual estimates were {maximum.unemployment_rate:.2f}% in {maximum.region} "
        f"({maximum.month:%B %Y}, region-wide) and {area_max.unemployment_rate:.2f}% in "
        f"{area_max.region} ({area_max.month:%B %Y}, {area_max.area.lower()}).",
        "area": f"Across {gap.regions} regions with both areas, urban unemployment rose from "
        f"{by_area.at['Urban', 'unemployment_rate_baseline']:.2f}% to "
        f"{by_area.at['Urban', 'unemployment_rate']:.2f}% ({by_area.at['Urban', 'unemployment_change_pp']:+.2f} pp); "
        f"rural unemployment rose from {by_area.at['Rural', 'unemployment_rate_baseline']:.2f}% to "
        f"{by_area.at['Rural', 'unemployment_rate']:.2f}% ({by_area.at['Rural', 'unemployment_change_pp']:+.2f} pp). "
        f"The estimated urban-minus-rural change was {gap.mean_pp:+.2f} pp "
        f"(95% region bootstrap interval {gap.bootstrap_95_low_pp:.2f} to {gap.bootstrap_95_high_pp:.2f} pp). "
        f"The interval spans zero, leaving the difference uncertain.",
        "inference": f"Unemployment rose in {test.positive} of {test.regions} regions (median change {test.median_pp:+.2f} pp). "
        f"The bootstrap 95% interval for the mean change was {test.bootstrap_95_low_pp:.2f} to "
        f"{test.bootstrap_95_high_pp:.2f} pp. The two-sided sign test gave p = {test.exact_sign_p:.2g}; "
        f"paired Cohen's dz was {test.paired_cohens_dz:.2f}. Shapiro-Wilk p = {test.shapiro_p:.2g} suggests "
        f"asymmetric or non-normal differences. Leaving out one region at a time put the mean change "
        f"between {test.leave_one_out_min_pp:.2f} and {test.leave_one_out_max_pp:.2f} pp.",
        "correlation": f"The pooled Pearson correlations of unemployment with estimated employment "
        f"and participation were {rho.employed:.2f} and {rho.participation_rate:.2f}. After removing "
        f"each region's average, they were {centered.employed:.2f} and {centered.participation_rate:.2f}. "
        "The within-region associations can still reflect common shocks and linked labour-market definitions.",
        "seasonality": f"Only May and June repeat across the {len(t['historical_monthly'])}-month area series. "
        f"The matched year-over-year differences were {year_match.at[5, 'change_pp']:+.2f} pp in May and "
        f"{year_match.at[6, 'change_pp']:+.2f} pp in June. The largest month-to-month increase was "
        f"{largest_jump:+.2f} pp in April 2020. With fewer than two annual cycles and pandemic disruption "
        "in one of the years, these comparisons are exploratory. STL and forecasting were omitted.",
        "overlap": f"The files overlap in {len(overlap)} region-month keys, with both areas present in {len(both)}. "
        f"In {mismatches} of those, the regional rate differs from the simple area mean by more than 0.011 pp. "
        "Area percentages require labour-force weights to aggregate. The two files serve separate analytical roles.",
        "policy_region": f"Pilot rapid-response placement and payroll support where unemployment increased most, "
        f"beginning with {largest.region} and other severely affected regions. Validate local indicators "
        "and employer demand before allocating support, then track re-employment and retention.",
        "policy_area": "Test job-matching and temporary income support for urban workers alongside rural public works "
        "for households facing a sudden loss of work. Both area means rose sharply, and the evidence leaves the "
        "urban-rural difference uncertain. Track access and outcomes separately by area.",
        "policy_monitoring": "Include participation and employment alongside unemployment in recovery dashboards. "
        "October unemployment had fallen below baseline while the other two measures remained lower. "
        "Use survey weights and worker-level follow-up to identify groups that stopped looking for work.",
    }


def make_report(path, tables, figures, metadata):
    t, f = tables, findings(tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=50, leftMargin=50,
                            topMargin=57, bottomMargin=50,
                            title="Unemployment in India | The 2020 lockdown and labour-market rebound",
                            author="Task 2 | Unemployment Analysis with Python")
    story = []
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.3, leading=14.5,
                               textColor=INK, spaceAfter=8, alignment=TA_LEFT),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=27, leading=32,
                                  textColor=INK, spaceAfter=14),
        "sub": ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=12, leading=15,
                              textColor=BLUE, spaceBefore=7, spaceAfter=5),
        "eyebrow": ParagraphStyle("eyebrow", fontName="Helvetica-Bold", fontSize=9, leading=12,
                                  textColor=TEAL, spaceAfter=8),
        "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=8.3, leading=11,
                                  textColor=GRAY, spaceAfter=9),
        "cover": ParagraphStyle("cover", fontName="Helvetica-Bold", fontSize=43, leading=46,
                                textColor=INK, spaceAfter=16),
    }

    def paragraph(message, kind="body"):
        story.append(Paragraph(message, styles[kind]))

    def plain(message, kind="body"):
        paragraph(escape(message), kind)

    def section(index, heading):
        story.append(PageBreak())
        plain(f"RESEARCH BRIEF   /   {index}", "eyebrow")
        plain(heading, "heading")

    def image(key, number, explanation, width=doc.width):
        file = str(figures[key])
        w, h = ImageReader(file).getSize()
        story.append(Image(file, width=width, height=width * h / w, hAlign="CENTER"))
        paragraph(f"<b>Figure {number}.</b> {escape(explanation)}", "caption")

    def table(rows, widths):
        grid = Table(rows, colWidths=widths, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EAF4F5"), colors.white]),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK)),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([grid, Spacer(1, 10)])

    story.append(Spacer(1, 115))
    plain("TASK 2   /   DATA SCIENCE RESEARCH BRIEF", "eyebrow")
    paragraph("Unemployment<br/>in India", "cover")
    plain("The 2020 lockdown and labour-market rebound", "sub")
    story.append(Spacer(1, 20))
    plain("Monthly evidence from regional and urban/rural labour-market estimates.")
    plain("May 2019 to October 2020 across two separate source series.", "caption")
    story.append(Spacer(1, 25))
    pp = t["covid_periods"].set_index("period")
    table([["FIXED REGIONAL PANEL", "JAN-FEB", "APR-MAY", "JUN-OCT"],
           ["Mean unemployment"] + [f"{pp.at[p, 'unemployment_rate']:.2f}%"
                                    for p in ["Baseline", "Shock", "Reopening"]]],
          [208, 85, 85, 85])
    plain("Each figure is an unweighted regional mean.", "caption")

    section("01", "Executive summary")
    for key in ["shock", "recovery", "region", "area"]:
        plain(f[key])
    plain("Contents", "sub")
    contents = ["Executive summary", "Sources and study design", "Distribution and regional differences",
                "Lockdown and short-term recovery", "Regional and area contrasts",
                "Inference and associations", "Time patterns and seasonality", "Policy implications and references"]
    table([["SECTION", "PAGE"]] + [[name, str(page)] for page, name in enumerate(contents, 2)], [405, 58])

    section("02", "Sources and study design")
    plain("The study measures unemployment variation, quantifies changes around the lockdown, "
          "examines local and urban/rural differences, and evaluates the evidence for recurring monthly patterns.")
    plain("Acquisition and coverage", "sub")
    inventories = t["source_inventory"]
    table([["SOURCE GRAIN", "RAW ROWS", "VALID ROWS", "REGIONS", "OBSERVED MONTHS"]] + [
        ["Region x area" if row.source == "area" else "Region-wide", str(row.raw_rows), str(row.cleaned_rows),
         str(row.regions), f"{row.first_month:%b %Y} to {row.last_month:%b %Y}"]
        for row in inventories.itertuples()], [105, 68, 70, 58, 162])
    plain(f"Source files were retrieved via the official Kaggle ZIP endpoint on "
          f"{metadata.get('downloaded_at_utc', 'date unavailable')[:10]} (UTC date). "
          "The manifest records file sizes and SHA-256 hashes. The region-wide series ends in October 2020.")
    plain("Quality and comparability", "sub")
    audit = t["data_quality"]
    blank = audit.query("check == 'blank_rows_removed'")["count"].sum()
    flags = audit.query("check == 'iqr_flags_retained'")["count"].sum()
    plain(f"Removed {blank} fully blank records. The valid rows had no missing fields, duplicate observation "
          f"keys or out-of-bounds labour values. All {flags} IQR flags were retained. "
          "Dates were parsed day-first, frequencies standardized and the transposed coordinate headers corrected.")
    plain(f["overlap"])
    plain(f"Coverage audits show {len(t['missing_region_month_areas'])} absent region-month-area combinations "
          f"and {len(t['missing_regional_months'])} absent region-month combinations. Missing cells were left unfilled.")
    plain("Period and aggregation rules", "sub")
    plain("January-February defines baseline. March is a transition month because the nationwide lockdown took effect "
          "on 25 March [2]. April-May captures the first full lockdown months. Reopening starts in June [3] and ends "
          "at the last available month (October for region-wide data, June for area data). The pandemic average pools "
          "observed April-onward months equally.")
    plain("Comparisons fix the set of regions and weight regions and months equally. The area comparison uses the "
          "same regions in both groups. Unemployment is unemployed persons as a share of the labour force. "
          "Percentage-point change is current minus baseline. Relative change divides that difference by baseline "
          "and multiplies by 100. Employment figures are mean counts per region or area.")

    section("03", "Distribution and geography")
    plain(f["distribution"])
    image("01_distribution", 1, "Mean unemployment exceeds the median because a small share of observations are very high.", width=435)
    plain(f["region"])
    image("04_regional_ranking", 2, "Ranks use every available month in each region. Two regions have incomplete coverage.", width=390)

    section("04", "Lockdown and recovery")
    plain(f["shock"])
    image("03_lockdown_and_reopening", 3, "The fixed-panel peak was in May, followed by a steep decline in June.")
    plain(f["recovery"])
    plain(f["labour"])
    image("10_participation_employment", 4, "The two companion measures remained below their January-February averages in October.")

    section("05", "Regional and area contrasts")
    image("05_regional_shock", 5, "Within-region changes span declines and increases above 40 percentage points.", width=411)
    plain(f["area"])
    image("06_urban_rural", 6, "Both areas showed a sharp April-May increase and a lower June reading.", width=440)

    section("06", "Inference and associations")
    plain(f["inference"])
    plain("The resampling unit is the region, using 20,000 draws of paired changes. The sign test assesses balance "
          "of positive and negative changes, avoiding a normality assumption. The national shock can correlate regional "
          "responses, so both inference summaries are exploratory and conditional on the observed regions.")
    sensitivity = t["covid_sensitivity"].query("period == 'Shock'")
    plain("Restricting completeness to January-May gives shock changes of "
          f"{sensitivity.iloc[0].unemployment_change_pp:+.2f} pp versus "
          f"{sensitivity.iloc[1].unemployment_change_pp:+.2f} pp in the full-window panel.")
    plain(f["correlation"])
    image("09_correlations", 7, "Regional demeaning changes the magnitude of the employment and participation relationships.")
    plain("Interpretation limits", "sub")
    plain("The regional panel observes the same broad shock at the same time. The short pre-period, lack of a "
          "comparison group and unknown sampling weights limit interpretation. These results describe associations "
          "with the lockdown period. They cannot isolate effects of lockdown, health risk, policy responses, "
          "concurrent changes or changes in survey conditions.")

    section("07", "Time patterns and seasonality")
    plain(f["seasonality"])
    image("02_historical_trend", 8, "The trailing three-month mean smooths the spike and carries it into June.")
    image("07_calendar_heatmap", 9, "Unobserved months remain blank. The highest calendar averages are driven by 2020.")
    yr = t["historical_years"]
    plain("Partial-year averages were " + " and ".join(
        f"{row.mean:.2f}% across {row.observed_months} months in {row.year}" for row in yr.itertuples()
    ) + ". The two averages cover different calendar months.")

    section("08", "Policy implications")
    for label, key in [("Target regional response", "policy_region"), ("Adapt support by area", "policy_area"),
                       ("Track who returns to work", "policy_monitoring")]:
        plain(label, "sub")
        plain(f[key])
    plain("Conclusion", "sub")
    plain("Unemployment rose sharply in the first full lockdown months and fell during phased reopening. "
          "Large local differences and continued shortfalls in participation and employment make the unemployment "
          "rate alone an incomplete measure of recovery. The panel ends in 2020 and cannot show durable recovery.")
    plain("References", "sub")
    references = [
        (SOURCE, "[1] Gokulraj K M V. Unemployment in India. Kaggle dataset release."),
        (LOCKDOWN, "[2] Government of India, Press Information Bureau. Nationwide lockdown order announcement. 24 March 2020."),
        (REOPENING, "[3] Government of India, Press Information Bureau. Guidelines effective 1 June 2020. 30 May 2020."),
    ]
    for url, label in references:
        paragraph(f'<link href="{escape(url)}" color="{BLUE}">{escape(label)}</link>', "caption")

    def page(canvas, current):
        w, h = current.pagesize
        canvas.setStrokeColor(colors.HexColor("#C9D9DF"))
        canvas.line(50, 42, w - 50, 42)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 28, "UNEMPLOYMENT IN INDIA   /   TASK 2")
        canvas.drawRightString(w - 50, 28, f"{current.page:02d}")
        canvas.setFillColor(colors.HexColor(TEAL))
        canvas.rect(50, h - 42, 29, 3, fill=1, stroke=0)

    doc.build(story, onFirstPage=page, onLaterPages=page)

