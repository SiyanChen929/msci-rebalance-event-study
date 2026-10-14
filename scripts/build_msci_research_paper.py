#!/usr/bin/env python3
"""Build the journal-style MSCI closing-window research paper."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from msci_rebalance.backtest import active_trade_metrics
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PAPER_TITLE = "The 15:50 Burst: Minute-Level Price Pressure around MSCI U.S. Index Reconstitutions"
PAPER_SUBTITLE = "Point-in-Time Event Construction, Review-Date Inference, and an Exploratory Top-10 Portfolio"
PRIMARY_COST_BPS = 0.0
SENSITIVITY_COST_BPS = 10.0


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def signed_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f}%"


def basis_points(value: float, digits: int = 2) -> str:
    return f"{10_000.0 * float(value):+.{digits}f}"


def number(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def cell(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def table_from_rows(
    rows: list[list[object]],
    widths: list[float],
    body_style: ParagraphStyle,
    header_style: ParagraphStyle,
    font_size: float = 8.2,
    repeat_rows: int = 1,
) -> Table:
    prepared: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else body_style
        prepared.append([cell(value, style) for value in row])
    result = Table(prepared, colWidths=widths, repeatRows=repeat_rows, hAlign="CENTER")
    result.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 1.6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
            ]
        )
    )
    return result


def equation(
    expression: str,
    equation_number: int,
    equation_style: ParagraphStyle,
    number_style: ParagraphStyle,
    width: float,
) -> Table:
    content = Table(
        [
            [
                "",
                Paragraph(expression, equation_style),
                Paragraph(f"({equation_number})", number_style),
            ]
        ],
        colWidths=[0.42 * inch, width - 0.84 * inch, 0.42 * inch],
        hAlign="CENTER",
    )
    content.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return content


def figure(
    image_path: Path,
    caption: str,
    figure_number: int,
    caption_style: ParagraphStyle,
    max_width: float,
    max_height: float,
) -> KeepTogether:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = Image(str(image_path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether(
        [
            image,
            Spacer(1, 0.08 * inch),
            Paragraph(f"<b>Figure {figure_number}:</b> {caption}", caption_style),
        ]
    )


def table_caption(number_: int, text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"<b>Table {number_}:</b> {text}", style)


def add_paragraphs(story: list, style: ParagraphStyle, paragraphs: list[str]) -> None:
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, style))
        story.append(Spacer(1, 0.075 * inch))


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=styles["Title"],
            fontName="Times-Roman",
            fontSize=18.5,
            leading=20.5,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "PaperSubtitle",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=12.3,
            leading=14.0,
            alignment=TA_CENTER,
            spaceAfter=15,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=11.3,
            leading=13,
            alignment=TA_CENTER,
            spaceAfter=9,
        ),
        "abstract_heading": ParagraphStyle(
            "AbstractHeading",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=11.2,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=7,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9.6,
            leading=11.5,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=11.4,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=11,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=14.2,
            leading=16,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=11.8,
            leading=13.5,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.25,
            leading=13.0,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            widows=2,
            orphans=2,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=9.8,
            leading=12.2,
            leftIndent=16,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=4,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=10.4,
            leading=12.5,
            alignment=TA_CENTER,
        ),
        "equation_number": ParagraphStyle(
            "EquationNumber",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9.7,
            leading=12,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=8.8,
            leading=10.7,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9.3,
            leading=11.2,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=5,
        ),
        "table_body": ParagraphStyle(
            "TableBody",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=8.1,
            leading=9.7,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=8.1,
            leading=9.7,
            alignment=TA_CENTER,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=8.6,
            leading=10.5,
            alignment=TA_JUSTIFY,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=8.7,
            leading=10.8,
            leftIndent=16,
            firstLineIndent=-16,
            spaceAfter=5,
        ),
    }


def draw_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 9.0)
    canvas.drawCentredString(letter[0] / 2.0, 0.40 * inch, str(doc.page))
    canvas.restoreState()


def load_inputs(output_dir: Path) -> dict[str, object]:
    cost = pd.read_csv(output_dir / "first_minute_alpha_cost_sensitivity.csv")
    selection = pd.read_csv(output_dir / "first_minute_alpha_selection_adjusted_tests.csv")
    controls = pd.read_csv(output_dir / "first_minute_alpha_control_metrics.csv")
    regime = pd.read_csv(output_dir / "first_minute_alpha_regime_metrics.csv")
    coverage = pd.read_csv(output_dir / "data_quality_coverage.csv")
    top_metrics = pd.read_csv(output_dir / "topn_capped10_metrics.csv")
    top_dates = pd.read_csv(output_dir / "topn_capped10_date_portfolios.csv")
    top_active_rows: list[dict[str, object]] = []
    for (strategy, cost_bps), group in top_dates.groupby(
        ["strategy", "cost_bps"], sort=False
    ):
        top_active_rows.append(
            {
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                **active_trade_metrics(group, "net_return"),
            }
        )
    top_active_metrics = pd.DataFrame(top_active_rows)
    top_metrics = top_metrics.drop(
        columns=[
            column
            for column in top_active_metrics.columns
            if column not in {"strategy", "cost_bps"}
        ],
        errors="ignore",
    ).merge(
        top_active_metrics,
        on=["strategy", "cost_bps"],
        how="left",
        validate="one_to_one",
    )
    top_paired = pd.read_csv(output_dir / "topn_capped10_paired_tests.csv")
    top_regime = pd.read_csv(output_dir / "topn_capped10_regime_metrics.csv")
    top_sndk = pd.read_csv(output_dir / "topn_capped10_sndk_sensitivity.csv")
    volatility = pd.read_csv(output_dir / "volatility_diagnostic_metrics.csv")
    time_window_security = pd.read_csv(output_dir / "first_minute_alpha_time_window_security.csv")
    leg_regime = pd.read_csv(output_dir / "topn_capped10_leg_regime_attribution.csv")
    event_attribution = pd.read_csv(output_dir / "topn_capped10_quarterly_event_attribution.csv")
    quarter_seasonality = pd.read_csv(output_dir / "topn_capped10_quarter_seasonality.csv")
    early_drivers = pd.read_csv(output_dir / "topn_capped10_early_regime_drivers.csv")
    review_type = pd.read_csv(output_dir / "first_minute_alpha_review_type_diagnostics.csv")
    broad_change_type = pd.read_csv(
        output_dir / "first_minute_alpha_change_type_date_balanced.csv"
    )
    regime_clock = pd.read_csv(output_dir / "first_minute_alpha_regime_clock_profile.csv")
    influence = pd.read_csv(output_dir / "topn_capped10_regime_influence_diagnostics.csv")
    yearly_attribution = pd.read_csv(output_dir / "topn_capped10_yearly_leg_attribution.csv")
    state_dir = output_dir.parent / "state_filter_extension"
    state_metrics = pd.read_csv(state_dir / "state_filter_rule_metrics.csv")
    state_paired = pd.read_csv(state_dir / "state_filter_paired_tests.csv")
    state_regime = pd.read_csv(state_dir / "state_filter_regime_metrics.csv")
    state_walk = pd.read_csv(state_dir / "state_filter_walk_forward_metrics.csv")
    state_blend = pd.read_csv(state_dir / "state_filter_blend_frontier.csv")
    state_opt_dir = output_dir.parent / "state_filter_optimization"
    state_opt_metrics = pd.read_csv(state_opt_dir / "state_optimization_rule_metrics.csv")
    state_opt_paired = pd.read_csv(state_opt_dir / "state_optimization_paired_tests.csv")
    state_opt_walk = pd.read_csv(state_opt_dir / "state_optimization_walk_forward_metrics.csv")
    state_opt_walk_paired = pd.read_csv(
        state_opt_dir / "state_optimization_walk_forward_paired_tests.csv"
    )
    state_opt_loo = pd.read_csv(
        state_opt_dir / "state_optimization_leave_one_review_out.csv"
    )
    state_opt_audit = json.loads(
        (state_opt_dir / "state_optimization_manual_audit.json").read_text()
    )

    return {
        "cost": cost,
        "selection": selection,
        "controls": controls,
        "regime": regime,
        "coverage": coverage,
        "top_metrics": top_metrics,
        "top_paired": top_paired,
        "top_regime": top_regime,
        "top_sndk": top_sndk,
        "volatility": volatility,
        "time_window_security": time_window_security,
        "leg_regime": leg_regime,
        "event_attribution": event_attribution,
        "quarter_seasonality": quarter_seasonality,
        "early_drivers": early_drivers,
        "review_type": review_type,
        "broad_change_type": broad_change_type,
        "regime_clock": regime_clock,
        "influence": influence,
        "yearly_attribution": yearly_attribution,
        "state_metrics": state_metrics,
        "state_paired": state_paired,
        "state_regime": state_regime,
        "state_walk": state_walk,
        "state_blend": state_blend,
        "state_opt_metrics": state_opt_metrics,
        "state_opt_paired": state_opt_paired,
        "state_opt_walk": state_opt_walk,
        "state_opt_walk_paired": state_opt_walk_paired,
        "state_opt_loo": state_opt_loo,
        "state_opt_audit": state_opt_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--paper-path", type=Path, default=Path("output/pdf/msci_rebalance_closing_window_alpha_paper.pdf"))
    args = parser.parse_args()

    data = load_inputs(args.output_dir)
    cost: pd.DataFrame = data["cost"]  # type: ignore[assignment]
    selection: pd.DataFrame = data["selection"]  # type: ignore[assignment]
    controls: pd.DataFrame = data["controls"]  # type: ignore[assignment]
    regime: pd.DataFrame = data["regime"]  # type: ignore[assignment]
    coverage: pd.DataFrame = data["coverage"]  # type: ignore[assignment]
    top_metrics: pd.DataFrame = data["top_metrics"]  # type: ignore[assignment]
    top_paired: pd.DataFrame = data["top_paired"]  # type: ignore[assignment]
    top_regime: pd.DataFrame = data["top_regime"]  # type: ignore[assignment]
    top_sndk: pd.DataFrame = data["top_sndk"]  # type: ignore[assignment]
    volatility: pd.DataFrame = data["volatility"]  # type: ignore[assignment]
    time_window_security: pd.DataFrame = data["time_window_security"]  # type: ignore[assignment]
    leg_regime: pd.DataFrame = data["leg_regime"]  # type: ignore[assignment]
    event_attribution: pd.DataFrame = data["event_attribution"]  # type: ignore[assignment]
    quarter_seasonality: pd.DataFrame = data["quarter_seasonality"]  # type: ignore[assignment]
    early_drivers: pd.DataFrame = data["early_drivers"]  # type: ignore[assignment]
    review_type: pd.DataFrame = data["review_type"]  # type: ignore[assignment]
    broad_change_type: pd.DataFrame = data["broad_change_type"]  # type: ignore[assignment]
    regime_clock: pd.DataFrame = data["regime_clock"]  # type: ignore[assignment]
    influence: pd.DataFrame = data["influence"]  # type: ignore[assignment]
    yearly_attribution: pd.DataFrame = data["yearly_attribution"]  # type: ignore[assignment]
    state_metrics: pd.DataFrame = data["state_metrics"]  # type: ignore[assignment]
    state_paired: pd.DataFrame = data["state_paired"]  # type: ignore[assignment]
    state_regime: pd.DataFrame = data["state_regime"]  # type: ignore[assignment]
    state_walk: pd.DataFrame = data["state_walk"]  # type: ignore[assignment]
    state_blend: pd.DataFrame = data["state_blend"]  # type: ignore[assignment]
    state_opt_metrics: pd.DataFrame = data["state_opt_metrics"]  # type: ignore[assignment]
    state_opt_paired: pd.DataFrame = data["state_opt_paired"]  # type: ignore[assignment]
    state_opt_walk: pd.DataFrame = data["state_opt_walk"]  # type: ignore[assignment]
    state_opt_walk_paired: pd.DataFrame = data["state_opt_walk_paired"]  # type: ignore[assignment]
    state_opt_loo: pd.DataFrame = data["state_opt_loo"]  # type: ignore[assignment]
    state_opt_audit: dict[str, object] = data["state_opt_audit"]  # type: ignore[assignment]

    burst_zero = cost.loc[np.isclose(cost["cost_bps"], PRIMARY_COST_BPS)].iloc[0]
    burst_ten = cost.loc[np.isclose(cost["cost_bps"], SENSITIVITY_COST_BPS)].iloc[0]
    two_bar = selection.loc[selection["analysis"].eq("adjacent_two_bar_window")].iloc[0]
    control_diff = controls.loc[controls["analysis"].eq("event_minus_matched_control")].iloc[0]
    control_event = controls.loc[controls["analysis"].eq("event_on_complete_control_sample")].iloc[0]
    control_mean = controls.loc[controls["analysis"].eq("matched_control_mean")].iloc[0]
    early_zero = regime.loc[
        regime["sample"].eq("2020_through_2022") & np.isclose(regime["cost_bps"], 0.0)
    ].iloc[0]
    late_zero = regime.loc[
        regime["sample"].eq("2023_through_2026") & np.isclose(regime["cost_bps"], 0.0)
    ].iloc[0]
    excluding_discovery = regime.loc[
        regime["sample"].eq("excluding_aug_2026_discovery") & np.isclose(regime["cost_bps"], 0.0)
    ].iloc[0]

    top_zero = top_metrics.loc[
        top_metrics["strategy"].eq("impact_proxy_top10") & np.isclose(top_metrics["cost_bps"], 0.0)
    ].iloc[0]
    top_ten = top_metrics.loc[
        top_metrics["strategy"].eq("impact_proxy_top10") & np.isclose(top_metrics["cost_bps"], 10.0)
    ].iloc[0]
    full_zero = top_metrics.loc[
        top_metrics["strategy"].eq("capped_full_valid_universe") & np.isclose(top_metrics["cost_bps"], 0.0)
    ].iloc[0]
    paired = top_paired.loc[top_paired["strategy"].eq("impact_proxy_top10")].iloc[0]
    top_early_zero = top_regime.loc[
        top_regime["strategy"].eq("impact_proxy_top10")
        & top_regime["regime"].eq("2020_2022")
        & np.isclose(top_regime["cost_bps"], 0.0)
    ].iloc[0]
    top_late_zero = top_regime.loc[
        top_regime["strategy"].eq("impact_proxy_top10")
        & top_regime["regime"].eq("2023_plus")
        & np.isclose(top_regime["cost_bps"], 0.0)
    ].iloc[0]
    sndk_row = top_sndk.loc[top_sndk["strategy"].eq("impact_proxy_top10")].iloc[0]
    abs_move = volatility.loc[volatility["metric"].eq("absolute_terminal_move")].iloc[0]
    leg_all = leg_regime.loc[leg_regime["period"].eq("All")].iloc[0]
    leg_early = leg_regime.loc[leg_regime["period"].eq("2020-2022")].iloc[0]
    leg_late = leg_regime.loc[leg_regime["period"].eq("2023+")].iloc[0]
    clock_early_1550 = regime_clock.loc[
        regime_clock["regime"].eq("2020-2022") & regime_clock["clock"].eq("15:50")
    ].iloc[0]
    clock_late_1550 = regime_clock.loc[
        regime_clock["regime"].eq("2023+") & regime_clock["clock"].eq("15:50")
    ].iloc[0]
    top_post_ex_2026 = influence.loc[
        influence["portfolio"].eq("Impact top-10") & influence["sample"].eq("2023-2025")
    ].iloc[0]
    broad_post_ex_2026 = influence.loc[
        influence["portfolio"].eq("All-name") & influence["sample"].eq("2023-2025")
    ].iloc[0]
    year_2026 = yearly_attribution.loc[yearly_attribution["period"].astype(str).eq("2026")].iloc[0]
    state_baseline = state_metrics.loc[
        state_metrics["filter_rule"].eq("baseline_all")
        & np.isclose(state_metrics["cost_bps"], 10.0)
    ].iloc[0]
    state_gate = state_metrics.loc[
        state_metrics["filter_rule"].eq("down_to_small_hour_confirmation")
        & np.isclose(state_metrics["cost_bps"], 10.0)
    ].iloc[0]
    state_gate_paired = state_paired.loc[
        state_paired["filter_rule"].eq("down_to_small_hour_confirmation")
        & np.isclose(state_paired["cost_bps"], 10.0)
    ].iloc[0]
    state_walk_adaptive = state_walk.loc[
        state_walk["filter_rule"].eq("adaptive_walk_forward")
    ].iloc[0]
    state_walk_baseline = state_walk.loc[
        state_walk["filter_rule"].eq("matched_baseline")
    ].iloc[0]
    state_blend_80 = state_blend.loc[np.isclose(state_blend["gate_weight"], 0.8)].iloc[0]
    state_opt_baseline = state_opt_metrics.loc[
        state_opt_metrics["filter_rule"].eq("baseline_all")
        & np.isclose(state_opt_metrics["cost_bps"], 10.0)
    ].iloc[0]
    state_opt_025 = state_opt_metrics.loc[
        state_opt_metrics["filter_rule"].eq("down_to_small_hour_soft_025")
        & np.isclose(state_opt_metrics["cost_bps"], 10.0)
    ].iloc[0]
    state_opt_hard = state_opt_metrics.loc[
        state_opt_metrics["filter_rule"].eq("down_to_small_hour_confirmation")
        & np.isclose(state_opt_metrics["cost_bps"], 10.0)
    ].iloc[0]
    state_opt_025_paired = state_opt_paired.loc[
        state_opt_paired["filter_rule"].eq("down_to_small_hour_soft_025")
    ].iloc[0]
    state_opt_walk_constrained = state_opt_walk.loc[
        state_opt_walk["filter_rule"].eq("constrained_minimum_intervention")
    ].iloc[0]
    state_opt_walk_max = state_opt_walk.loc[
        state_opt_walk["filter_rule"].eq("maximum_training_mean")
    ].iloc[0]
    state_opt_walk_base = state_opt_walk.loc[
        state_opt_walk["filter_rule"].eq("matched_baseline")
    ].iloc[0]
    state_opt_walk_constrained_pair = state_opt_walk_paired.loc[
        state_opt_walk_paired["selection_policy"].eq(
            "constrained_minimum_intervention"
        )
    ].iloc[0]
    state_opt_loo_share = state_opt_loo["constrained_selected_rule"].eq(
        "down_to_small_hour_soft_025"
    ).mean()

    expected_events = int(coverage["expected_security_events"].sum())
    minute_usable = int(coverage["minute_entry_usable"].sum())
    close_usable = int(coverage["daily_close_proxy_usable"].sum())
    review_dates = int(coverage["effective_close_date"].nunique())
    scheduled_review_dates = int(
        pd.read_csv(Path(__file__).resolve().parents[1] / "data/events/review_calendar.csv").shape[0]
    )
    clock_panel_events = int(
        time_window_security[["effective_close_date", "ticker", "action"]].drop_duplicates().shape[0]
    )

    styles = build_styles()
    page_width, page_height = letter
    left_margin = 0.78 * inch
    right_margin = 0.78 * inch
    top_margin = 0.60 * inch
    bottom_margin = 0.62 * inch
    text_width = page_width - left_margin - right_margin
    text_height = page_height - top_margin - bottom_margin

    args.paper_path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(args.paper_path),
        pagesize=letter,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=PAPER_TITLE,
        author="Quantitative Research",
        subject="MSCI closing-window event study and exploratory portfolio construction",
    )
    frame = Frame(left_margin, bottom_margin, text_width, text_height, id="body")
    doc.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=draw_page_number)])

    story: list = []
    story.append(Spacer(1, 0.52 * inch))
    story.append(Paragraph(PAPER_TITLE, styles["title"]))
    story.append(Paragraph(PAPER_SUBTITLE, styles["subtitle"]))
    story.append(Paragraph("Quantitative Research", styles["author"]))
    story.append(Paragraph(date(2026, 9, 1).strftime("%B %-d, %Y"), styles["author"]))
    story.append(Paragraph("Abstract", styles["abstract_heading"]))

    abstract = (
        f"This paper studies direction-signed minute-level price pressure around MSCI U.S. Standard "
        f"Index implementation closes. The calendar contains {scheduled_review_dates} scheduled reviews, of which "
        f"{review_dates} have nonempty U.S. Standard changes from May 2020 through August 2026. The analyzed sample has "
        f"{expected_events} public addition/deletion events and "
        f"{minute_usable} usable 15:49 minute windows. The baseline proxy enters at the open of the "
        f"start-stamped 15:49 bar and exits at the close of the start-stamped 15:50 bar, approximately "
        f"15:50:59. At the 0 bp baseline transaction cost, the all-name direction-signed basket earns "
        f"{basis_points(burst_zero['mean_return_per_event'])} bp per scheduled review, compounds to "
        f"{signed_pct(burst_zero['total_return'])}, and has an event Sharpe of "
        f"{number(burst_zero['event_sharpe_sqrt4'])}. The interval was selected from 29 adjacent windows; "
        f"its four-review block max-t p-value is {number(two_bar['block4_max_t_familywise_pvalue'], 3)}, "
        f"and performance is concentrated after 2023. An exploratory balanced top-10 construction earns "
        f"{basis_points(top_zero['mean_return_per_event'])} bp per scheduled review at 0 bp, with "
        f"{signed_pct(top_zero['total_return'])} cumulative return, {pct(top_zero['cagr'])} cash-funded CAGR, "
        f"{number(top_zero['active_trade_sharpe_sqrt4'])} active-trade Sharpe, and {signed_pct(top_zero['max_drawdown'])} "
        f"event-step maximum drawdown. Its gross lift over the same-gate capped basket does not survive the 15-rule "
        f"family adjustment (block max-t p={number(paired['block4_max_t_familywise_pvalue'], 3)}). Additions contribute "
        f"{number(leg_all['long_signed_share_pct'], 1)}% and deletion shorts {number(leg_all['short_signed_share_pct'], 1)}% "
        f"of arithmetic gross profit. In 2020-2022, however, additions earn {number(leg_early['long_total_bp'])} bp while "
        f"deletion shorts lose {number(leg_early['short_total_bp'])} bp; this leg cancellation is the principal accounting "
        f"source of the weak early estimate. A pre-15:49 migration-aware gate raises the 10 bp active-date hit rate from "
        f"{pct(state_baseline['active_review_hit_rate'], 1)} to {pct(state_gate['active_review_hit_rate'], 1)}, but retains "
        f"only {number(100.0 * state_gate['arithmetic_net_return_bp'] / state_baseline['arithmetic_net_return_bp'], 1)}% of "
        f"baseline arithmetic net profit and does not improve paired return after family adjustment. The results document a "
        f"research effect, not executable arbitrage: prices are minute OHLCV proxies, the rank omits licensed "
        f"MSCI flows and float-adjusted capitalization, and the test has no historical NBBO, auction-imbalance, "
        f"depth, latency, borrow, or option-quote replay."
    )
    story.append(Paragraph(abstract, styles["abstract"]))
    story.append(
        Paragraph(
            "<b>Keywords:</b> index reconstitution, closing auction, price pressure, event study, "
            "point-in-time data, block bootstrap, multiple testing, portfolio construction.",
            styles["keywords"],
        )
    )

    story.append(Paragraph("1    Introduction", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "Index-tracking funds are evaluated against closing benchmark weights. When an index provider changes "
            "constituents or float-adjusted weights, benchmark-aware portfolios have an incentive to complete residual "
            "trading near the implementation close. Prior work documents announcement-to-implementation effects and "
            "unusually large closing-auction activity around MSCI reconstitutions [1]. Separate market-microstructure "
            "research shows that closing-imbalance dissemination can move prices before the auction itself [2-5]. The "
            "research question is whether these mechanisms produce a repeatable, direction-signed burst in U.S. stocks "
            "during the minute surrounding 15:50 Eastern Time.",
            "The empirical problem is unusually easy to overstate. Hundreds of stocks trade on each review date, but the "
            "independent shock is the review date, not the security. The exact interval was identified after inspecting the "
            "event-time profile. Public addition/deletion lists omit weight-only changes and do not reveal net product-level "
            "forced flow. Finally, a minute-bar open or close is not an executable bid or offer. The paper therefore treats "
            "review dates as the statistical units, adjusts for window search, keeps the 0 bp user scenario separate from a "
            "10 bp execution sensitivity, and labels every fill as a research proxy.",
            "The paper makes five contributions. First, it constructs a point-in-time event panel from official public MSCI "
            "review material. Second, it decomposes the closing window at one-minute frequency and tests the 15:49-open to "
            "15:50-bar-close interval. Third, it evaluates a pre-event top-10 concentration rule with explicit cash gates, "
            "weight caps, event-step performance, and security-level attribution. Fourth, it tests pre-entry stock, market, "
            "liquidity, volatility, and migration-state filters under a 15:47 safe cutoff. Fifth, it specifies the message-level data "
            "and sequential holdout evidence required before the strategy can be treated as deployable.",
        ],
    )

    story.append(CondPageBreak(4.0 * inch))
    story.append(Paragraph("2    Institutional Setting and Hypotheses", styles["h1"]))
    story.append(Paragraph("2.1    Reconstitution and the closing auction", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "MSCI announces periodic reviews before implementation. Index funds, separately managed accounts, derivatives "
            "desks, and benchmark-aware active managers may trade in advance, in the continuous market, or in the official "
            "closing auction. Their aggregate residual demand depends on old and new index weights, product assets, "
            "replication choices, crossing opportunities, and overlap across Standard, Small Cap, IMI, World, and ACWI "
            "families. An addition to Standard can simultaneously be a deletion from Small Cap, so a binary label is not an "
            "exact forced-flow sign.",
            "In the U.S. closing process, venue-specific imbalance information becomes available before the 16:00 auction. "
            "Nasdaq historically begins early imbalance dissemination around 15:50, while current NYSE auction procedures "
            "also publish imbalance information during the final minutes [2-4, 7-8]. Fast participants can incorporate "
            "indicative demand into continuous-market prices before the official close. This motivates a narrow event window "
            "without establishing that the observed minute bar is caused by a particular imbalance message.",
        ],
    )
    story.append(Paragraph("2.2    Mechanism and testable predictions", styles["h2"]))
    story.append(
        equation(
            "ExpectedFlow<sub>i,t</sub> = sum<sub>p</sub> AUM<sub>p,t</sub> replication<sub>p,t</sub> "
            "(weight<sub>i,t</sub><super>post</super> - weight<sub>i,t</sub><super>pre</super>)",
            1,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "Equation (1) defines the production quantity that should drive the trade; related benchmark-intensity evidence "
            "links assets times benchmark weights to price pressure [25]. It cannot be reconstructed from public "
            "lists alone. The current study instead tests the reduced-form implication that additions tend to move up and "
            "deletions down during the disclosure interval. The primary hypothesis is a positive mean direction-signed return "
            "from the 15:49 bar open to the 15:50 bar close. The second hypothesis is that the event-date effect exceeds the "
            "same-name, same-clock return from nearby prior sessions. The third is that a pre-event liquidity/volatility rank "
            "can concentrate the effect without creating excessive name risk.",
        ],
    )

    story.append(Paragraph("3    Data, Timing, and Event Construction", styles["h1"]))
    story.append(Paragraph("3.1    Sample and source lineage", styles["h2"]))
    coverage_rows = [
        ["Item", "Count", "Definition"],
        ["Scheduled / nonempty dates", f"{scheduled_review_dates} / {review_dates}", "One empty February 2020 review; analyzed sample begins May 2020"],
        ["Public security events", expected_events, "U.S. Standard additions and deletions after ticker resolution"],
        ["Usable 15:49 minute windows", minute_usable, f"{100 * minute_usable / expected_events:.1f}% of public events"],
        ["Usable daily-close proxies", close_usable, f"{100 * close_usable / expected_events:.1f}% of public events"],
        ["Controlled review dates", int(abs_move["review_dates"]), "Four complete prior sessions available"],
        ["Controlled security-events", int(abs_move["controlled_security_events"]), "Same-name pre-event comparison panel"],
        ["Top-10 portfolio dates", int(top_zero["event_dates"]), f"{int(top_zero['active_review_dates'])} active, {int(top_zero['cash_review_dates'])} cash"],
        ["Top-10 security records", 161, "80 long and 81 short research-proxy positions"],
    ]
    story.append(table_caption(1, "Sample construction and measurement coverage.", styles["table_caption"]))
    story.append(
        table_from_rows(
            coverage_rows,
            [2.05 * inch, 0.72 * inch, text_width - 2.77 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=8.15,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    add_paragraphs(
        story,
        styles["body"],
        [
            "Official public review lists identify security names and announced additions/deletions. Permanent identifiers and "
            "complete pro-forma index weights are unavailable in the public source. Ticker mappings are therefore resolved "
            "with an explicit override ledger, and coverage failures remain in the audit output rather than being silently "
            "dropped. MarketData minute OHLCV is used for the underlying price path [9]. Its timestamps identify the start of "
            "each candle. The 15:50 exit proxy is consequently the close of the bar labeled 15:50, approximately 15:50:59.",
            "The base impact ranking is frozen before the event session. The second-stage state filter uses only bars "
            "completed by 15:47:59.999 Eastern Time, leaving one full minute before the 15:49 entry proxy; daily liquidity "
            "features use data no later than the preceding session. No 15:48 bar, 15:50 imbalance, entry-window return, "
            "implementation-session daily close, or later observation is permitted to enter the production-safe rule.",
        ],
    )

    story.append(CondPageBreak(4.0 * inch))
    story.append(Paragraph("4    Empirical Design and Portfolio Construction", styles["h1"]))
    story.append(Paragraph("4.1    Direction-signed event return", styles["h2"]))
    story.append(
        equation(
            "SignedReturn<sub>i,t</sub> = side<sub>i,t</sub> "
            "(Price<sub>i,t,15:50-close</sub> / Price<sub>i,t,15:49-open</sub> - 1), "
            "where side = +1 for additions and -1 for deletions",
            2,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    story.append(
        equation(
            "ReviewReturn<sub>t</sub> = 0.5 mean<sub>adds</sub>(SignedReturn<sub>i,t</sub>) "
            "+ 0.5 mean<sub>deletes</sub>(SignedReturn<sub>i,t</sub>)",
            3,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "The date portfolio balances the addition and deletion legs so dates with many deletions do not mechanically "
            "dominate. If a leg is unavailable, its capital remains in cash. Security observations within a date are not "
            "treated as independent. The review return is the atomic observation for t-tests, sign tests, block bootstraps, "
            "equity curves, and multiple-testing adjustments.",
        ],
    )
    story.append(Paragraph("4.2    Exploratory top-10 impact proxy", styles["h2"]))
    story.append(
        equation(
            "ImpactProxy<sub>i,t</sub> = Volatility<sub>i,t,20d</sub> "
            "sqrt( ADV<sub>i,t,20d</sub> / ExpectedClockDollarVolume<sub>i,t</sub> )",
            4,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "Equation (4) is a shadow ranking score, not an estimate of MSCI capital. It favors securities with high pre-event "
            "volatility and low expected same-clock liquidity relative to a synthetic order-size proxy. On each eligible date, "
            "the strategy selects up to five additions and five deletions. Each name receives at most 10% of capital; each leg "
            "receives at most 50%. A date remains entirely in cash unless at least three point-in-time-valid names are available "
            "on both sides. This produces 17 active and nine cash reviews, with six to ten names on active dates.",
        ],
    )
    strategy_rows = [
        ["Rule", "Signal / selection", "Weights", "Cost treatment", "Role"],
        ["All-name burst", "Public add/delete sign; all usable names", "50% per leg; date-balanced", "0 bp primary; 10 bp sensitivity", "Reduced-form baseline"],
        ["Capped valid universe", "All PIT-valid names after three-per-side gate", "10% name cap; residual cash", "0 and 10 bp", "Matched-gate control"],
        ["Impact top-10", "Up to five per side by Equation (4)", "10% name cap; up to 100% gross", "0 bp primary; 10 bp sensitivity", "Exploratory strategy"],
        ["State-filter challenger", "15:47 stock/SPY state plus migration gate", "Original cap; rebalance legs down; residual cash", "10 bp comparison", "Retrospective risk control"],
        ["Volatility diagnostic", "Absolute 15:49-15:59 movement", "No investable option weights", "Underlying only", "Mechanism test"],
    ]
    story.append(table_caption(2, "Strategy definitions and research status.", styles["table_caption"]))
    story.append(
        table_from_rows(
            strategy_rows,
            [1.12 * inch, 2.05 * inch, 1.38 * inch, 1.18 * inch, text_width - 5.73 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.65,
        )
    )

    story.append(Paragraph("4.3    Performance and inference", styles["h2"]))
    story.append(
        equation(
            "Wealth<sub>T</sub> = product<sub>t=1...T</sub>(1 + ReviewReturn<sub>t</sub>)",
            5,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    story.append(
        equation(
            "ActiveTradeSharpe = sqrt(4) mean(ReviewReturn<sub>t</sub> | GrossExposure<sub>t</sub> &gt; 0) / sd(ReviewReturn<sub>t</sub> | GrossExposure<sub>t</sub> &gt; 0)",
            6,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    story.append(
        equation(
            "Drawdown<sub>t</sub> = Wealth<sub>t</sub> / max<sub>s&lt;=t</sub>(Wealth<sub>s</sub>) - 1",
            7,
            styles["equation"],
            styles["equation_number"],
            text_width,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "Active-trade Sharpe annualizes the conditional trade series by sqrt(4) and excludes every zero-exposure review. "
            "The raw output retains Scheduled Event Sharpe as a separately labeled coverage diagnostic. The current risk-free "
            "input is zero per one-minute event until a point-in-time funding series is joined. CAGR is cash-funded over elapsed "
            "calendar time, despite trading only on implementation dates. Four-review moving blocks "
            "preserve serial dependence. The time-window family contains 29 adjacent two-bar candidates; the portfolio family "
            "contains 15 combinations of ranker and top-5/top-10/top-20 breadth. These adjustments are essential because the "
            "reported interval and concentrated rule were selected after exploration [10].",
        ],
    )

    story.append(PageBreak())
    story.append(Paragraph("5    Empirical Results", styles["h1"]))
    story.append(Paragraph("5.1    Headline scorecard", styles["h2"]))
    scorecard = [
        ["Portfolio", "Cost", "Mean / sched.", "Cumulative", "CAGR", "Active Sharpe", "Max DD", "Block mean CI"],
        [
            "All-name burst",
            "0 bp",
            f"{basis_points(burst_zero['mean_return_per_event'])} bp",
            signed_pct(burst_zero["total_return"]),
            signed_pct(burst_zero["cagr"]),
            number(burst_zero["event_sharpe_sqrt4"]),
            signed_pct(burst_zero["max_drawdown"]),
            f"[{basis_points(burst_zero['block_bootstrap_mean_ci_low'])}, {basis_points(burst_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "All-name burst",
            "10 bp sensitivity",
            f"{basis_points(burst_ten['mean_return_per_event'])} bp",
            signed_pct(burst_ten["total_return"]),
            signed_pct(burst_ten["cagr"]),
            number(burst_ten["event_sharpe_sqrt4"]),
            signed_pct(burst_ten["max_drawdown"]),
            f"[{basis_points(burst_ten['block_bootstrap_mean_ci_low'])}, {basis_points(burst_ten['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Capped full valid universe",
            "0 bp",
            f"{basis_points(full_zero['mean_return_per_event'])} bp",
            signed_pct(full_zero["total_return"]),
            signed_pct(full_zero["cagr"]),
            number(full_zero["active_trade_sharpe_sqrt4"]),
            signed_pct(full_zero["max_drawdown"]),
            f"[{basis_points(full_zero['block_bootstrap_mean_ci_low'])}, {basis_points(full_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Impact top-10",
            "0 bp primary",
            f"{basis_points(top_zero['mean_return_per_event'])} bp",
            signed_pct(top_zero["total_return"]),
            signed_pct(top_zero["cagr"]),
            number(top_zero["active_trade_sharpe_sqrt4"]),
            signed_pct(top_zero["max_drawdown"]),
            f"[{basis_points(top_zero['block_bootstrap_mean_ci_low'])}, {basis_points(top_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Impact top-10",
            "10 bp sensitivity",
            f"{basis_points(top_ten['mean_return_per_event'])} bp",
            signed_pct(top_ten["total_return"]),
            signed_pct(top_ten["cagr"]),
            number(top_ten["active_trade_sharpe_sqrt4"]),
            signed_pct(top_ten["max_drawdown"]),
            f"[{basis_points(top_ten['block_bootstrap_mean_ci_low'])}, {basis_points(top_ten['block_bootstrap_mean_ci_high'])}] bp",
        ],
    ]
    story.append(table_caption(3, "Cash-funded event performance, May 2020-August 2026.", styles["table_caption"]))
    story.append(
        table_from_rows(
            scorecard,
            [1.20 * inch, 0.85 * inch, 0.82 * inch, 0.70 * inch, 0.62 * inch, 0.76 * inch, 0.63 * inch, text_width - 5.58 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.35,
        )
    )
    story.append(
        Paragraph(
            "<i>Notes:</i> The 0 bp rows implement the user's selected transaction-cost assumption. The 10 bp rows are "
            "execution sensitivities, not the primary scenario. Active Sharpe uses sqrt(4) and excludes zero-exposure "
            "review dates; Scheduled Event Sharpe remains in the downloadable metrics for coverage analysis. CAGR is "
            "cash-funded over elapsed calendar time; Max DD is event-step. "
            "Prices are minute-bar proxies rather than executable quotes.",
            styles["note"],
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The all-name burst produces {basis_points(burst_zero['mean_return_per_event'])} bp per review at 0 bp cost, "
            f"with a {number(burst_zero['event_sharpe_sqrt4'])} event Sharpe. A 10 bp haircut reduces the mean to "
            f"{basis_points(burst_ten['mean_return_per_event'])} bp and the Sharpe to "
            f"{number(burst_ten['event_sharpe_sqrt4'])}; its block confidence interval crosses zero. The impact top-10 "
            f"portfolio earns {basis_points(top_zero['mean_return_per_event'])} bp per scheduled review and "
            f"{basis_points(top_zero['active_mean_return'])} bp per active review, with a smoother 0 bp event-step path and a "
            f"{number(top_zero['active_trade_sharpe_sqrt4'])} active-trade Sharpe, but "
            "this descriptive improvement is selected from a strategy family and cannot be interpreted as independent "
            "confirmation.",
        ],
    )

    story.append(Paragraph("5.2    Event-time anatomy", styles["h2"]))
    story.append(
        figure(
            args.output_dir / "charts" / "event_time_profile.png",
            f"Direction-signed cumulative return and same-clock volume profile across {review_dates} implementation "
            "dates. The bar timestamp denotes the minute bucket; vertical markers identify the 15:49/15:50 boundary "
            "and venue timing landmarks. Shading is descriptive across review dates. Source: audited minute-OHLCV "
            "research output.",
            1,
            styles["caption"],
            text_width,
            3.72 * inch,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "The mean direction-signed path accelerates sharply around the 15:50 boundary, while same-clock volume is "
            "already elevated. The event profile is consistent with residual benchmark demand and imbalance information "
            "being incorporated into continuous-market prices. It is not a causal NOII test: a one-minute candle aggregates "
            "all prints in the interval and does not identify which venue message arrived first.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "window_decomposition.png",
            "Non-overlapping closing-window return decomposition. Points are gross date-balanced means and bars are "
            "95% four-review block-bootstrap intervals. The 15:49-open to 15:50-open component is positive; later "
            "continuous-market and daily-close-proxy components are unstable. Source: audited backtest output.",
            2,
            styles["caption"],
            text_width,
            3.42 * inch,
        )
    )

    story.append(Paragraph("5.3    Equity path and cost sensitivity", styles["h2"]))
    story.append(
        figure(
            args.output_dir / "charts" / "first_minute_alpha_equity.png",
            "Event-step equity and drawdown for the all-name 15:49-open to 15:50-bar-close proxy. The gross line uses "
            "0 bp; the conservative comparison applies the stated 10 bp haircut. Cash is held between quarterly "
            "implementation dates. Source: audited backtest output.",
            3,
            styles["caption"],
            text_width,
            4.35 * inch,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The gross break-even estimate is {number(burst_zero['break_even_cost_bps'])} bp per round trip. That number "
            "is an arithmetic threshold under the bar-fill proxy, not a capacity estimate. The 10 bp sensitivity remains "
            "positive in cumulative terms, but its uncertainty is wide and the realized drawdown is materially larger. "
            "Observed commissions of zero do not eliminate spread, latency, market impact, or adverse selection.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "first_minute_alpha_cost_regime.png",
            "Left: cumulative return over an all-in haircut grid. Right: gross mean by historical slice. The vertical "
            "line marks estimated arithmetic break-even. Performance is concentrated after 2023, near the transition "
            "to comprehensive quarterly index reviews. Source: audited backtest output.",
            4,
            styles["caption"],
            text_width,
            3.35 * inch,
        )
    )

    story.append(Paragraph("5.4    Regime dependence and discovery bias", styles["h2"]))
    regime_rows = [
        ["Slice", "Dates", "Cost", "Mean / review", "Cumulative", "Event Sharpe", "Block mean CI"],
        [
            "2020-2022",
            int(early_zero["event_dates"]),
            "0 bp",
            f"{basis_points(early_zero['mean_return_per_event'])} bp",
            signed_pct(early_zero["total_return"]),
            number(early_zero["event_sharpe_sqrt4"]),
            f"[{basis_points(early_zero['block_bootstrap_mean_ci_low'])}, {basis_points(early_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "2023-2026",
            int(late_zero["event_dates"]),
            "0 bp",
            f"{basis_points(late_zero['mean_return_per_event'])} bp",
            signed_pct(late_zero["total_return"]),
            number(late_zero["event_sharpe_sqrt4"]),
            f"[{basis_points(late_zero['block_bootstrap_mean_ci_low'])}, {basis_points(late_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Excluding August 2026",
            int(excluding_discovery["event_dates"]),
            "0 bp",
            f"{basis_points(excluding_discovery['mean_return_per_event'])} bp",
            signed_pct(excluding_discovery["total_return"]),
            number(excluding_discovery["event_sharpe_sqrt4"]),
            f"[{basis_points(excluding_discovery['block_bootstrap_mean_ci_low'])}, {basis_points(excluding_discovery['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Impact top-10, 2020-2022",
            int(top_early_zero["event_dates"]),
            "0 bp",
            f"{basis_points(top_early_zero['mean_return_per_event'])} bp",
            "-",
            "-",
            f"[{basis_points(top_early_zero['block_bootstrap_mean_ci_low'])}, {basis_points(top_early_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
        [
            "Impact top-10, 2023+",
            int(top_late_zero["event_dates"]),
            "0 bp",
            f"{basis_points(top_late_zero['mean_return_per_event'])} bp",
            "-",
            "-",
            f"[{basis_points(top_late_zero['block_bootstrap_mean_ci_low'])}, {basis_points(top_late_zero['block_bootstrap_mean_ci_high'])}] bp",
        ],
    ]
    story.append(table_caption(4, "Regime and discovery-sample diagnostics.", styles["table_caption"]))
    story.append(
        table_from_rows(
            regime_rows,
            [1.62 * inch, 0.48 * inch, 0.55 * inch, 0.88 * inch, 0.78 * inch, 0.78 * inch, text_width - 5.09 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.65,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The all-name mean is {basis_points(early_zero['mean_return_per_event'])} bp in 2020-2022 and "
            f"{basis_points(late_zero['mean_return_per_event'])} bp in 2023-2026. Excluding the August 2026 observation "
            f"that motivated the research leaves {basis_points(excluding_discovery['mean_return_per_event'])} bp per "
            "review, but does not create a genuine out-of-sample test. MSCI's first comprehensive quarterly review was "
            "implemented at the February 2023 close; full quarterly NOS/FIF review followed in May 2023 [16-18]. The first "
            "future reviews, using a frozen rule and "
            "unchanged data pipeline, are the relevant holdouts. Appendices C-E decompose every review into addition-long "
            "and deletion-short contributions and audit the 2020-2022 regime break.",
        ],
    )
    story.append(Paragraph("5.5    Window search and controls", styles["h2"]))
    robustness_rows = [
        ["Test", "Estimate", "Interval / adjustment", "Decision"],
        [
            f"Clock panel (n={clock_panel_events})",
            f"{basis_points(two_bar['candidate_mean_return'])} bp",
            f"29-window block max-t p={number(two_bar['block4_max_t_familywise_pvalue'], 3)}",
            f"No reject; n={minute_usable}: {basis_points(burst_zero['mean_return_per_event'])} bp",
        ],
        [
            "Event on complete-control sample",
            f"{basis_points(control_event['mean_return_per_event'])} bp",
            f"block CI [{basis_points(control_event['block_bootstrap_mean_ci_low'])}, {basis_points(control_event['block_bootstrap_mean_ci_high'])}]",
            "Positive event mean",
        ],
        [
            "Four-prior-session mean",
            f"{basis_points(control_mean['mean_return_per_event'])} bp",
            f"block CI [{basis_points(control_mean['block_bootstrap_mean_ci_low'])}, {basis_points(control_mean['block_bootstrap_mean_ci_high'])}]",
            "Positive but much smaller",
        ],
        [
            "Event minus prior-session control",
            f"{basis_points(control_diff['mean_return_per_event'])} bp",
            f"block CI [{basis_points(control_diff['block_bootstrap_mean_ci_low'])}, {basis_points(control_diff['block_bootstrap_mean_ci_high'])}]",
            "Interval crosses zero",
        ],
        [
            "Impact top-10 lift vs matched full basket",
            f"{basis_points(paired['mean_difference'])} bp",
            f"15-rule block max-t p={number(paired['block4_max_t_familywise_pvalue'], 3)}",
            "Selection lift not validated",
        ],
    ]
    story.append(
        table_caption(
            5,
            "Selection-adjusted and same-name control evidence; clock row is 15:49-open to 15:50-bar-close.",
            styles["table_caption"],
        )
    )
    story.append(
        table_from_rows(
            robustness_rows,
            [1.82 * inch, 0.85 * inch, 2.10 * inch, text_width - 4.77 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.8,
        )
    )
    story.append(Paragraph("6    Identification and Robustness", styles["h1"]))
    story.append(
        figure(
            args.output_dir / "charts" / "first_minute_alpha_controls.png",
            f"Same-name, same-clock comparison on {int(control_diff['event_dates'])} review dates with four complete "
            "prior sessions. Points are date-balanced means; intervals are four-review block bootstraps. The event-minus-"
            "control point estimate is positive, but its interval crosses zero. Source: audited control output.",
            5,
            styles["caption"],
            text_width,
            3.55 * inch,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            "The same-name control reduces static differences in volatility and liquidity, but it is not a complete causal "
            "design. Prior sessions do not share the review day's market shock, month-end flow, earnings news, sector news, "
            "or other index events. A stronger design would combine exact historical NOII, message-level quotes, official "
            "auction prints, contemporaneous non-event controls, and predetermined interaction terms based on licensed "
            "expected flow.",
            "The cross-section cannot manufacture statistical breadth. Treating 459 security-events as independent would "
            "severely understate uncertainty because all names on a date share the same close, index review, and market state. "
            "All reported confidence intervals and p-values therefore operate on the 26-date series or on date-level paired "
            "differences.",
        ],
    )

    story.append(Paragraph("7    Exploratory Top-10 Portfolio", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            f"At the selected 0 bp cost, impact top-10 earns {basis_points(top_zero['mean_return_per_event'])} bp per "
            f"scheduled review and {basis_points(top_zero['active_mean_return'])} bp per active review. It compounds to "
            f"{signed_pct(top_zero['total_return'])}, with {pct(top_zero['cagr'])} cash-funded CAGR, a "
            f"{number(top_zero['active_trade_sharpe_sqrt4'])} active-trade Sharpe, and {signed_pct(top_zero['max_drawdown'])} event-step "
            f"maximum drawdown. The 10 bp sensitivity compounds to {signed_pct(top_ten['total_return'])} and has a "
            f"{number(top_ten['active_trade_sharpe_sqrt4'])} active-trade Sharpe.",
            f"The gross lift over the same-gate capped universe is only {basis_points(paired['mean_difference'])} bp per "
            f"scheduled review, with a block interval of [{basis_points(paired['block4_mean_ci_low'])}, "
            f"{basis_points(paired['block4_mean_ci_high'])}] bp and a 15-rule familywise p-value of "
            f"{number(paired['block4_max_t_familywise_pvalue'], 3)}. Removing SNDK without renormalizing lowers the gross "
            f"mean from {basis_points(top_zero['mean_return_per_event'])} to "
            f"{basis_points(sndk_row['full_mean_without_sndk_cash'])} bp. The motivating name matters, but it does not "
            "explain the full result.",
            f"The long and short legs receive symmetric capital budgets on active reviews, but their realized arithmetic "
            f"profit shares are {number(leg_all['long_signed_share_pct'], 2)}% from addition longs and "
            f"{number(leg_all['short_signed_share_pct'], 2)}% from deletion shorts. Before 2023, the long leg contributes "
            f"{number(leg_early['long_total_bp'])} bp and the short leg contributes {number(leg_early['short_total_bp'])} bp; "
            f"after the February 2023 breakpoint the figures are {number(leg_late['long_total_bp'])} and "
            f"{number(leg_late['short_total_bp'])} bp. This distinction matters because signed contribution shares can "
            "exceed 100% or become negative whenever one leg offsets the other.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "topn_capped10_equity_drawdown.png",
            "Exploratory impact top-10 event-step equity. The green line is the selected 0 bp primary scenario; the blue "
            "line is the 10 bp sensitivity; the dashed line is the same-gate full-basket control at 0 bp. Hollow markers "
            "denote cash reviews. The lower panel is 0 bp primary drawdown. Source: audited top-10 output.",
            6,
            styles["caption"],
            text_width,
            4.25 * inch,
        )
    )

    story.append(Paragraph("8    Volatility Evidence and Option Boundary", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            f"Implementation names exhibit much larger pre-close movement than the same names in four prior sessions. The "
            f"mean absolute terminal move is {basis_points(abs_move['event_mean_across_dates'], 1)} bp on event dates "
            f"versus {basis_points(abs_move['pre_event_control_mean_across_dates'], 1)} bp in controls, an excess of "
            f"{basis_points(abs_move['mean_event_minus_pre_event_control'], 1)} bp with a block interval of "
            f"[{basis_points(abs_move['block_bootstrap_excess_ci_low'], 1)}, "
            f"{basis_points(abs_move['block_bootstrap_excess_ci_high'], 1)}] bp. This supports the premise that the "
            "closing window is unusually volatile.",
            "The result is not an option-straddle backtest. A tradeable long-volatility claim requires historical OPRA bid, "
            "ask, size, trades, contract specifications, implied volatility, Greeks, exercise rules, and exact entry/exit "
            "timestamps. Evidence from scheduled earnings announcements shows that large underlying moves can coexist "
            "with negative delta-neutral straddle returns when event risk is richly priced [11]. The current option "
            "calculations are sensitivity analysis and remain "
            "outside the validated strategy.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "volatility_pre_event_excess.png",
            f"Event-minus-control movement diagnostics on {int(abs_move['review_dates'])} controlled review dates and "
            f"{int(abs_move['controlled_security_events'])} security-events. Intervals are four-review block bootstraps. "
            "These are underlying-price diagnostics, not option P&amp;L. Source: audited volatility output.",
            7,
            styles["caption"],
            text_width,
            3.55 * inch,
        )
    )

    story.append(Paragraph("9    Economic Interpretation", styles["h1"]))
    story.append(Paragraph("9.1    Why a burst may exist", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "A plausible chain is: announced weight change creates expected benchmark demand; residual orders accumulate "
            "toward the close; venue imbalance messages reveal a portion of the remaining demand; continuous-market "
            "liquidity providers update prices; and the official auction clears a large quantity near the newly discovered "
            "price. This mechanism explains how the largest volume can occur at or after 15:51 while much of the price change "
            "begins around 15:50. The current data show timing consistency, not message-level causality.",
        ],
    )
    story.append(Paragraph("9.2    Why concentration may appear to help", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "Price impact should be larger when a given order is large relative to available liquidity and when short-horizon "
            "volatility is high [5]. The impact proxy encodes that intuition. Its descriptive improvement can also arise from "
            "selection noise, omitted actual flow, or late-sample concentration. Because the 15-rule familywise test does not "
            "reject, the score is a frozen challenger rather than a validated production rank.",
        ],
    )
    story.append(Paragraph("9.3    Why 0 bp is informative but insufficient", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "The 0 bp scenario isolates the underlying price pattern. It is useful for measuring the economic "
            "signal before explicit costs. It does not imply costless execution. Bid-ask spread, price impact, queue priority, "
            "latency, borrow, rejects, and partial fills remain unobserved. The 10 bp sensitivity demonstrates that a modest "
            "haircut materially reduces performance, especially before 2023.",
        ],
    )

    story.append(Paragraph("9.4    Pre-entry market-state and migration filters", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "A second-stage filter study uses only bars completed by 15:47:59.999 Eastern Time, leaving one full minute "
            "before the 15:49 entry proxy. Candidate variables include stock day, hour and week returns; side-signed stock-minus-SPY "
            "returns; intraday realized volatility; cumulative volume relative to ADV; same-clock historical RVOL; and "
            "MSCI migration type. Market-level intraday momentum and stock-level same-clock/reversal research motivate testing "
            "both continuation and exhaustion rather than imposing one sign [26-27]. One TPL row on August 31, 2022 lacks a "
            "complete safe-cutoff hour and fails closed.",
            "The most useful mechanism rule targets the weak deletion leg. It retains every addition and true Standard "
            "exit, but requires a down-to-small deletion to have positive direction-signed stock-minus-SPY return from "
            "the 14:48 bar open through the 15:47 bar close. Surviving positions keep the original 10% cap; only the "
            "larger leg is scaled down to restore dollar neutrality, and unused capital remains cash.",
        ],
    )
    story.append(
        figure(
            args.output_dir.parent
            / "state_filter_extension"
            / "charts"
            / "state_filter_comparison.png",
            "Cash-funded 10 bp equity for the baseline, migration-aware hour confirmation, same-clock RVOL confirmation, "
            "and expanding rule selection. The right panel shows the return, active-date hit-rate and exposure trade-off. "
            "Source: frozen 15:47 state-feature panel and review-date portfolio outputs.",
            8,
            styles["caption"],
            text_width,
            3.25 * inch,
        )
    )
    state_rows = [
        ["Rule / sample", "Reviews", "Net bp", "Active", "Active hit", "Mean gross", "Active Sharpe", "Max DD"],
        [
            "Baseline / all",
            26,
            number(state_baseline["arithmetic_net_return_bp"]),
            int(state_baseline["active_review_dates"]),
            pct(state_baseline["active_review_hit_rate"], 1),
            pct(state_baseline["mean_gross_exposure"], 1),
            number(state_baseline["active_trade_sharpe_sqrt4"]),
            signed_pct(state_baseline["max_drawdown"]),
        ],
        [
            "Migration gate / all",
            26,
            number(state_gate["arithmetic_net_return_bp"]),
            int(state_gate["active_review_dates"]),
            pct(state_gate["active_review_hit_rate"], 1),
            pct(state_gate["mean_gross_exposure"], 1),
            number(state_gate["active_trade_sharpe_sqrt4"]),
            signed_pct(state_gate["max_drawdown"]),
        ],
        [
            "80% gate blend / all",
            26,
            number(state_blend_80["arithmetic_net_return_bp"]),
            int(state_blend_80["active_review_dates"]),
            pct(state_blend_80["active_review_hit_rate"], 1),
            pct(state_blend_80["mean_gross_exposure"], 1),
            number(state_blend_80["active_trade_sharpe_sqrt4"]),
            signed_pct(state_blend_80["max_drawdown"]),
        ],
        [
            "Adaptive / pseudo-OOS",
            int(state_walk_adaptive["event_dates"]),
            number(state_walk_adaptive["arithmetic_net_return_bp"]),
            int(state_walk_adaptive["active_review_dates"]),
            pct(state_walk_adaptive["active_review_hit_rate"], 1),
            pct(state_walk_adaptive["mean_gross_exposure"], 1),
            number(state_walk_adaptive["active_trade_sharpe_sqrt4"]),
            signed_pct(state_walk_adaptive["max_drawdown"]),
        ],
        [
            "Baseline / matched pseudo-OOS",
            int(state_walk_baseline["event_dates"]),
            number(state_walk_baseline["arithmetic_net_return_bp"]),
            int(state_walk_baseline["active_review_dates"]),
            pct(state_walk_baseline["active_review_hit_rate"], 1),
            pct(state_walk_baseline["mean_gross_exposure"], 1),
            number(state_walk_baseline["active_trade_sharpe_sqrt4"]),
            signed_pct(state_walk_baseline["max_drawdown"]),
        ],
    ]
    story.append(table_caption(6, "Pre-15:49 state-filter return, hit-rate and exposure trade-off at 10 bp.", styles["table_caption"]))
    story.append(
        table_from_rows(
            state_rows,
            [1.48 * inch, 0.52 * inch, 0.58 * inch, 0.48 * inch, 0.66 * inch, 0.68 * inch, 0.70 * inch, text_width - 5.10 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.15,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            f"At 10 bp, the static migration gate raises active-date hit rate from "
            f"{pct(state_baseline['active_review_hit_rate'], 1)} to {pct(state_gate['active_review_hit_rate'], 1)} and "
            f"reduces maximum drawdown from {signed_pct(state_baseline['max_drawdown'])} to "
            f"{signed_pct(state_gate['max_drawdown'])}. Arithmetic net profit falls from "
            f"{number(state_baseline['arithmetic_net_return_bp'])} to {number(state_gate['arithmetic_net_return_bp'])} bp. "
            f"The paired incremental mean is {number(state_gate_paired['mean_paired_increment_bp'])} bp per scheduled "
            f"review and the seven-rule block max-t p-value is "
            f"{number(state_gate_paired['block4_max_t_familywise_pvalue'], 3)}; the rule does not beat baseline return.",
            f"The 80% gate blend retains {pct(state_blend_80['net_return_retention_vs_baseline'], 1)} of baseline net profit and raises "
            f"active-date hit rate by {100 * state_blend_80['active_hit_rate_improvement']:.1f} percentage points. Its "
            "mixing weight was selected on the same sample and is reported only as a return-reliability frontier, not as a "
            "frozen production parameter.",
            f"The expanding selector starts after eight reviews and uses only earlier dates at every step. Across 18 "
            f"pseudo-out-of-sample reviews it increases active-date hit rate from "
            f"{pct(state_walk_baseline['active_review_hit_rate'], 1)} to "
            f"{pct(state_walk_adaptive['active_review_hit_rate'], 1)} while retaining "
            f"{pct(state_walk_adaptive['net_return_retention_vs_matched_baseline'], 1)} of matched-baseline net profit. "
            "This is retrospective pseudo-OOS because the candidate family was formed after the full history existed. "
            "High same-clock RVOL underperforms the low-RVOL half and is rejected as a standalone gate. The migration rule "
            "is therefore a stability challenger that requires eight genuinely future quarterly reviews.",
        ],
    )

    story.append(Paragraph("9.5    Bounded soft-gate optimization", styles["h2"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "Before running the extension, the candidate family was frozen to the baseline, three monotone soft gates "
            "that retain 75%, 50%, or 25% of an unconfirmed down-to-small short, and the existing hard gate. Day/hour "
            "OR and AND rules are mechanism diagnostics and cannot be selected. The expanding selector begins after eight "
            "reviews and chooses the least intervention that retains at least 80% of training net profit, improves active-date "
            "hit rate by five percentage points, preserves at least 35% of baseline exposure, and has three active dates.",
            "The 25% soft gate is the sole full-history constrained choice. After attenuation, only the larger surviving "
            "long or short leg is scaled down to restore dollar neutrality; unused capital stays in cash. This construction "
            "prevents an adverse state signal from reversing the MSCI direction and keeps the 10% single-name cap.",
        ],
    )
    story.append(
        figure(
            args.output_dir.parent
            / "state_filter_optimization"
            / "charts"
            / "state_filter_optimization.png",
            "Frozen soft-gate optimization. Top left: fixed 10 bp equity. Top right: profit versus active-date hit rate. "
            "Bottom left: 0-20 bp cost stress. Bottom right: expanding pseudo-out-of-sample paths against the matched baseline. "
            "Source: frozen optimization specification and review-date grouped portfolios.",
            9,
            styles["caption"],
            text_width,
            4.35 * inch,
        )
    )
    state_opt_rows = [
        ["Rule / policy", "Reviews", "Net bp", "Net retained", "Active hit", "Active Sharpe", "Max DD", "Mean gross"],
        [
            "Baseline / full",
            int(state_opt_baseline["event_dates"]),
            number(state_opt_baseline["arithmetic_net_return_bp"]),
            "100.0%",
            pct(state_opt_baseline["active_review_hit_rate"], 1),
            number(state_opt_baseline["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_baseline["max_drawdown"]),
            pct(state_opt_baseline["mean_gross_exposure"], 1),
        ],
        [
            "Retain 25% / full",
            int(state_opt_025["event_dates"]),
            number(state_opt_025["arithmetic_net_return_bp"]),
            pct(
                state_opt_025["arithmetic_net_return_bp"]
                / state_opt_baseline["arithmetic_net_return_bp"],
                1,
            ),
            pct(state_opt_025["active_review_hit_rate"], 1),
            number(state_opt_025["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_025["max_drawdown"]),
            pct(state_opt_025["mean_gross_exposure"], 1),
        ],
        [
            "Hard gate / full",
            int(state_opt_hard["event_dates"]),
            number(state_opt_hard["arithmetic_net_return_bp"]),
            pct(
                state_opt_hard["arithmetic_net_return_bp"]
                / state_opt_baseline["arithmetic_net_return_bp"],
                1,
            ),
            pct(state_opt_hard["active_review_hit_rate"], 1),
            number(state_opt_hard["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_hard["max_drawdown"]),
            pct(state_opt_hard["mean_gross_exposure"], 1),
        ],
        [
            "Matched baseline / pseudo-OOS",
            int(state_opt_walk_base["event_dates"]),
            number(state_opt_walk_base["arithmetic_net_return_bp"]),
            "100.0%",
            pct(state_opt_walk_base["active_review_hit_rate"], 1),
            number(state_opt_walk_base["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_walk_base["max_drawdown"]),
            pct(state_opt_walk_base["mean_gross_exposure"], 1),
        ],
        [
            "Constrained / pseudo-OOS",
            int(state_opt_walk_constrained["event_dates"]),
            number(state_opt_walk_constrained["arithmetic_net_return_bp"]),
            pct(
                state_opt_walk_constrained[
                    "net_return_retention_vs_matched_baseline"
                ],
                1,
            ),
            pct(state_opt_walk_constrained["active_review_hit_rate"], 1),
            number(state_opt_walk_constrained["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_walk_constrained["max_drawdown"]),
            pct(state_opt_walk_constrained["mean_gross_exposure"], 1),
        ],
        [
            "Max mean / pseudo-OOS",
            int(state_opt_walk_max["event_dates"]),
            number(state_opt_walk_max["arithmetic_net_return_bp"]),
            pct(state_opt_walk_max["net_return_retention_vs_matched_baseline"], 1),
            pct(state_opt_walk_max["active_review_hit_rate"], 1),
            number(state_opt_walk_max["active_trade_sharpe_sqrt4"]),
            signed_pct(state_opt_walk_max["max_drawdown"]),
            pct(state_opt_walk_max["mean_gross_exposure"], 1),
        ],
    ]
    story.append(
        table_caption(
            7,
            "Frozen soft-gate fixed and expanding pseudo-out-of-sample performance at 10 bp.",
            styles["table_caption"],
        )
    )
    story.append(
        table_from_rows(
            state_opt_rows,
            [1.48 * inch, 0.48 * inch, 0.58 * inch, 0.68 * inch, 0.64 * inch, 0.68 * inch, 0.62 * inch, text_width - 5.16 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=6.95,
        )
    )
    add_paragraphs(
        story,
        styles["body"],
        [
            f"At 10 bp the 25% soft gate retains {pct(state_opt_025['arithmetic_net_return_bp'] / state_opt_baseline['arithmetic_net_return_bp'], 1)} "
            f"of full-sample net profit, raises active hit from {pct(state_opt_baseline['active_review_hit_rate'], 1)} to "
            f"{pct(state_opt_025['active_review_hit_rate'], 1)}, and lowers maximum drawdown from "
            f"{signed_pct(state_opt_baseline['max_drawdown'])} to {signed_pct(state_opt_025['max_drawdown'])}. Its paired "
            f"return increment is {number(state_opt_025_paired['mean_paired_increment_bp'])} bp per review, block CI "
            f"[{number(state_opt_025_paired['block4_increment_ci_low_bp'])}, {number(state_opt_025_paired['block4_increment_ci_high_bp'])}], "
            f"and max-t p={number(state_opt_025_paired['block4_max_t_familywise_pvalue'], 3)}. It improves reliability by "
            "reducing exposure, not by increasing absolute profit.",
            f"In the 18-review expanding path, the constrained policy retains only "
            f"{pct(state_opt_walk_constrained['net_return_retention_vs_matched_baseline'], 1)}, below its frozen 80% target. "
            f"Its paired increment is {number(state_opt_walk_constrained_pair['mean_paired_increment_bp'])} bp per review "
            f"with max-t p={number(state_opt_walk_constrained_pair['block4_max_t_familywise_pvalue'], 3)}. Leave-one-review-out "
            f"selection chooses the 25% rule {pct(state_opt_loo_share, 1)} of the time. An independent reconstruction of "
            f"{int(state_opt_audit['security_rows'])} security rows reconciles all {int(state_opt_audit['review_dates'])} "
            f"dates to maximum error {float(state_opt_audit['maximum_date_reconciliation_error']):.2e}. The execution label "
            f"remains {state_opt_audit['execution_status']}.",
        ],
    )

    story.append(Paragraph("10    Limitations and Required Validation", styles["h1"]))
    limitations = [
        "<b>Small independent sample.</b> The study has 26 review dates. Stock-level rows share common shocks and cannot be treated as independent observations.",
        "<b>Post-hoc discovery.</b> The interval, ranking family, original state-filter family, and 80% blend were explored on the same sample. The later monotone soft-gate family was frozen before its first run, but the full 2020-2026 history was already visible. Block-adjusted familywise tests do not reject a return improvement at conventional levels, and the constrained pseudo-OOS path misses its 80% profit-retention target.",
        "<b>Regime concentration.</b> Most profits occur after 2023. The February 2023 comprehensive-quarterly-review transition is both a possible mechanism and a structural-break risk.",
        "<b>No future holdout.</b> August 2026 motivated the research. Removing it is a sensitivity check, not a genuine out-of-sample test.",
        "<b>Proxy execution.</b> Minute OHLCV endpoints contain no NBBO spread, depth, venue, queue, latency, rejects, or partial fills. The official auction print is not identified.",
        "<b>Incomplete forced-flow model.</b> Public lists omit old/new weights, NOS, FIF, product AUM, replication fractions, crossing, and cross-index netting. Standard-Small migrations can create offsetting benchmark flows.",
        "<b>Borrow and capacity.</b> The short leg has no historical locate/fee panel. The current capacity score is based on trailing liquidity proxies and does not model nonlinear impact.",
        "<b>Control limitations.</b> Four prior sessions do not control for contemporaneous earnings, market news, other index events, sector shocks, or month-end flows.",
        "<b>Options boundary.</b> No historical OPRA execution data support option P&amp;L. Underlying movement cannot substitute for an ask-in/bid-out straddle replay.",
        "<b>Top-10 artifact provenance.</b> The reported capped Top-10 ledgers reconcile internally, but their exact builder was a temporary, unversioned research script and the manifest lacks complete input and runner hashes. The durable repository runner applies a different sparse-date policy and produces slightly different results. The current figures therefore require generator promotion and a fully hashed rerun before independent reproduction.",
        "<b>Licensing.</b> Commercial use requires licensed MSCI constituent and pro-forma data plus legal review of redistribution restrictions.",
    ]
    for item in limitations:
        story.append(Paragraph(f"- {item}", styles["bullet"]))

    story.append(Paragraph("10.1    Production data gate", styles["h2"]))
    gate_rows = [
        ["Component", "Required production input", "Current status"],
        ["Event and weights", "Licensed point-in-time constituents, old/new weights, NOS, FIF, amendments, permanent IDs", "Public lists only"],
        ["Expected flow", "Tracker AUM, replication fractions, product overlap, crossing estimates", "Unavailable"],
        ["Equity execution", "TAQ NBBO/trades, official auction print, depth, venue rules, latency and queue replay", "Minute OHLCV proxy"],
        ["Closing imbalance", "NYSE imbalance feed and Nasdaq ITCH/NOII messages with deterministic timestamps", "Unavailable"],
        ["Short implementation", "Locate availability, borrow fees, recalls and fail policy", "Unavailable"],
        ["Option implementation", "OPRA NBBO/trades/size, contract master, IV/Greeks, exercise and fees", "Untested"],
        ["Validation", "Frozen protocol and next eight quarterly reviews as sequential holdouts", "No future holdout yet"],
    ]
    story.append(table_caption(8, "Minimum production and causal-validation inputs.", styles["table_caption"]))
    story.append(
        table_from_rows(
            gate_rows,
            [1.25 * inch, 3.75 * inch, text_width - 5.0 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.75,
        )
    )

    story.append(Paragraph("11    Conclusion", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The U.S. MSCI implementation sample contains a visible direction-signed price burst around 15:50. The "
            f"all-name basket earns {basis_points(burst_zero['mean_return_per_event'])} bp per review at the selected 0 bp "
            f"cost, while the exploratory impact top-10 earns {basis_points(top_zero['mean_return_per_event'])} bp with a "
            f"{number(top_zero['active_trade_sharpe_sqrt4'])} active-trade Sharpe and {signed_pct(top_zero['max_drawdown'])} event-step maximum "
            "drawdown. These figures are economically interesting but statistically and operationally incomplete. Window "
            "selection does not survive the block max-t threshold, the ranking lift is not familywise significant, and the "
            f"effect is concentrated after 2023. Arithmetic attribution assigns {number(leg_all['long_signed_share_pct'], 1)}% "
            f"of gross profit to addition longs and {number(leg_all['short_signed_share_pct'], 1)}% to deletion shorts. The "
            "weak early regime is primarily a deletion-leg sign failure and cross-leg cancellation, while the post-2023 "
            "sample has positive contributions from both legs.",
            f"The bounded 25% soft gate is useful as a risk-control challenger. At a 10 bp haircut it raises active-date "
            f"hit rate from {pct(state_opt_baseline['active_review_hit_rate'], 1)} to "
            f"{pct(state_opt_025['active_review_hit_rate'], 1)} and reduces maximum drawdown from "
            f"{signed_pct(state_opt_baseline['max_drawdown'])} to {signed_pct(state_opt_025['max_drawdown'])}, while retaining "
            f"{pct(state_opt_025['arithmetic_net_return_bp'] / state_opt_baseline['arithmetic_net_return_bp'], 1)} of baseline "
            f"arithmetic net profit. The constrained expanding path retains only "
            f"{pct(state_opt_walk_constrained['net_return_retention_vs_matched_baseline'], 1)}, below the frozen 80% target, "
            f"and its paired mean increment is {number(state_opt_walk_constrained_pair['mean_paired_increment_bp'])} bp per review. "
            "The reliability improvement comes from lower exposure; it does not establish incremental alpha.",
            "The correct decision is to freeze the rule and run it as a shadow strategy. Production consideration requires "
            "licensed point-in-time MSCI flows, message-level imbalance and quote replay, borrow data, realistic impact, and "
            "future sequential holdouts. Until those gates are passed, the paper documents a research candidate rather than "
            "a validated arbitrage product.",
        ],
    )

    story.append(PageBreak())
    story.append(Paragraph("A    Reproducible Implementation Checklist", styles["h1"]))
    checklist = [
        "Freeze the official review announcement, implementation date, amendments, and permanent security identifiers.",
        "Resolve Standard additions/deletions together with Small Cap migrations and overlapping product families.",
        "For the state filter, use only bars completed by 15:47:59.999 Eastern Time; reserve 15:48:59.999 for the strict theoretical boundary and research-only cutoff sensitivity.",
        "Compute the base impact-rank liquidity and volatility only from completed pre-event sessions; compute event-day state variables only through the 15:47 safe cutoff.",
        "Require at least three valid additions and three valid deletions; otherwise hold the review entirely in cash.",
        "Rank each side independently; select up to five names per side; break ties deterministically by ticker.",
        "Assign at most 10% of capital per name and at most 50% per leg; leave unused capacity in cash.",
        "Record the 15:49 start-stamped bar open as the entry proxy and the 15:50 start-stamped bar close as the exit proxy.",
        "Store gross contribution, selected transaction-cost scenario, net contribution, exposure, and fixed-notional attribution for every security.",
        "Aggregate to one return per review date before inference; retain all cash reviews as zero-return observations.",
        "Report event-step equity, cash-funded calendar CAGR, event-step maximum drawdown, Active-trade Sharpe sqrt(4), Scheduled Event Sharpe, block intervals, and familywise p-values.",
        "Do not change the rule during the next eight quarterly sequential holdouts; log deviations before observing outcomes.",
    ]
    for index, item in enumerate(checklist, 1):
        story.append(Paragraph(f"{index}. {item}", styles["bullet"]))

    story.append(Paragraph("B    Research Ledger and Audit Trail", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "The 26-row review ledger is the schedule spine. It retains every active and cash decision, eligible side counts, "
            "selected names, gross and net exposure, modeled cost, event-step equity, and drawdown. The 161-row security "
            "ledger contains date, side, side rank, ticker, MSCI change type, proxy score, target weight, bar-start timestamps, "
            "entry open, exit close, gross contribution, 0 bp modeled cost, net contribution, fixed USD 1 million attribution, "
            "feature cutoff, liquidity as-of date, fill model, information cutoff, and a point-in-time-valid flag.",
            "Dollar columns use a fresh fixed USD 1 million notional for each review and therefore sum arithmetically. The "
            "equity curve compounds capital through time, so the two dollar totals differ slightly. The security log has no "
            "placeholder rows for nine cash reviews; the date ledger supplies the complete schedule.",
        ],
    )
    ledger_rows = [
        ["Audit invariant", "Verified condition"],
        ["Point-in-time selection", "Feature cutoff and liquidity as-of strictly precede implementation date"],
        ["Information clock", "Production state features end at 15:47:59.999; 15:48 is research-only sensitivity"],
        ["Weight integrity", "Absolute name weight <=10%; sides ranked independently; cash gate enforced"],
        ["P&amp;L reconciliation", "Security contributions sum to each active review within floating-point tolerance"],
        ["Schedule integrity", "26 unique dates: 17 traded and 9 cash"],
        ["Cost integrity", "Primary modeled cost exactly 0 bp; 10 bp fields retained only as sensitivity"],
        ["Leg attribution", "161 security rows reconcile to 26 review returns; maximum date error below 5e-16"],
        ["Top-10 provenance", "Frozen ledgers reconcile; exact temporary builder is not versioned and must be promoted before replication"],
        ["Test status", "48 automated tests pass; run archived in outputs/baseline/paper_qa_test_log.txt; execution realism remains untested"],
    ]
    story.append(table_caption(9, "Research-ledger invariants.", styles["table_caption"]))
    story.append(
        table_from_rows(
            ledger_rows,
            [1.62 * inch, text_width - 1.62 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=8.1,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("C    Long and Short Profit Attribution", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The active portfolio budgets up to 50% of capital to addition longs and 50% to deletion shorts. Realized "
            f"arithmetic profit is less symmetric: additions contribute {number(leg_all['long_total_bp'])} bp, or "
            f"{number(leg_all['long_signed_share_pct'], 2)}% of the {number(leg_all['arithmetic_total_bp'])} bp total; "
            f"deletion shorts contribute {number(leg_all['short_total_bp'])} bp, or "
            f"{number(leg_all['short_signed_share_pct'], 2)}%. These are additive contribution shares of the fixed-capital "
            "review ledger, not shares of compounded wealth.",
            f"The regime split is economically sharper. During 2020-2022, addition longs contribute "
            f"{number(leg_early['long_total_bp'])} bp while deletion shorts detract {number(abs(leg_early['short_total_bp']))} "
            f"bp. The signed shares are therefore {number(leg_early['long_signed_share_pct'], 1)}% and "
            f"{number(leg_early['short_signed_share_pct'], 1)}%. From February 2023 onward both legs are positive, with "
            f"{number(leg_late['long_total_bp'])} bp from longs and {number(leg_late['short_total_bp'])} bp from shorts.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "topn_capped10_event_leg_attribution.png",
            "Addition-long and deletion-short contribution for every scheduled implementation close. Black dots are "
            "total review return; x marks are cash reviews. The shaded period is 2020-2022 and the dashed line marks the "
            "February 2023 regime boundary. Source: frozen 161-row security ledger reconciled to the 26-date schedule.",
            10,
            styles["caption"],
            text_width,
            3.02 * inch,
        )
    )

    attribution_rows = [["Slice", "Reviews S/A/C", "Long bp", "Short bp", "Total bp", "Signed L/S share", "Both + / disagree", "Net mean @10bp"]]
    for _, row in leg_regime.loc[leg_regime["period"].isin(["All", "2020-2022", "2023+"])].iterrows():
        attribution_rows.append(
            [
                row["period"],
                f"{int(row['scheduled_reviews'])}/{int(row['active_reviews'])}/{int(row['cash_reviews'])}",
                f"{row['long_total_bp']:+.2f}",
                f"{row['short_total_bp']:+.2f}",
                f"{row['arithmetic_total_bp']:+.2f}",
                f"{row['long_signed_share_pct']:.1f}% / {row['short_signed_share_pct']:.1f}%",
                f"{int(row['both_legs_positive'])} / {int(row['leg_sign_disagreement'])}",
                f"{row['net_10bp_mean_per_scheduled_bp']:+.2f} bp",
            ]
        )
    story.append(table_caption(10, "Arithmetic leg attribution by regime.", styles["table_caption"]))
    story.append(
        table_from_rows(
            attribution_rows,
            [0.82 * inch, 0.82 * inch, 0.67 * inch, 0.67 * inch, 0.67 * inch, 1.12 * inch, 0.90 * inch, 1.27 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.55,
        )
    )
    story.append(
        Paragraph(
            "<i>Notes:</i> S/A/C denotes scheduled/active/cash reviews. Signed contribution shares may exceed 100% or "
            "be negative when one leg offsets the other. The 10 bp column applies the recorded exposure-aware round-trip "
            "haircut; it is a sensitivity, not an observed fill cost.",
            styles["note"],
        )
    )

    seasonal_rows = [["Calendar quarter", "Reviews S/A/C", "Long bp", "Short bp", "Total bp", "Mean / scheduled", "Active hit L/S"]]
    for _, row in quarter_seasonality.iterrows():
        seasonal_rows.append(
            [
                row["period"],
                f"{int(row['scheduled_reviews'])}/{int(row['active_reviews'])}/{int(row['cash_reviews'])}",
                f"{row['long_total_bp']:+.2f}",
                f"{row['short_total_bp']:+.2f}",
                f"{row['arithmetic_total_bp']:+.2f}",
                f"{row['gross_mean_per_scheduled_bp']:+.2f} bp",
                f"{100.0 * row['long_active_hit_rate']:.0f}% / {100.0 * row['short_active_hit_rate']:.0f}%",
            ]
        )
    story.append(table_caption(11, "Calendar-quarter seasonality across the full sample.", styles["table_caption"]))
    story.append(
        table_from_rows(
            seasonal_rows,
            [0.87 * inch, 0.93 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 1.02 * inch, 1.00 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.65,
        )
    )
    story.append(
        Paragraph(
            "<i>Notes:</i> Q1-Q4 pool only six or seven scheduled reviews each and have unequal active counts. These "
            "figures describe seasonality; they are not independent quarterly subsamples.",
            styles["note"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("D    Review-by-Review Performance Ledger", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            "Table 11 reports all 26 nonempty analyzed reviews from May 2020 through August 2026, including nine cash "
            "decisions. The February 2020 public review had zero U.S. Standard additions and deletions and lies outside the "
            "return sample. Long and short columns are "
            "portfolio contributions in basis points of total capital. The eligible counts are the point-in-time-valid "
            "additions and deletions before the three-per-side gate; selected counts are shown separately.",
        ],
    )
    event_rows = [["Quarter", "Date", "Status", "n L/S", "Long", "Short", "Gross", "Net10", "Abs L/S", "Elig A/D", "Largest absolute name"]]
    for _, row in event_attribution.iterrows():
        absolute_share = "-"
        if pd.notna(row["absolute_long_share_pct"]):
            absolute_share = f"{row['absolute_long_share_pct']:.0f}%/{row['absolute_short_share_pct']:.0f}%"
        contributor = "-" if pd.isna(row["top_contributor"]) else str(row["top_contributor"])
        event_rows.append(
            [
                row["quarter"],
                pd.to_datetime(row["effective_close_date"]).strftime("%Y-%m-%d"),
                row["status"],
                f"{int(row['n_long'])}/{int(row['n_short'])}",
                f"{row['long_contribution_bp']:+.2f}",
                f"{row['short_contribution_bp']:+.2f}",
                f"{row['gross_return_bp']:+.2f}",
                f"{row['net_10bp_bp']:+.2f}",
                absolute_share,
                f"{int(row['eligible_adds'])}/{int(row['eligible_deletes'])}",
                contributor,
            ]
        )
    story.append(table_caption(12, "All 26 nonempty analyzed MSCI implementation closes in the Top-10 ledger.", styles["table_caption"]))
    story.append(
        table_from_rows(
            event_rows,
            [0.48 * inch, 0.66 * inch, 0.50 * inch, 0.44 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch, 0.78 * inch, 0.62 * inch, 1.14 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=6.55,
        )
    )
    story.append(
        Paragraph(
            "<i>Notes:</i> Gross = Long + Short. Net10 applies the 10 bp exposure-aware sensitivity. Abs L/S divides "
            "the absolute leg contributions on that date and is undefined for cash reviews. Largest absolute name may be "
            "a gain or loss. Minute-bar endpoints remain research fill proxies.",
            styles["note"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("E    Why the 2020-2022 Effect Is Weak", styles["h1"]))
    add_paragraphs(
        story,
        styles["body"],
        [
            f"The early Top-10 estimate is weak rather than literally zero: {number(leg_early['gross_mean_per_scheduled_bp'])} "
            f"bp per scheduled review across 11 quarters, with only seven active reviews and a four-review block interval "
            f"of [{basis_points(top_early_zero['block_bootstrap_mean_ci_low'])}, "
            f"{basis_points(top_early_zero['block_bootstrap_mean_ci_high'])}] bp. A 10 bp haircut changes the mean to "
            f"{number(leg_early['net_10bp_mean_per_scheduled_bp'])} bp. Four cash reviews and cross-leg cancellation therefore "
            "consume most of the gross signal.",
            "Weak signed return does not imply missing mechanical flow. NYSE reports that the May 29, 2020 MSCI rebalance "
            "closed 1.55 billion shares and approximately USD 60 billion, 21.2% of that day's NYSE-listed volume [20]. "
            "Removing that review lowers rather than improves the early result, so May 2020 actually supports the early estimate.",
            f"The central accounting failure is the deletion leg. Addition longs earn {number(leg_early['long_total_bp'])} "
            f"bp, deletion shorts lose {number(abs(leg_early['short_total_bp']))} bp, and the active-date leg correlation is "
            f"{number(leg_early['active_leg_correlation'], 3)}. On 2022-05-31 alone, longs add +43.57 bp while shorts lose "
            "42.54 bp, leaving only +1.03 bp. The two legs are simultaneously positive on two of seven active early dates, "
            "versus eight of ten after the February 2023 breakpoint.",
            "Migration mechanics give a plausible reason for the sign failure. In the all-name early sample, 95 of 99 "
            "deletions are down-to-small migrations, and 66 of 105 additions are up-from-small migrations. Standard-index "
            "selling can be offset by Small Cap tracker buying, so a public Standard add/delete label is not a net-flow "
            "estimate. After first averaging within each review date, early down-to-small deletion shorts average -6.26 bp, "
            "versus +21.12 bp after 2023; new-to-Standard additions average +8.13 versus +24.85 bp, and up-from-small "
            "additions +1.34 versus +25.59 bp. These pooled-regime summaries are descriptive and have no formal clustered "
            "breakpoint test. Testing the mechanism requires old/new weights, NOS, FIF, Standard and Small Cap tracker AUM, "
            "and cross-index netting.",
            "MSCI implemented its first comprehensive quarterly index review at the February 28, 2023 close and began full "
            "quarterly NOS/FIF reviews in May 2023 [16-18]. February/August coverage expands from 3.6 to 14.25 all-name events "
            "per review in this sample. Comparable May/November reviews also improve, so QCIR explains sparse early quarters "
            "and a change in event composition, but it does not by itself explain the full return break.",
            f"The exact 15:50 signed minute changes from {number(clock_early_1550['mean_signed_bar_return_bp'])} bp in "
            f"2020-2022 to {number(clock_late_1550['mean_signed_bar_return_bp'])} bp after the breakpoint; mean 15:50 volume "
            f"share rises from {number(clock_early_1550['mean_minute_volume_share_pct'])}% to "
            f"{number(clock_late_1550['mean_minute_volume_share_pct'])}%. This is consistent with more concentrated price "
            "discovery, but minute OHLCV cannot identify NOII causality [21]. Nasdaq moved its NOII start to 15:50 in 2018, and "
            "NYSE's D-order feed inclusion moved to 15:50 in August 2024. NYSE also introduced new floor-broker connectivity "
            "in late 2022 and subsequently reported a higher D-order share [22]. We find no common NYSE/Nasdaq 2023 "
            "imbalance-dissemination-time change, but a venue-composition channel cannot be excluded without venue splits "
            "[19,22,23].",
            f"One-off exclusions refine the conclusion. Removing May 2020 lowers the early Top-10 mean to +2.48 bp and "
            f"the all-name mean to -1.36 bp; removing May 2023, when MSCI also implemented one-time GICS revisions [24], "
            f"leaves +17.41 and +23.85 bp, respectively. Removing all 2026 "
            f"observations leaves the broad all-name mean at {number(broad_post_ex_2026['mean_bp_per_scheduled_review'])} bp, "
            f"but compresses the Top-10 mean to {number(top_post_ex_2026['mean_bp_per_scheduled_review'])} bp, close to the "
            f"early +5.72 bp estimate. Moreover, 2026 supplies "
            f"{100.0 * year_2026['arithmetic_total_bp'] / leg_all['arithmetic_total_bp']:.1f}% of total Top-10 arithmetic "
            "profit. The all-name regime pattern is more persistent; the concentrated Top-10 result is recent and needs "
            "future holdout evidence.",
        ],
    )
    story.append(
        figure(
            args.output_dir / "charts" / "topn_capped10_early_regime_diagnostics.png",
            "Regime decomposition. Left: Top-10 long and short mean contribution per scheduled review. Center: the same "
            "decomposition for the all-name basket. Right: closing-volume concentration. The deletion leg, rather than "
            "the addition leg, accounts for most of the early-to-late return change. Source: audited portfolio and volume outputs.",
            11,
            styles["caption"],
            text_width,
            2.55 * inch,
        )
    )

    driver_rows = [["Diagnostic", "2020-22", "2023+", "Difference", "Unit", "Interpretation"]]
    for _, row in early_drivers.iterrows():
        driver_rows.append(
            [
                row["metric"],
                f"{row['2020_2022']:.2f}",
                f"{row['2023_plus']:.2f}",
                f"{row['difference']:+.2f}",
                row["unit"],
                row["interpretation"],
            ]
        )
    story.append(table_caption(13, "Observed drivers of the early/late difference.", styles["table_caption"]))
    story.append(
        table_from_rows(
            driver_rows,
            [1.68 * inch, 0.59 * inch, 0.59 * inch, 0.64 * inch, 0.70 * inch, 2.18 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=6.85,
        )
    )

    migration_rows = [["Regime", "Side", "Change type", "Dates", "Names", "Mean date bp", "Date hit"]]
    for _, row in broad_change_type.iterrows():
        migration_rows.append(
            [
                row["regime"],
                "Long" if row["action"] == "add" else "Short",
                row["change_type"],
                int(row["review_dates"]),
                int(row["security_events"]),
                f"{row['mean_date_directional_return_bp']:+.2f}",
                f"{100.0 * row['date_hit_rate']:.1f}%",
            ]
        )
    story.append(table_caption(14, "Date-balanced all-name signed return by migration type.", styles["table_caption"]))
    story.append(
        table_from_rows(
            migration_rows,
            [0.84 * inch, 0.58 * inch, 1.42 * inch, 0.52 * inch, 0.52 * inch, 1.13 * inch, 0.78 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.45,
        )
    )

    review_rows = [["Regime", "Review class", "Dates", "Names/date", "Long bp", "Short bp", "Gross bp"]]
    for _, row in review_type.iterrows():
        review_rows.append(
            [
                row["regime"],
                row["review_class"],
                int(row["review_dates"]),
                f"{row['mean_security_events']:.2f}",
                f"{row['mean_long_contribution_bp']:+.2f}",
                f"{row['mean_short_contribution_bp']:+.2f}",
                f"{row['mean_gross_return_bp']:+.2f}",
            ]
        )
    story.append(table_caption(15, "All-name return by review class before and after QCIR.", styles["table_caption"]))
    story.append(
        table_from_rows(
            review_rows,
            [0.88 * inch, 1.30 * inch, 0.55 * inch, 0.82 * inch, 0.82 * inch, 0.82 * inch, 0.82 * inch],
            styles["table_body"],
            styles["table_header"],
            font_size=7.35,
        )
    )
    story.append(
        Paragraph(
            "<i>Interpretation boundary:</i> The data document a QCIR-aligned descriptive return break, a date-balanced "
            "migration-leg sign change, and stronger 15:50 concentration after 2023. No formal clustered breakpoint test "
            "is claimed, and actual net forced flow is unidentified. Causal validation requires licensed delta weights and "
            "tracker AUM plus message-level NOII, NBBO, venue and auction data.",
            styles["note"],
        )
    )

    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("References", styles["h1"]))
    references = [
        "[1] Chang, X., Luo, J., Peng, J., Qian, S., and Tan, C. W. Index-tracking rigidity and arbitrage opportunities in MSCI index reconstitutions. <i>Pacific-Basin Finance Journal</i>, 93, 2025.",
        "[2] Bogousslavsky, V., and Muravyev, D. Who Trades at the Close? Implications for Price Discovery and Liquidity. <i>Journal of Financial Markets</i>, 66:100852, 2023. doi:10.1016/j.finmar.2023.100852.",
        "[3] Nasdaq Economic Research. How Much Does the MOC Imbalance Matter? September 27, 2019.",
        "[4] New York Stock Exchange. Closing Auction Order Impact. NYSE Data Insights, 2024.",
        "[5] Goyal, A., Jegadeesh, N., and Wu, Y. Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets: A Benchmark for Cost of Trading on Anomalies. <i>Journal of Financial and Quantitative Analysis</i>, 2026.",
        "[6] MSCI. Global Standard Index Review public lists and quarterly-review announcements, February/May/August/November 2020-2026.",
        "[7] New York Stock Exchange. Closing Auction Timetable and NYSE Pillar Order Imbalances Client Specification, version 2.2n.",
        "[8] Nasdaq Trader. Nasdaq Opening and Closing Cross Frequently Asked Questions; TotalView-ITCH Data Feed Specification.",
        "[9] MarketData. Stock Candles API Documentation. Minute-candle timestamp and OHLCV schema.",
        "[10] Bailey, D. H., Borwein, J., Lopez de Prado, M., and Zhu, Q. J. The Probability of Backtest Overfitting. <i>Journal of Computational Finance</i>, 20(4):39-69, 2017.",
        "[11] Alexiou, L., Goyal, A., Kostakis, A., and Rompolis, L. Pricing Event Risk: Evidence from Concave Implied Volatility Curves. <i>Review of Finance</i>, 29(4):963-1007, 2025. doi:10.1093/rof/rfaf016.",
        "[12] Options Price Reporting Authority. OPRA consolidated option quotes and trades; historical data specifications.",
        "[13] New York Stock Exchange. Daily TAQ Trades and Quotes, TAQ Order Imbalances, and TAQ Integrated Feed product documentation.",
        "[14] Comerton-Forde, C., and Rindi, B. Trading at the Close. Working paper, 2021.",
        "[15] MSCI. Index Terms of Use. Licensing and redistribution terms for index data.",
        "[16] MSCI. Transition to Quarterly Comprehensive Index Reviews in MSCI Global Investable Market Indexes. November 2022. <link href=\"https://app2.msci.com/webapp/index_ann/DocGet?format=html&amp;lang=en&amp;pub_key=Nlk1P2UjZO4%3D\" color=\"blue\">Official announcement</link>.",
        "[17] MSCI. Results of the February 2023 Quarterly Index Review for the MSCI Equity Indexes. February 2023. <link href=\"https://app2.msci.com/webapp/index_ann/DocGet?format=html&amp;lang=en&amp;pub_key=QXmTJkx1PjA%3D\" color=\"blue\">Official announcement</link>.",
        "[18] MSCI. Consultation on Introducing Quarterly Comprehensive Index Reviews. 2021. <link href=\"https://www.msci.com/documents/1296102/22675532/Consultation%2Bon%2BIntroducing%2BQuarterly%2BComprehensive%2BIndex%2BReviews.pdf/85987bdb-388d-51f6-79d6-6ea118c899ff?t=1614683476145\" color=\"blue\">Methodology consultation</link>.",
        "[19] Nasdaq Economic Research. Automation and Information Produce Efficient Price Discovery in Nasdaq's Auction Process. <link href=\"https://www.nasdaq.com/articles/automation-and-information-produce-efficient-price-discovery-in-nasdaqs-auction-process\" color=\"blue\">Online</link>.",
        "[20] New York Stock Exchange. NYSE Floor Unrivaled in Key Auction Events. 2020. <link href=\"https://www.nyse.com/data-insights/nyse-floor-unrivaled-in-key-auction-events\" color=\"blue\">Online</link>.",
        "[21] New York Stock Exchange. Closing Auction Liquidity Momentum Around Rebalances. <link href=\"https://www.nyse.com/data-insights/closing-auction-liquidity-momentum-around-rebalances\" color=\"blue\">Online</link>.",
        "[22] New York Stock Exchange. NYSE Closing Auction Dynamics 2023. <link href=\"https://beta.nyse.com/data-insights/nyse-closing-auction-dynamics-2023\" color=\"blue\">Online</link>.",
        "[23] New York Stock Exchange. Closing Auction Price Discovery Opportunities Reach New Highs. 2024. <link href=\"https://www.nyse.com/data-insights/nyse-closing-auction-price-discovery-opportunities-reach-new-highs\" color=\"blue\">Online</link>.",
        "[24] MSCI. Implementation of 2023 GICS Changes in the MSCI Indexes. <link href=\"https://www.msci.com/documents/1296102/29559863/Implementation_of_2023_GICS_Changes.pdf/3b8bf06e-fe28-42f3-f6bc-73681bd3ec29?t=1666901607879\" color=\"blue\">Methodology notice</link>.",
        "[25] Pavlova, A., and Sikorskaya, T. Benchmarking Intensity. <i>Review of Financial Studies</i>, 36(3):859-903, 2023. doi:10.1093/rfs/hhac055. <link href=\"https://academic.oup.com/rfs/article/36/3/859/6665907\" color=\"blue\">Publisher page</link>.",
        "[26] Gao, L., Han, Y., Li, S. Z., and Zhou, G. Market Intraday Momentum. <i>Journal of Financial Economics</i>, 129(2):394-414, 2018. doi:10.1016/j.jfineco.2018.05.009. <link href=\"https://www.sciencedirect.com/science/article/pii/S0304405X18301351\" color=\"blue\">Publisher page</link>.",
        "[27] Heston, S. L., Korajczyk, R. A., and Sadka, R. Intraday Patterns in the Cross-section of Stock Returns. <i>Journal of Finance</i>, 65(4):1369-1407, 2010. doi:10.1111/j.1540-6261.2010.01573.x. <link href=\"https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.2010.01573.x\" color=\"blue\">Publisher page</link>.",
    ]
    for reference in references:
        story.append(Paragraph(reference, styles["reference"]))

    doc.build(story)
    print(f"paper={args.paper_path} bytes={args.paper_path.stat().st_size}")


if __name__ == "__main__":
    main()
