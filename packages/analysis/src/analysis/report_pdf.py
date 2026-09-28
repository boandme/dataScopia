"""The DataScopia PDF report.

Structure follows the two reporting conventions the product is built around:

  * data-analysis / policy-report practice (UNICEF/WHO style): title block, executive summary with
    key findings and recommended actions, introduction, methods, results, discussion, conclusions and
    recommendations, references, appendices; numbered, captioned tables and figures that are each
    referenced in the text; plain language; methods transparent enough to replicate.
  * the scientific-manuscript convention: an abstract-style summary (problem, purpose, methods,
    results, interpretation, implication), If-Then hypotheses, methods in the past tense, *results
    kept strictly separate from interpretation*, mean +/- SD with significance and effect size for
    every comparison, captions that say what a figure means, and consistent author-year references.

It is deliberately compact (about 6-10 pages): the emphasis is on the analysis - descriptive
statistics, tests with effect sizes, a baseline model, quality audit and figures - not on prose.
"""

from __future__ import annotations

import base64
import io
import math
import re
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

from . import report_stats as RS
from . import schema as schema_mod
from .profiler import analyze_dataframe
from .report import generate_chart_narrative

# Greek letters are spelled out: the base-14 fonts have no Greek glyphs, and bundling a font would make
# the export depend on files that may not exist on the host.
RHO, ETA, CHI, ALPHA = "<i>rho</i>", "<i>eta</i>", "<i>chi</i>", "alpha"
ETA2 = f"{ETA}<super>2</super>"
CHI2 = f"{CHI}<super>2</super>"

ROLE_NAMES = {
    "numeric": "numeric measure", "categorical": "categorical", "binary": "binary", "binary_numeric": "binary (0/1)",
    "ordinal": "ordinal", "ordinal_numeric": "ordinal (numeric)", "identifier": "identifier", "year": "year",
    "datetime": "date/time", "geo_country": "country", "geo_state": "US state", "geo_lat": "latitude", "geo_lon": "longitude",
    "text": "free text", "constant": "constant", "empty": "empty", "unknown": "unknown",
}
LICENSE_LEVEL_TEXT = {"open": "open license", "conditions": "license with conditions", "restricted": "restrictive or custom license", "unknown": "no license recorded"}


def _safe(text: Any) -> str:
    """Helvetica is WinAnsi-only: anything outside it becomes '?' instead of a missing-glyph box."""
    return str(text if text is not None else "").encode("cp1252", "replace").decode("cp1252")


def esc(text: Any) -> str:
    return xml_escape(_safe(text))


def first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}%"


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


def _clean_description(text: str, max_chars: int = 650) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


# ============================================================================= native figures

def _diverging(value: float):
    from reportlab.lib import colors

    stops = [(-1.0, "#2f7bff"), (-0.5, "#3a5fb0"), (0.0, "#e9ecf3"), (0.5, "#e8907d"), (1.0, "#d9364f")]
    v = max(-1.0, min(1.0, value))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if a <= v <= b:
            t = (v - a) / (b - a)
            c1, c2 = colors.HexColor(ca), colors.HexColor(cb)
            return colors.Color(c1.red + (c2.red - c1.red) * t, c1.green + (c2.green - c1.green) * t, c1.blue + (c2.blue - c1.blue) * t)
    return colors.HexColor("#e9ecf3")


def correlation_figure(matrix: pd.DataFrame, width: float):
    from reportlab.graphics.shapes import Drawing, Group, Rect, String
    from reportlab.lib import colors

    labels = [_safe(c)[:14] for c in matrix.columns]
    n = len(labels)
    left, top = 78, 46
    cell = min(34.0, (width - left - 10) / n)
    height = top + cell * n + 12
    d = Drawing(width, height)
    for j, label in enumerate(labels):  # column labels, rotated
        g = Group(String(0, 0, label, fontName="Helvetica", fontSize=6.6, fillColor=colors.HexColor("#3b4358"), textAnchor="start"))
        g.translate(left + j * cell + cell / 2 - 1, height - top + 5)
        g.rotate(50)
        d.add(g)
    for i, label in enumerate(labels):
        y = height - top - (i + 1) * cell
        d.add(String(left - 4, y + cell / 2 - 2.3, label, fontName="Helvetica", fontSize=6.6, fillColor=colors.HexColor("#3b4358"), textAnchor="end"))
        for j in range(n):
            v = float(matrix.iloc[i, j]) if not np.isnan(matrix.iloc[i, j]) else 0.0
            d.add(Rect(left + j * cell, y, cell - 1, cell - 1, fillColor=_diverging(v), strokeColor=None))
            d.add(String(left + j * cell + cell / 2 - 0.5, y + cell / 2 - 2.2, f"{v:.2f}".replace("0.", ".").replace("-.", "-."), fontName="Helvetica", fontSize=5.8,
                         fillColor=colors.white if abs(v) > 0.55 else colors.HexColor("#1a2233"), textAnchor="middle"))
    return d


def missing_figure(rows: list[tuple[str, float]], width: float):
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib import colors

    bar_h, gap, left = 11, 5, 118
    height = len(rows) * (bar_h + gap) + 14
    d = Drawing(width, height)
    scale = (width - left - 46) / max(5.0, max(v for _, v in rows))
    for i, (name, value) in enumerate(rows):
        y = height - (i + 1) * (bar_h + gap)
        d.add(String(left - 6, y + 2.5, _safe(name)[:24], fontName="Helvetica", fontSize=7.4, fillColor=colors.HexColor("#3b4358"), textAnchor="end"))
        d.add(Rect(left, y, max(1.5, value * scale), bar_h, fillColor=colors.HexColor("#5546e0" if value < 20 else "#e2409b" if value < 50 else "#d9364f"), strokeColor=None))
        d.add(String(left + max(1.5, value * scale) + 4, y + 2.5, f"{value:.1f}%", fontName="Helvetica-Bold", fontSize=7.2, fillColor=colors.HexColor("#1a2233")))
    return d


def histogram_grid(series: dict[str, pd.Series], width: float):
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors

    names = list(series)[:6]
    cols = 3
    rows = (len(names) + cols - 1) // cols
    pw, ph, gap = (width - 20) / cols, 74, 26
    height = rows * (ph + gap) + 8
    d = Drawing(width, height)
    palette = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
    for k, name in enumerate(names):
        r, c = divmod(k, cols)
        x0 = 10 + c * pw + 4
        y0 = height - (r + 1) * (ph + gap) + 4
        values = series[name].dropna().to_numpy()
        counts, edges = np.histogram(values, bins=18)
        peak = counts.max() or 1
        bw = (pw - 12) / len(counts)
        d.add(String(x0, y0 + ph + 8, _safe(name)[:26], fontName="Helvetica-Bold", fontSize=7.6, fillColor=colors.HexColor("#1a2233")))
        for i, cnt in enumerate(counts):
            d.add(Rect(x0 + i * bw, y0, bw - 0.6, ph * cnt / peak, fillColor=colors.HexColor(palette[k % len(palette)]), strokeColor=None))
        span = edges[-1] - edges[0] or 1.0
        for value, dash in ((float(np.mean(values)), None), (float(np.median(values)), [2, 2])):
            xm = x0 + (value - edges[0]) / span * bw * len(counts)
            line = Line(xm, y0, xm, y0 + ph, strokeColor=colors.HexColor("#1a2233"), strokeWidth=0.9)
            if dash:
                line.strokeDashArray = dash
            d.add(line)
        d.add(Line(x0, y0, x0 + bw * len(counts), y0, strokeColor=colors.HexColor("#9aa3b5"), strokeWidth=0.6))
        d.add(String(x0, y0 - 9, RS.fmt_num(float(edges[0])), fontName="Helvetica", fontSize=6.2, fillColor=colors.HexColor("#5b6472")))
        d.add(String(x0 + bw * len(counts), y0 - 9, RS.fmt_num(float(edges[-1])), fontName="Helvetica", fontSize=6.2, fillColor=colors.HexColor("#5b6472"), textAnchor="end"))
    return d


# ============================================================================= options

THEMES = {  # band, accent (headings), table header, highlight
    "indigo": ("#1e1b4b", "#4338ca", "#312e81", "#ffd23a"),
    "teal": ("#0b3b3c", "#0f766e", "#115e59", "#fbbf24"),
    "slate": ("#1f2937", "#334155", "#1e293b", "#38bdf8"),
    "crimson": ("#4c0519", "#be123c", "#881337", "#fbbf24"),
    "forest": ("#14352a", "#15803d", "#166534", "#fde047"),
}
DEFAULT_SECTIONS = {"introduction": True, "methods": True, "quality": True, "results": True, "discussion": True, "conclusions": True, "references": True, "appendices": True}


def normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Validated, defaulted report options (the browser may send anything)."""
    o = options if isinstance(options, dict) else {}
    sections = {**DEFAULT_SECTIONS, **{k: bool(v) for k, v in (o.get("sections") or {}).items() if k in DEFAULT_SECTIONS}}
    text = lambda key, limit: str(o.get(key) or "").strip()[:limit]  # noqa: E731
    return {
        "title": text("title", 140), "subtitle": text("subtitle", 200), "author": text("author", 120), "organization": text("organization", 120),
        "notes": text("notes", 1500), "footer": text("footer", 120),
        "detail": o.get("detail") if o.get("detail") in {"brief", "standard", "deep"} else "deep",
        "theme": o.get("theme") if o.get("theme") in THEMES else "indigo",
        "page_size": "a4" if o.get("page_size") == "a4" else "letter",
        "font_size": o.get("font_size") if o.get("font_size") in {"compact", "normal", "large"} else "normal",
        "sections": sections,
        "include_charts": o.get("include_charts", True) is not False,
        "include_hypotheses": o.get("include_hypotheses", True) is not False,
        "include_evidence": o.get("include_evidence", True) is not False,
        "include_figures_native": o.get("include_figures_native", True) is not False,
    }


# ============================================================================= builder

def build_report_pdf(
    dataset_name: str,
    df: pd.DataFrame,
    analysis: dict[str, Any],
    kaggle_description: str | None,
    charts: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    dataset_info: dict[str, Any] | None = None,
    transform_log: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import CondPageBreak, Flowable, Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from . import report_deep as D

    opt = normalize_options(options)
    sec_on = opt["sections"]
    detail = opt["detail"]
    is_deep, is_std = detail == "deep", detail in {"standard", "deep"}
    band_hex, accent_hex, head_hex, hilite_hex = THEMES[opt["theme"]]
    PAGE = A4 if opt["page_size"] == "a4" else LETTER

    dataset_info = dataset_info or {}
    transform_log = [t for t in (transform_log or []) if isinstance(t, dict) and t.get("description")]
    kaggle_info = dataset_info.get("kaggle_info") if isinstance(dataset_info.get("kaggle_info"), dict) else {}
    license_info = dataset_info.get("license_info") if isinstance(dataset_info.get("license_info"), dict) else {}
    sample = analysis.get("sample") or dataset_info.get("sample") or {}

    # Re-derive everything from the dataframe itself so the report always matches the data (and any
    # cleaning applied), not whatever the browser last held.
    fresh = analyze_dataframe(df, dataset_name=dataset_name, sample_info=sample or None)
    schema = schema_mod.build_schema(df)
    profiles = schema["profiles"]
    snapshot = fresh["research_snapshot"]
    quality = fresh["data_quality"]
    recs = fresh.get("recommendations") or []
    targets = fresh.get("target_details") or []
    target = targets[0] if targets else None
    rows_n, cols_n = df.shape
    S = RS.compute_all(df, fresh, schema)
    desc, cats, corr, groups, tassoc, trend, model = (S["descriptives"], S["categorical"], S["correlations"], S["groups"],
                                                        S["target"], S["trend"], S["model"])
    deep = D.compute_deep(df, fresh, schema, S) if is_std else {"corr": {"strongest": [], "nulls": [], "vif": [], "clusters": [], "isolated": [], "n_pairs": 0, "share_weak": 0.0},
                                                               "trends": None, "dist": [], "cats": [], "missing": None, "posthoc": []}
    dc = deep["corr"]
    trends_deep = (deep["trends"] or {}).get("series", [])

    # ---------------------------------------------------------------- styles
    ink, muted, accent, rule = colors.HexColor("#1a2233"), colors.HexColor("#5b6472"), colors.HexColor(accent_hex), colors.HexColor("#d7dce3")
    fs = {"compact": 8.8, "normal": 9.4, "large": 10.4}[opt["font_size"]]
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=fs, leading=fs * 1.45, textColor=ink, alignment=TA_LEFT, spaceAfter=7)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=11, textColor=muted, spaceAfter=4)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=ink, spaceBefore=16, spaceAfter=7, keepWithNext=1)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.6, leading=13, textColor=accent, spaceBefore=10, spaceAfter=4, keepWithNext=1)
    h3 = ParagraphStyle("H3", parent=body, fontName="Helvetica-BoldOblique", fontSize=fs, textColor=ink, spaceBefore=3, spaceAfter=2, keepWithNext=1)
    cap = ParagraphStyle("Cap", parent=body, fontSize=8.2, leading=11, spaceBefore=4, spaceAfter=6)
    tcap = ParagraphStyle("TCap", parent=cap, spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold", keepWithNext=1)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.9, leading=9.8, spaceAfter=0)
    cell_r = ParagraphStyle("CellR", parent=cell, alignment=2)
    cell_h = ParagraphStyle("CellH", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    cell_hr = ParagraphStyle("CellHR", parent=cell_h, alignment=2)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=13, bulletIndent=2, spaceAfter=3.5)
    analysis_box = ParagraphStyle("AnalysisBox", parent=body, fontSize=fs - 0.4, leading=(fs - 0.4) * 1.42, leftIndent=8, spaceAfter=4)

    W = PAGE[0] - 1.7 * inch
    story: list[Any] = []

    def P(text: str, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(text, style)

    def bullets(items: list[str], numbered: bool = False) -> list[Paragraph]:
        return [Paragraph(item, bullet, bulletText=(f"{i}." if numbered else "•")) for i, item in enumerate(items, start=1)]

    def table(header: list[str], data: list[list[str]], widths: list[float], right: set[int] | None = None) -> Table:
        right = right or set()
        head = [Paragraph(h, cell_hr if i in right else cell_h) for i, h in enumerate(header)]
        body_rows = [[Paragraph(str(v), cell_r if i in right else cell) for i, v in enumerate(row)] for row in data]
        t = Table([head] + body_rows, colWidths=[w * W for w in widths], repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_hex)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.8, rule), ("BOX", (0, 0), (-1, -1), 0.4, rule)]
        for r in range(1, len(body_rows) + 1):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f6fb")))
        t.setStyle(TableStyle(style))
        return t

    # ---------------------------------------------------------------- what exists, in document order (drives numbering)
    charts_on = [c_ for c_ in charts if c_.get("image_base64")] if opt["include_charts"] else []
    has_hist = len(desc) >= 1 and opt["include_figures_native"]
    has_miss_fig = snapshot["missing_percentage"] > 0 and opt["include_figures_native"]
    has_corr_fig = corr["matrix"] is not None and len(corr["columns"]) >= 3 and opt["include_figures_native"]
    plan: list[tuple[str, bool]] = [
        ("t_quality", bool(quality.get("issues"))), ("t_desc", bool(desc)), ("f_hist", has_hist), ("t_dist", bool(deep["dist"]) and is_deep), ("t_cat", bool(cats)),
        ("f_missing", has_miss_fig), ("t_corr", bool(corr["pairs"])), ("f_corr", has_corr_fig), ("t_null", bool(dc["nulls"]) and is_std), ("t_vif", bool(dc["vif"]) and is_deep),
        ("t_group", bool(groups["rows"])), ("t_post", bool(deep["posthoc"]) and is_deep), ("t_assoc", bool(deep["cats"]) and is_deep),
        ("t_target", bool(tassoc.get("rows"))), ("t_model", bool(model)), ("t_trend", bool(trend) or bool(trends_deep)),
    ]
    ref: dict[str, int] = {}
    counters = {"table": 0, "figure": 0}
    for key, present in plan:
        if present and (sec_on["quality"] if key == "t_quality" else sec_on["results"]):
            counters["table" if key.startswith("t_") else "figure"] += 1
            ref[key] = counters["table" if key.startswith("t_") else "figure"]

    def cite(key: str) -> str:
        if key not in ref:
            return ""
        return f"({'Table' if key.startswith('t_') else 'Figure'} {ref[key]})"

    def label(key: str) -> str:
        return f"{'Table' if key.startswith('t_') else 'Figure'} {ref[key]}" if key in ref else "the results tables"

    def table_block(key: str, title: str, tbl: Table, note: str | None = None) -> KeepTogether:
        parts: list[Any] = [P(f"Table {ref[key]}. {title}", tcap), tbl]
        if note:
            parts.append(P(note, small))
        return KeepTogether(parts + [Spacer(1, 4)])

    def figure_block(key: str, title: str, drawable: Any, caption: str) -> KeepTogether:
        return KeepTogether([drawable, P(f"<b>Figure {ref[key]}. {title}.</b> {caption}", cap)])

    class _Anchor(Flowable):
        """Zero-size flowable that drops a named PDF destination and a viewer-sidebar outline entry
        at this point in the story, so the Contents block and PDF readers alike can jump straight here."""

        def __init__(self, key: str, title: str, level: int = 0) -> None:
            Flowable.__init__(self)
            self.width = self.height = 0
            self.key, self.title, self.level = key, title, level

        def draw(self) -> None:
            # NumberedCanvas defers each page's real commit to save(), so bookmarking here (against
            # reportlab's not-yet-committed "current" page) would bind every entry to page 1; instead
            # queue it and let NumberedCanvas.save() register it while that exact page is being committed.
            self.canv.registerBookmark(self.key, self.title, self.level)

    # section numbering follows what is switched on
    sec_no = [0]
    sec_ids: dict[str, int] = {}

    def H1(key: str, title: str) -> Paragraph:
        sec_no[0] += 1
        sec_ids[key] = sec_no[0]
        story.append(_Anchor(key, f"{sec_no[0]}. {title}"))
        return P(f"{sec_no[0]}. {title}", h1)

    def H2(text: str) -> Paragraph:
        return P(text, h2)

    # ---------------------------------------------------------------- derived facts
    slug = dataset_info.get("slug") or (dataset_info.get("dataset_id") if "/" in str(dataset_info.get("dataset_id", "")) and not str(dataset_info.get("dataset_id", "")).startswith(("local-", "upload-")) else None)
    display_title = opt["title"] or (kaggle_info.get("title") or dataset_name).strip()
    data_title = (kaggle_info.get("title") or dataset_name).strip()
    n_measures = sum(1 for c in df.columns if profiles[c]["role"] == "numeric")
    n_groupers = sum(1 for c in df.columns if profiles[c]["role"] in {"categorical", "binary", "ordinal", "geo_country", "geo_state"})
    top_corr = corr["pairs"][0] if corr["pairs"] else None
    top_group = groups["rows"][0] if groups["rows"] else None
    issues = quality.get("issues") or []
    actionable = [i for i in issues if i["severity"] in {"high", "medium"}]
    fixable = [i for i in actionable if i.get("fix")]
    missing_total = snapshot["missing_percentage"]
    is_sample = bool(sample and sample.get("is_sample"))
    total_tests = corr["tests"] + groups["tests"] + tassoc.get("tests", 0) + len(trends_deep or ([trend] if trend else []))
    lic = license_info.get("name") or kaggle_info.get("license")
    strongest = dc["strongest"]

    def corr_sentence(row: dict[str, Any]) -> str:
        return f"{esc(row['a'])} and {esc(row['b'])} (r = {row['r']:.2f}, {RHO} = {row['rho']:.2f}, adjusted p {RS.fmt_p(row['p_adj'])}, n = {row['n']:,})"

    # ---------------------------------------------------------------- title block
    prepared = f"Prepared by {esc(opt['author'])}" + (f", {esc(opt['organization'])}" if opt["organization"] else "") if opt["author"] else (esc(opt["organization"]) if opt["organization"] else "")
    sub_line = esc(opt["subtitle"]) or esc(f"{rows_n:,} observations x {cols_n} variables")
    band_rows = [[Paragraph("DATA ANALYSIS REPORT", ParagraphStyle("k", parent=cell, fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor(hilite_hex), leading=10, spaceAfter=0))],
                 [Paragraph(esc(display_title), ParagraphStyle("t", parent=cell, fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.white, spaceAfter=0))],
                 [Paragraph(f"{sub_line}  |  {prepared + '  |  ' if prepared else ''}{datetime.now().strftime('%d %B %Y')}", ParagraphStyle("s", parent=cell, fontSize=8.8, textColor=colors.HexColor("#d5d9f5"), leading=12, spaceAfter=0))]]
    band = Table(band_rows, colWidths=[W])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(band_hex)), ("LEFTPADDING", (0, 0), (-1, -1), 16), ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                              ("TOPPADDING", (0, 0), (0, 0), 14), ("BOTTOMPADDING", (0, -1), (-1, -1), 14), ("TOPPADDING", (0, 1), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -2), 2)]))
    story += [band, Spacer(1, 8)]
    source_line = f"Source: Kaggle dataset {slug}" if slug else f"Source: {dataset_info.get('file_type', 'user-supplied')} file \"{dataset_name}\""
    story.append(P("  |  ".join(esc(b) for b in [source_line] + ([f"License: {lic}"] if lic else []) + [f"Data fingerprint: {S['fingerprint']}"]), small))

    score_color = "#16a34a" if quality["score"] >= 85 else "#d97706" if quality["score"] >= 65 else "#dc2626"
    n_sig_links = corr["n_significant"] + sum(1 for g in groups["rows"] if g["significant"])
    tiles = [("Observations", f"{rows_n:,}", "#3987e5"), ("Variables", str(cols_n), "#d95926"), ("Missing cells", pct(missing_total), "#199e70"),
             ("Data quality", f"{quality['score']}/100", score_color), ("Significant links", str(n_sig_links), "#9085e9")]
    tile_table = Table([[Paragraph(f'<font size="15"><b>{v}</b></font><br/><font size="7" color="#5b6472">{k.upper()}</font>', ParagraphStyle("tile", parent=cell, leading=17, alignment=1)) for k, v, _ in tiles]],
                       colWidths=[W / len(tiles)] * len(tiles))
    ts = [("BOX", (0, 0), (-1, -1), 0.4, rule), ("INNERGRID", (0, 0), (-1, -1), 0.4, rule), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f8fc"))]
    for i, (_, _, colr) in enumerate(tiles):
        ts.append(("LINEABOVE", (i, 0), (i, 0), 2.6, colors.HexColor(colr)))
    tile_table.setStyle(TableStyle(ts))
    story += [tile_table, Spacer(1, 6)]

    # ---------------------------------------------------------------- executive summary
    story.append(P("Executive Summary", h1))
    methods_used = ["descriptive statistics", "Pearson and Spearman correlation with Bonferroni adjustment and robustness checks", "Kruskal-Wallis tests with post-hoc contrasts and effect sizes"]
    if trend:
        methods_used.append("trend, change-point and growth analysis")
    if model:
        methods_used.append("a baseline random-forest model evaluated on held-out rows")
    abstract = (f"This report analyses {esc(data_title)} ({rows_n:,} observations, {cols_n} variables) to describe its structure and data quality, quantify the relationships and trends within it, "
                f"and judge how far each finding can be trusted. The data were audited, prepared{' (' + str(len(transform_log)) + ' cleaning step' + ('s' if len(transform_log) != 1 else '') + ')' if transform_log else ''} and examined using "
                f"{', '.join(methods_used[:-1])}{' and ' if len(methods_used) > 1 else ''}{methods_used[-1]}. ")
    if top_corr and top_corr["significant"]:
        abstract += f"The clearest association was between {corr_sentence(top_corr)}"
        abstract += (f"; it {'held up' if strongest and strongest[0]['consistent'] else 'was sensitive to'} robustness checks (outlier removal, rank correlation and within-group tests). " if strongest else ". ")
    elif corr["pairs"]:
        abstract += "No pair of numeric variables showed a statistically reliable association after adjustment for multiple comparisons. "
    if dc["nulls"]:
        abstract += f"{len(dc['nulls'])} variable pair{'s' if len(dc['nulls']) != 1 else ''} showed no detectable association, which is itself informative. "
    if top_group and top_group["significant"]:
        abstract += f"{esc(top_group['measure'])} differed across {esc(top_group['group'])} groups ({ETA2} = {top_group['eta2']:.2f}, a {top_group['effect']} effect). "
    abstract += f"Overall data quality was rated {quality['label'].lower()} ({quality['score']}/100)" + (f", with {len(actionable)} issue{'s' if len(actionable) != 1 else ''} worth fixing before further use." if actionable else ".")
    story.append(P(abstract))

    story.append(P("Key findings", h2))
    kf = [f"<b>Data:</b> {rows_n:,} rows and {cols_n} variables ({n_measures} numeric measures, {n_groupers} categorical); {pct(missing_total)} of cells missing; quality {quality['score']}/100 ({esc(quality['label'])}) {cite('t_quality')}."]
    if top_corr:
        rating = f" Evidence: <b>{strongest[0]['rating']}</b>." if strongest else ""
        kf.append(f"<b>Strongest relationship:</b> {corr_sentence(top_corr)} - a {top_corr['strength']} association {cite('t_corr')}.{rating}")
    if dc["nulls"]:
        n0 = dc["nulls"][0]
        kf.append(f"<b>Lack of association:</b> {esc(n0['a'])} and {esc(n0['b'])} were unrelated (r = {n0['r']:.2f}, 95% CI {n0['ci'][0]:.2f} to {n0['ci'][1]:.2f}); with n = {n0['n']:,} any correlation above {n0['min_detectable']:.2f} would have been detected {cite('t_null')}.")
    if top_group:
        kf.append(f"<b>Group difference:</b> {esc(top_group['measure'])} varies by {esc(top_group['group'])} (Kruskal-Wallis p {RS.fmt_p(top_group['p_adj'])}, {ETA2} = {top_group['eta2']:.2f}); "
                  f"median is highest for {esc(top_group['high'][0])} ({RS.fmt_num(top_group['high'][1])}) and lowest for {esc(top_group['low'][0])} ({RS.fmt_num(top_group['low'][1])}) {cite('t_group')}.")
    if trends_deep:
        t0 = trends_deep[0]
        direction = "rose" if t0["slope"] > 0 else "fell"
        kf.append(f"<b>Trend:</b> mean {esc(t0['measure'])} {direction}" + (f" by {abs(t0['change_pct']):.0f}%" if t0.get("change_pct") is not None else "") + f" across {t0['periods']} periods (Kendall tau = {t0['tau']:.2f}, p {RS.fmt_p(t0['p_tau'])}"
                  + (f"; compound rate {t0['cagr'] * 100:+.1f}% per {t0['unit']}" if t0.get("cagr") is not None else "") + f") {cite('t_trend')}.")
    elif trend:
        kf.append(f"<b>Trend:</b> mean {esc(trend['measure'])} {'rose' if trend['slope'] > 0 else 'fell'} across {trend['periods']} periods ({RHO} = {trend['rho']:.2f}, p {RS.fmt_p(trend['p_rho'])}) {cite('t_trend')}.")
    if model:
        m_key = model["metric"]
        lab = "balanced accuracy" if m_key == "balanced_accuracy" else "R²"
        kf.append(f"<b>Prediction:</b> a baseline model for {esc(model['target'])} reached {lab} {model['model'][m_key]:.2f} on held-out rows versus {model['baseline'][m_key]:.2f} for a naive baseline {cite('t_model')}.")
    story += bullets(kf[:6])

    story.append(P("Recommended actions", h2))
    actions: list[str] = []
    if fixable:
        actions.append(f"Resolve the data-quality issues first: {esc(fixable[0]['title'].rstrip('.'))}" + (f" (and {len(fixable) - 1} more with one-click fixes)" if len(fixable) > 1 else "") + ".")
    if is_sample:
        actions.append("Re-run the analysis on the complete dataset; these results describe a partial sample.")
    if top_corr and top_corr["significant"]:
        actions.append(f"Investigate the {esc(top_corr['a'])}-{esc(top_corr['b'])} link with a multivariate model that controls for other variables before drawing causal conclusions.")
    if model and model["improvement"] > 0.05:
        actions.append(f"Build on the baseline model for {esc(model['target'])}: tune it, and validate it on data from a different period or source.")
    if len(actions) < 3:
        actions.append("Have a domain expert review the findings before they inform a decision.")
    story += bullets(actions[:4], numbered=True)

    # ---------------------------------------------------------------- Contents
    # Page numbers aren't known until the whole story is laid out, so instead of a printed "page N" this links each
    # entry straight to its section (via the same anchors that populate the PDF viewer's sidebar outline) - always
    # correct, and a click gets you there faster than a page number would anyway.
    toc_entries = [
        ("introduction", "1. Introduction and Objectives"), ("methods", "2. Data and Methods"), ("quality", "3. Data Quality Assessment"),
        ("results", "4. Results"), ("discussion", "5. Discussion"), ("conclusions", "6. Conclusions and Recommendations"),
        ("references", "7. References"), ("appendix_a", "8. Appendices"),
    ]
    toc_on = [(k, t) for k, t in toc_entries if sec_on.get("appendices" if k == "appendix_a" else k)]
    # renumber to match H1()'s own gapless numbering (sections can be switched off)
    toc_on = [(k, re.sub(r"^\d+", str(i), t)) for i, (k, t) in enumerate(toc_on, start=1)]
    if len(toc_on) >= 2:
        story.append(P("Contents", h2))
        story += [P(f'<link href="#{k}" color="{accent_hex}"><u>{esc(t)}</u></link>', body) for k, t in toc_on]
        story.append(Spacer(1, 6))

    # ---------------------------------------------------------------- 1 Introduction
    if sec_on["introduction"]:
        story.append(H1("introduction", "Introduction and Objectives"))
        if kaggle_description:
            cleaned = _clean_description(kaggle_description)
            if cleaned:
                story.append(P(f"<b>Background.</b> {esc(cleaned)}"))
        if opt["notes"]:
            story.append(P(f"<b>Analyst's note.</b> {esc(opt['notes'])}"))
        story.append(P(f"<b>Purpose.</b> This report examines {esc(data_title)} to (i) describe what the data contain and how reliable they are, (ii) quantify the main relationships, differences and trends, test whether they are robust, "
                       f"and state clearly where no relationship exists, and (iii) identify where further analysis is most worthwhile. It is exploratory: the questions were formed from the structure of the data, so they are tested with "
                       f"multiple-comparison control and robustness checks rather than presented as confirmatory findings."))
        hyps: list[str] = []
        hid: dict[str, int] = {}
        if top_corr:
            hid["corr"] = len(hyps) + 1
            hyps.append(f"If {esc(top_corr['a'])} and {esc(top_corr['b'])} share an underlying driver, then they should be monotonically associated (H0: {RHO} = 0).")
        if top_group:
            hid["group"] = len(hyps) + 1
            hyps.append(f"If {esc(top_group['group'])} influences {esc(top_group['measure'])}, then the distribution of {esc(top_group['measure'])} should differ between {esc(top_group['group'])} groups (H0: all groups are drawn from the same distribution).")
        if trends_deep or trend:
            t_meas = trends_deep[0]["measure"] if trends_deep else trend["measure"]
            t_col = (deep["trends"] or {}).get("time") if trends_deep else trend["time"]
            hid["trend"] = len(hyps) + 1
            hyps.append(f"If {esc(t_meas)} has changed over time, then a monotonic trend across {esc(t_col)} should be detectable (H0: no trend).")
        if model:
            hid["model"] = len(hyps) + 1
            hyps.append(f"If the other variables carry information about {esc(model['target'])}, then a model using them should outperform always predicting the {'most common class' if model['kind'] == 'classification' else 'mean'} on unseen rows.")
        if hyps and opt["include_hypotheses"]:
            story.append(P("<b>Hypotheses.</b>", ParagraphStyle("hh", parent=body, spaceAfter=2, keepWithNext=1)))
            story += [Paragraph(h, bullet, bulletText=f"H{i}") for i, h in enumerate(hyps[:4], start=1)]
        story.append(P("<b>Why it matters.</b> Knowing which variables move together, where groups differ, how things change over time and how trustworthy the data are determines which fields should be modelled, monitored or collected more carefully."))
    else:
        hid = {}
    if not (opt["include_hypotheses"] and sec_on["introduction"]):
        hid = {}  # no hypotheses listed, so the discussion must not cite them

    # ---------------------------------------------------------------- 2 Data and Methods
    if sec_on["methods"]:
        story.append(H1("methods", "Data and Methods"))
        story.append(H2("Data source and preparation"))
        src = (f"The data were obtained from the Kaggle dataset {esc(slug)} (https://www.kaggle.com/datasets/{esc(slug)})" if slug else f"The data were supplied as the {esc(dataset_info.get('file_type', 'tabular'))} file \"{esc(dataset_name)}\"")
        src += (f", released under the {esc(lic)} license ({LICENSE_LEVEL_TEXT.get(license_info.get('level', ''), 'terms should be checked before reuse')})." if lic else ". No license information was recorded, so terms of reuse should be checked with the owner.")
        if is_sample:
            src += f" Only a partial sample of the file was analysed ({rows_n:,} rows" + (f" of an estimated {sample['estimated_total_rows']:,}" if sample.get("estimated_total_rows") else "") + ")."
        story.append(P(src))
        if transform_log:
            story.append(P(f"The following {len(transform_log)} preparation step{'s were' if len(transform_log) != 1 else ' was'} applied, in order, before analysis:", ParagraphStyle("pl", parent=body, spaceAfter=2, keepWithNext=1)))
            story += bullets([esc(t["description"]) for t in transform_log[:12]], numbered=True)
            if len(transform_log) > 12:
                story.append(P(f"...and {len(transform_log) - 12} further steps (see the reproducibility record).", small))
        else:
            story.append(P("No rows or columns were removed or altered: the analysis used the data exactly as loaded."))
        rc = schema["role_counts"]
        story.append(P(f"Each column was classified by what it represents rather than by storage type: {n_measures} numeric measure{'s' if n_measures != 1 else ''}, {n_groupers} categorical grouping variable{'s' if n_groupers != 1 else ''}, "
                       f"{rc.get('identifier', 0)} identifier{'s' if rc.get('identifier', 0) != 1 else ''} and {rc.get('text', 0)} free-text column{'s' if rc.get('text', 0) != 1 else ''}. Identifier and free-text columns were excluded from statistics."))
        story.append(H2("Statistical analysis"))
        story.append(P(
            f"Numeric variables were summarised as mean &plusmn; standard deviation (SD) and as median with interquartile range (IQR); normality was assessed with the Shapiro-Wilk test (Shapiro &amp; Wilk, 1965). "
            f"Pairwise associations were quantified with Pearson's r (Pearson, 1895), given with Fisher-z 95% confidence intervals, and Spearman's {RHO} (Spearman, 1904). Each headline correlation was stress-tested: "
            f"by removing outliers (1.5 x IQR rule; Tukey, 1977), by comparing a straight-line with a quadratic fit, and by recomputing it within subgroups to detect Simpson's paradox. Multicollinearity was assessed with variance inflation factors. "
            f"Group differences used the Kruskal-Wallis test (Kruskal &amp; Wallis, 1952) with the effect size {ETA2} (0.01, 0.06 and 0.14 mark small, medium and large effects; Cohen, 1988) and Holm-adjusted pairwise Mann-Whitney contrasts with rank-biserial effect sizes. "
            f"Trends used ordinary least squares, Kendall's tau, compound growth, and a search for a single change-point. Associations between categorical variables used the {CHI2} test with Cram&eacute;r's V (Cram&eacute;r, 1946). "
            f"The significance level was {ALPHA} = 0.05; within each family of tests p-values were multiplied by the number of tests (Bonferroni; Dunn, 1961), and {total_tests} tests were run in total. "
            f"For null results, the smallest correlation detectable with 80% power was reported so that 'no association' is not confused with 'too little data'. Effect sizes accompany every p-value."))
        if model:
            story.append(P(f"For the most plausible target ({esc(model['target'])}, a {model['kind']} task) a random-forest model (Breiman, 2001) was trained on a random 75% of rows ({model['train']:,}) using {model['features']} features, "
                           f"and evaluated on the remaining {model['test']:,} unseen rows against a naive baseline. Feature reliance was measured by permutation importance. Random seed 42 was used throughout."))
        story.append(H2("Evidence ratings"))
        story.append(P("Each headline finding carries a rating. <b>High</b>: statistically significant after adjustment, a large effect, at least 30 observations, and unchanged by the robustness checks. "
                       "<b>Moderate</b>: significant with a medium effect, or a large effect that is sensitive to a robustness check. <b>Low</b>: everything else, including all null results. Ratings summarise reliability, not importance."))
        story.append(H2("Software and reproducibility"))
        story.append(P(f"Analyses were run in {esc(S['software'])} (McKinney, 2010; Virtanen et al., 2020; Pedregosa et al., 2011). The analysed table has fingerprint {S['fingerprint']}. The exported DataScopia notebook replays the preparation steps and reproduces the statistics and charts in this report."))

    # ---------------------------------------------------------------- 3 Data quality
    if sec_on["quality"]:
        story.append(H1("quality", "Data Quality Assessment"))
        c = quality["counts"]
        story.append(P(f"The audit scored the data <b>{quality['score']}/100 ({esc(quality['label'])})</b>: {c['high']} high, {c['medium']} medium and {c['low']} low severity findings across {rows_n:,} rows and {cols_n} columns. "
                       f"Overall {pct(missing_total)} of cells were missing and {snapshot['duplicate_rows']:,} rows were exact duplicates. " + (f"The findings are listed in {label('t_quality')}." if issues else "No quality issues were detected.")))
        if issues:
            sev_col = {"high": "#dc2626", "medium": "#d97706", "low": "#0284c7"}
            qrows = []
            for i in issues[:10]:
                fix = (i.get("fix") or {}).get("type", "")
                qrows.append([f'<font color="{sev_col.get(i["severity"], "#000")}"><b>{esc(i["severity"].title())}</b></font>', esc(", ".join(i.get("columns") or [])[:60] or "whole table"), esc(i["title"]), esc(fix.replace("_", " ") if fix else "review")])
            if "t_quality" not in ref:
                counters["table"] += 1
                ref["t_quality"] = counters["table"]
            story.append(table_block("t_quality", "Data-quality findings, most severe first", table(["Severity", "Where", "Finding", "Suggested action"], qrows, [0.11, 0.22, 0.47, 0.20]),
                                     f"Showing {min(10, len(issues))} of {len(issues)} findings." if len(issues) > 10 else None))

    # ---------------------------------------------------------------- 4 Results
    if sec_on["results"]:
        story.append(H1("results", "Results"))
        story.append(P("Results are stated without interpretation; the Discussion explains what they mean. Every test reports an effect size, and p-values are adjusted within each family of tests.", small))
        sub_no = [0]

        def sub(title: str) -> Paragraph:
            sub_no[0] += 1
            return P(f"{sec_no[0]}.{sub_no[0]} {title}", h2)

        if desc:
            story.append(sub("Descriptive statistics"))
            drows = [[esc(d["column"]), f"{d['n']:,}", f"{RS.fmt_num(d['mean'])} &plusmn; {RS.fmt_num(d['sd'])}", RS.fmt_num(d["median"]), f"{RS.fmt_num(d['q1'])}–{RS.fmt_num(d['q3'])}",
                      f"{RS.fmt_num(d['min'])}–{RS.fmt_num(d['max'])}", f"{d['skew']:.2f}", "n/a" if d["shapiro_p"] is None else RS.fmt_p(d["shapiro_p"])] for d in desc]
            story.append(P(f"{label('t_desc')} summarises the {len(desc)} numeric measures" + (f"; {label('f_hist')} shows their distributions." if has_hist else ".")))
            story.append(table_block("t_desc", "Descriptive statistics of numeric variables (mean &plusmn; SD; median with IQR)",
                                     table(["Variable", "n", "Mean &plusmn; SD", "Median", "IQR", "Range", "Skew", "Shapiro p"], drows, [0.17, 0.07, 0.19, 0.10, 0.15, 0.15, 0.07, 0.10], right={1, 2, 3, 4, 5, 6, 7}),
                                     "n = non-missing values. Shapiro p &lt; 0.05 indicates departure from a normal distribution."))
            if has_hist:
                series = {d["column"]: RS._numeric(df, d["column"]) for d in desc[:6]}
                story.append(figure_block("f_hist", "Distributions of the main numeric variables", histogram_grid(series, W), "Solid line = mean, dashed line = median. Where the two lines are far apart the distribution is skewed and the median is the better summary of a typical value."))
        if deep["dist"] and "t_dist" in ref:
            drows2 = [[esc(d["column"]), d["shape"], d["tails"], f"{d['skew']:.2f}", f"{d['kurtosis']:.2f}", f"{d['outlier_pct']:.1f}%", f"{d['zero_pct']:.0f}%", esc(d["suggestion"])] for d in deep["dist"]]
            story.append(table_block("t_dist", "Distribution diagnostics", table(["Variable", "Shape", "Tails", "Skew", "Excess kurt.", "Outliers", "Zeros", "Suggested handling"], drows2, [0.15, 0.17, 0.16, 0.07, 0.09, 0.09, 0.07, 0.20], right={3, 4, 5, 6}),
                                     "Outliers = share of values outside 1.5 x IQR. Heavy tails: excess kurtosis above 3."))
            skewed = [d for d in deep["dist"] if abs(d["skew"]) > 1]
            if skewed:
                story.append(P(f"{len(skewed)} of {len(deep['dist'])} variables were strongly skewed (|skew| &gt; 1): " + ", ".join(f"{esc(d['column'])} ({d['skew']:+.1f})" for d in skewed[:6]) + "."))
        if cats:
            crow = [[esc(ci["column"]), f"{ci['levels']:,}", f"{ci['n']:,}", "; ".join(f"{esc(k)} ({pv:.0f}%)" for k, _, pv in ci["top"])] for ci in cats]
            story.append(table_block("t_cat", "Most common values of categorical variables", table(["Variable", "Levels", "n", "Most frequent values (share of rows)"], crow, [0.22, 0.09, 0.10, 0.59], right={1, 2})))
        if missing_total > 0:
            miss = sorted(((c_, float(df[c_].isna().mean() * 100)) for c_ in df.columns), key=lambda kv: kv[1], reverse=True)
            miss = [m for m in miss if m[1] > 0][:10]
            story.append(sub("Missing data"))
            story.append(P(f"{len([1 for c_ in df.columns if df[c_].isna().any()])} of {cols_n} columns contained missing values; the most affected was {esc(miss[0][0])} at {miss[0][1]:.1f}%" + (f" ({label('f_missing')})." if has_miss_fig else ".")))
            md = deep["missing"]
            if md:
                bits = []
                if md["co_missing"]:
                    bits.append("columns that tended to be missing together: " + "; ".join(f"{esc(a)} and {esc(b)} (phi = {v:.2f})" for a, b, v in md["co_missing"]))
                for d_ in md["dependent"][:3]:
                    bits.append(f"rows missing {esc(d_['missing'])} had a {'lower' if d_['median_missing'] < d_['median_present'] else 'higher'} median {esc(d_['other'])} ({RS.fmt_num(d_['median_missing'])} vs {RS.fmt_num(d_['median_present'])}, rank-biserial {d_['effect']:+.2f})")
                story.append(P("Missingness pattern: " + ("; ".join(bits) + "." if bits else "no column pair went missing together and missingness was not related to the values of the numeric columns tested.")))
            if has_miss_fig:
                story.append(figure_block("f_missing", "Share of missing values by column", missing_figure(miss, W), "Bars show the percentage of rows with no value. Columns with substantial gaps limit every analysis that uses them."))
        if corr["pairs"]:
            story.append(sub("Relationships between numeric variables"))
            story.append(P(f"Of {corr['tests']} variable pairs tested, {corr['n_significant']} remained significant after Bonferroni adjustment; {dc['share_weak']:.0%} of pairs had |r| below 0.3. The strongest pairs are in {label('t_corr')}."))
            deep_by_pair = {(d["a"], d["b"]): d for d in strongest}
            crows = []
            for r in corr["pairs"]:
                d = deep_by_pair.get((r["a"], r["b"]))
                ci = f"{d['ci'][0]:.2f} to {d['ci'][1]:.2f}" if d else "-"
                rob = f"{d['r_robust']:.2f}" if d and d["r_robust"] is not None else "-"
                crows.append([f"{esc(r['a'])} &ndash; {esc(r['b'])}", f"{r['n']:,}", f"{r['r']:.2f}", ci, f"{r['r'] ** 2:.0%}", f"{r['rho']:.2f}", rob, RS.fmt_p(r["p_adj"]), r["strength"].capitalize()])
            story.append(table_block("t_corr", "Strongest pairwise correlations with confidence intervals and robustness",
                                     table(["Pair", "n", "Pearson r", "95% CI", "R²", f"Spearman {RHO}", "r w/o outliers", "Adj. p", "Strength"], crows, [0.23, 0.07, 0.10, 0.15, 0.06, 0.11, 0.11, 0.08, 0.09], right={1, 2, 4, 5, 6, 7}),
                                     "Adj. p = Bonferroni-adjusted. R² = share of variance shared. 'r w/o outliers' = Pearson r after removing rows outside 1.5 x IQR (computed for the top pairs). Strength follows |r|: &lt;0.1 negligible, &lt;0.3 weak, &lt;0.5 moderate, &lt;0.7 strong, otherwise very strong."))
            if has_corr_fig:
                story.append(figure_block("f_corr", "Correlation matrix (Pearson r)", correlation_figure(corr["matrix"], W),
                                          "Blue cells are negative and red cells positive correlations; darker colours mean stronger relationships. Variables that form a red block move together and may carry overlapping information."))
            if is_std and strongest:
                story.append(P("Detailed results for the strongest relationships", h3))
                for d in strongest[: (3 if is_deep else 1)]:
                    story.append(P(f"<b>{esc(d['a'])} and {esc(d['b'])}.</b> " + " ".join(esc(x_) for x_ in D.pair_facts(d)), analysis_box))
                if dc["clusters"]:
                    story.append(P("<b>Clusters of related variables</b> (|r| &ge; 0.5): " + "; ".join(esc(", ".join(c_)) for c_ in dc["clusters"][:4]) + ".", analysis_box))
            if "t_null" in ref:
                nrows = [[f"{esc(n_['a'])} &ndash; {esc(n_['b'])}", f"{n_['n']:,}", f"{n_['r']:.2f}", f"{n_['ci'][0]:.2f} to {n_['ci'][1]:.2f}", f"{n_['min_detectable']:.2f}"] for n_ in dc["nulls"]]
                story.append(table_block("t_null", "Variable pairs with no detectable association", table(["Pair", "n", "Pearson r", "95% CI", "Min. detectable |r|"], nrows, [0.36, 0.10, 0.14, 0.22, 0.18], right={1, 2, 3, 4}),
                                         "Confidence interval spans zero and |r| &lt; 0.15. 'Min. detectable |r|' = the smallest correlation this sample would detect with 80% power, so effects larger than this are unlikely to have been missed."))
            if "t_vif" in ref:
                vrows = [[esc(v["column"]), f"{v['vif']:.1f}", v["level"].capitalize()] for v in dc["vif"]]
                story.append(table_block("t_vif", "Variance inflation factors (multicollinearity)", table(["Variable", "VIF", "Level"], vrows, [0.5, 0.2, 0.3], right={1}), "VIF &ge; 5 is moderate and &ge; 10 severe: the variable is largely predictable from the others."))
        if groups["rows"]:
            story.append(sub("Differences between groups"))
            story.append(P(f"{label('t_group')} lists the {len(groups['rows'])} largest of {groups['tests']} group comparisons, ranked by effect size."))
            grows = [[esc(g["measure"]), esc(g["group"]), str(g["k"]), f"{g['n']:,}", f"{g['H']:.1f}", RS.fmt_p(g["p_adj"]), f"{g['eta2']:.2f}", g["effect"].capitalize(), f"{esc(g['high'][0])} ({RS.fmt_num(g['high'][1])})", f"{esc(g['low'][0])} ({RS.fmt_num(g['low'][1])})"] for g in groups["rows"]]
            story.append(table_block("t_group", "Kruskal-Wallis comparisons across groups", table(["Measure", "Grouped by", "Groups", "n", "H", "Adj. p", ETA2, "Effect", "Highest median", "Lowest median"], grows, [0.13, 0.12, 0.09, 0.07, 0.06, 0.08, 0.06, 0.09, 0.15, 0.15], right={2, 3, 4, 5, 6}),
                                     f"{ETA2} = share of variance explained by group membership. Up to the 12 largest groups were compared."))
            if is_std and deep["posthoc"]:
                for ph, g_row in deep["posthoc"][: (2 if is_deep else 1)]:
                    story.append(P(f"<b>{esc(ph['measure'])} by {esc(ph['group'])}.</b> " + " ".join(esc(x_) for x_ in D.posthoc_facts(ph)), analysis_box))
            if "t_post" in ref:
                ph0 = deep["posthoc"][0][0]
                prow = [[f"{esc(r['a'])} vs {esc(r['b'])}", RS.fmt_num(r["median_a"]), RS.fmt_num(r["median_b"]), f"{r['effect']:+.2f}", r["magnitude"].capitalize(), RS.fmt_p(r["p_adj"])] for r in ph0["pairs"][:8]]
                story.append(table_block("t_post", f"Post-hoc pairwise contrasts: {esc(ph0['measure'])} by {esc(ph0['group'])}", table(["Contrast", "Median A", "Median B", "Rank-biserial", "Magnitude", "Holm adj. p"], prow, [0.32, 0.13, 0.13, 0.15, 0.13, 0.14], right={1, 2, 3, 5}),
                                         f"Showing the 8 largest of {ph0['n_pairs']} contrasts. Positive rank-biserial means group A tends to have larger values."))
        if "t_assoc" in ref:
            story.append(sub("Associations between categorical variables"))
            arows = [[f"{esc(r['a'])} &times; {esc(r['b'])}", f"{r['n']:,}", f"{r['chi2']:.1f}", RS.fmt_p(r["p_adj"]), f"{r['v']:.2f}", r["effect"].capitalize(),
                      f"{esc(r['cell'][0])} / {esc(r['cell'][1])}: {r['cell'][2]:.0f} observed vs {r['cell'][3]:.1f} expected"] for r in deep["cats"]]
            story.append(table_block("t_assoc", "Chi-square tests between categorical variables", table(["Pair", "n", CHI2, "Adj. p", "V", "Effect", "Cell driving the association"], arows, [0.22, 0.07, 0.08, 0.09, 0.07, 0.10, 0.37], right={1, 2, 3, 4}),
                                     "V = Cram&eacute;r's V. The driving cell has the largest standardised residual."))
        if tassoc.get("rows"):
            story.append(sub(f"Variables associated with {esc(tassoc['target'])}"))
            story.append(P(f"The column {esc(tassoc['target'])} was identified as the most plausible outcome ({esc(target['reason']) if target else ''}). {label('t_target')} ranks the variables most associated with it."))
            measure_name = "Cram&eacute;r's V / " + ETA2 if tassoc["kind"] == "classification" else f"Spearman {RHO}"
            trows = [[esc(r["feature"]), r["kind"], f"{r['n']:,}", f"{r['stat']:.2f}", RS.fmt_p(r["p_adj"]), r["effect"].capitalize()] for r in tassoc["rows"]]
            story.append(table_block("t_target", f"Association of each variable with {esc(tassoc['target'])}", table(["Variable", "Type", "n", measure_name, "Adj. p", "Effect"], trows, [0.28, 0.13, 0.10, 0.22, 0.12, 0.15], right={2, 3, 4})))
        if model:
            m_key = model["metric"]
            if model["kind"] == "classification":
                mrows = [[nm, f"{model[k]['accuracy']:.2f}", f"{model[k]['balanced_accuracy']:.2f}", f"{model[k]['macro_f1']:.2f}"] for nm, k in (("Naive baseline (majority class)", "baseline"), ("Random forest", "model"))]
                head = ["Model", "Accuracy", "Balanced accuracy", "Macro F1"]
            else:
                mrows = [[nm, f"{model[k]['r2']:.2f}", RS.fmt_num(model[k]["mae"])] for nm, k in (("Naive baseline (mean)", "baseline"), ("Random forest", "model"))]
                head = ["Model", "R²", "Mean abs. error"]
            story.append(P(f"{label('t_model')} compares a random forest with a naive baseline on {model['test']:,} held-out rows."))
            imp = "; ".join(f"{esc(k)} ({v:+.3f})" for k, v in model["importance"][:5])
            story.append(table_block("t_model", f"Baseline model performance for {esc(model['target'])} (held-out rows)", table(head, mrows, [0.4] + [0.6 / (len(head) - 1)] * (len(head) - 1), right=set(range(1, len(head)))),
                                     (f"Permutation importance (drop in {'balanced accuracy' if m_key == 'balanced_accuracy' else 'R²'} when shuffled): {imp}." if imp else None)))
        if trends_deep or trend:
            story.append(sub("Change over time"))
            time_name = (deep["trends"] or {}).get("time") or trend["time"]
            if trends_deep:
                story.append(P(f"Mean values were computed within each period of {esc(time_name)}; {label('t_trend')} summarises {len(trends_deep)} series."))
                trows = []
                for t_ in trends_deep:
                    cp = t_.get("changepoint")
                    trows.append([esc(t_["measure"]), f"{RS.fmt_num(t_['first'])} to {RS.fmt_num(t_['last'])}", f"{t_['change_pct']:+.0f}%" if t_.get("change_pct") is not None else "-",
                                  f"{t_['cagr'] * 100:+.1f}%" if t_.get("cagr") is not None else "-", f"{RS.fmt_num(t_['slope'])}/{t_['unit']}", f"{t_['r2']:.2f}", f"{t_['tau']:.2f}", RS.fmt_p(t_["p_tau"]),
                                  f"{t_['volatility'] * 100:.1f}%" if t_.get("volatility") is not None else "-", f"{cp['at']:.0f}" if cp else "none"])
                story.append(table_block("t_trend", "Trend statistics for the period means", table(["Measure", "First to last", "Change", "CAGR", "Slope", "R²", "Tau", "p", "Volatility", "Break"], trows, [0.14, 0.17, 0.08, 0.08, 0.13, 0.06, 0.06, 0.08, 0.09, 0.08], right={2, 3, 4, 5, 6, 7, 8}),
                                         "CAGR = compound growth rate per unit of time. Volatility = standard deviation of period-to-period % changes. Break = single change-point that fits much better than one line."))
                if is_std:
                    for t_ in trends_deep[: (3 if is_deep else 1)]:
                        story.append(P(f"<b>{esc(t_['measure'])}.</b> " + " ".join(esc(x_) for x_ in D.trend_facts(t_, time_name)), analysis_box))
                        g = t_.get("groups")
                        if g and is_deep:
                            story.append(P(f"By {esc(g['column'])}: " + "; ".join(f"{esc(r['group'])} {r['pct_per_period']:+.1f}% of mean per {t_['unit']}" for r in g["rows"][:6] if r["pct_per_period"] is not None) + ".", analysis_box))
            else:
                story.append(P(f"Mean {esc(trend['measure'])} was averaged within each of {trend['periods']} periods of {esc(time_name)} ({label('t_trend')})."))
                trows = [[esc(trend["measure"]), f"{trend['start']:.0f}–{trend['end']:.0f}", f"{RS.fmt_num(trend['first'])} to {RS.fmt_num(trend['last'])}", RS.fmt_num(trend["slope"]), f"{trend['r2']:.2f}", RS.fmt_p(trend["p"]), f"{trend['rho']:.2f}"]]
                story.append(table_block("t_trend", "Linear trend of the period mean", table(["Measure", "Span", "First to last", "Slope / year", "R²", "p", f"Spearman {RHO}"], trows, [0.2, 0.13, 0.22, 0.14, 0.08, 0.10, 0.13], right={3, 4, 5, 6})))

        if charts_on:
            story.append(sub("Visualizations from the workspace"))
            story.append(P("Each figure is followed by what the underlying data show and what that means.", small))
            for chart in charts_on:
                counters["figure"] += 1
                fig_no = counters["figure"]
                title = chart.get("title") or (f"{chart.get('x', '')}" + (f" vs {chart.get('y')}" if chart.get("y") else ""))
                try:
                    img = RLImage(io.BytesIO(base64.b64decode(chart["image_base64"])), width=W * 0.92, height=W * 0.92 * (440 / 820))
                except Exception:  # pragma: no cover
                    continue
                deep_read = D.analyze_chart(df, schema, chart, corr) if is_std else None
                block: list[Any] = [img, P(f"<b>Figure {fig_no}. {esc(title)}.</b>", cap)]
                if deep_read:
                    block.append(P("<b>What the data show.</b> " + " ".join(esc(x_) for x_ in deep_read["shows"]), analysis_box))
                    block.append(P("<b>What it means.</b> " + " ".join(esc(x_) for x_ in deep_read["means"]), analysis_box))
                else:
                    block.append(P(esc(first_sentences(generate_chart_narrative(df, chart), 3 if is_std else 2)), analysis_box))
                story.append(KeepTogether(block[:3]))
                story += block[3:]
                story.append(Spacer(1, 6))
        if findings:
            story.append(sub("Analyst findings saved during exploration"))
            for f in findings[:8]:
                if f.get("type") == "question":
                    story.append(P(f"<b>Q:</b> {esc(f.get('question', ''))} <b>A:</b> {esc(f.get('answer', ''))}", bullet))
                else:
                    story.append(P(f"<b>{esc(f.get('title', 'Finding'))}.</b> {esc(f.get('summary', ''))}", bullet))

    # ---------------------------------------------------------------- 5 Discussion
    disc_headline: list[str] = []
    if sec_on["discussion"]:
        story.append(H1("discussion", "Discussion"))
        subd = [0]

        def dsub(title: str) -> Paragraph:
            subd[0] += 1
            return P(f"{sec_no[0]}.{subd[0]} {title}", h2)

        if strongest or top_corr:
            story.append(dsub("Relationships between variables"))
            if strongest:
                for i, d in enumerate(strongest[: (3 if is_deep else 1)]):
                    tag = f"H{hid['corr']}: " if i == 0 and "corr" in hid else ""
                    story.append(P(f"<b>{tag}{esc(d['a'])} and {esc(d['b'])} (evidence: {d['rating']}).</b> " + " ".join(esc(x_) for x_ in D.pair_interpretation(d, caveat=(i == 0)))))
            elif top_corr:
                story.append(P(f"The strongest numeric association was {corr_sentence(top_corr)}. Because this is observational data it does not show that either variable causes the other."))
            if dc["clusters"] and is_std:
                worst = dc["vif"][0] if dc["vif"] else None
                story.append(P(f"<b>Overlapping information.</b> {len(dc['clusters'])} group{'s' if len(dc['clusters']) != 1 else ''} of variables move together ({esc('; '.join(', '.join(c_) for c_ in dc['clusters'][:3]))}). Variables in the same cluster carry largely the same signal, so treating them as independent evidence overstates how much the data say"
                               + (f", and a model using them jointly would suffer multicollinearity (highest VIF {worst['vif']:.1f} for {esc(worst['column'])})." if worst and worst["vif"] >= 5 else ".")))
        if is_std and (dc["nulls"] or dc["isolated"]):
            story.append(dsub("What the data do not show"))
            para = []
            if dc["nulls"]:
                n0 = dc["nulls"][0]
                para.append(f"Several pairs of variables showed no detectable association - for example {esc(n0['a'])} and {esc(n0['b'])} (r = {n0['r']:.2f}, 95% CI {n0['ci'][0]:.2f} to {n0['ci'][1]:.2f}). "
                            f"With {n0['n']:,} observations any linear correlation stronger than about {n0['min_detectable']:.2f} would very likely have been detected, so the absence is a genuine finding for effects of practical size, not just a lack of data; "
                            f"it does not rule out a very weak relationship, a curved or threshold relationship, or an effect that appears only within a subgroup.")
            for iso in dc["isolated"][:3]:
                para.append(f"{esc(iso['column'])} was essentially unrelated to every other numeric variable (largest |r| = {iso['max_abs']:.2f}, with {esc(iso['partner'])}); it behaves as an independent dimension of the data and cannot be predicted from, or used to predict, the others in a linear way.")
            if dc["share_weak"] >= 0.5:
                para.append(f"Overall {dc['share_weak']:.0%} of variable pairs were weakly or not at all correlated, so most of the variation in this dataset is not explained by simple pairwise relationships.")
            story += [P(x_) for x_ in para]
        if trends_deep and is_std:
            story.append(dsub("Change over time"))
            time_name = (deep["trends"] or {}).get("time", "time")
            for i, t_ in enumerate(trends_deep[: (3 if is_deep else 1)]):
                tag = f"H{hid['trend']}: " if i == 0 and "trend" in hid else ""
                story.append(P(f"<b>{tag}{esc(t_['measure'])}.</b> " + " ".join(esc(x_) for x_ in D.trend_interpretation(t_, time_name))))
        elif trend:
            story.append(dsub("Change over time"))
            story.append(P(f"H{hid.get('trend', '')} is {'supported' if trend['p_rho'] < RS.ALPHA else 'not supported'}: {esc(trend['measure'])} shows a {'consistent upward' if trend['slope'] > 0 else 'consistent downward'} movement across {esc(trend['time'])} (R² = {trend['r2']:.2f})."))
        if deep["posthoc"] and is_std:
            story.append(dsub("Differences between groups"))
            for i, (ph, g_row) in enumerate(deep["posthoc"][: (2 if is_deep else 1)]):
                tag = f"H{hid['group']}: " if i == 0 and "group" in hid else ""
                story.append(P(f"<b>{tag}{esc(ph['measure'])} by {esc(ph['group'])}.</b> " + " ".join(esc(x_) for x_ in D.posthoc_interpretation(ph, g_row, caveat=(i == 0)))))
        elif top_group:
            story.append(dsub("Differences between groups"))
            share = top_group["eta2"] * 100
            story.append(P(f"For H{hid.get('group', '')}, {esc(top_group['group'])} {'accounts for' if top_group['significant'] else 'explains only'} about {share:.0f}% of the variation in {esc(top_group['measure'])}, a {top_group['effect']} effect."))
        skewed_d = [d for d in deep["dist"] if abs(d["skew"]) > 1]
        if is_deep and (skewed_d or deep["missing"]):
            story.append(dsub("Data characteristics that affect interpretation"))
            if skewed_d:
                story.append(P(f"{len(skewed_d)} variable{'s are' if len(skewed_d) != 1 else ' is'} strongly skewed ({esc(', '.join(d['column'] for d in skewed_d[:4]))}). For these the mean overstates a typical value and Pearson correlations are pulled by the tail; medians and rank correlations (which this report reports alongside) are more trustworthy, and a "
                               f"{esc(skewed_d[0]['suggestion'])} would be the natural step before any modelling."))
            md = deep["missing"]
            if md and (md["dependent"] or md["co_missing"]):
                story.append(P("Missing values were not random: " + ("some columns went missing together, suggesting a shared cause such as one source or form that did not collect them. " if md["co_missing"] else "") + ("Missingness related to other variables' values, which can bias averages and correlations computed on the complete rows only." if md["dependent"] else "")))
        if model:
            story.append(dsub("Prediction"))
            m_key = model["metric"]
            good = model["improvement"] > 0.05
            imp_vals = [abs(v) for _, v in model["importance"]] if model["importance"] else []
            conc = (imp_vals[0] / sum(imp_vals)) if imp_vals and sum(imp_vals) else None
            story.append(P(f"<b>H{hid.get('model', '')} is {'supported' if good else 'only weakly supported'}:</b> the random forest {'improved on' if good else 'barely improved on'} the naive baseline by {model['improvement']:.2f} in {'balanced accuracy' if m_key == 'balanced_accuracy' else 'R²'} "
                           f"({model['baseline'][m_key]:.2f} to {model['model'][m_key]:.2f}) on {model['test']:,} unseen rows. "
                           + (f"Predictive signal is concentrated: {esc(model['importance'][0][0])} alone accounts for about {conc:.0%} of the measured importance, so the model is essentially a statement about that one variable. " if conc and conc >= 0.5 else "")
                           + "Permutation importance shows which variables the model relied on, not which cause the outcome, and performance on a different period or source may be lower."))

        story.append(dsub("Limitations and sources of error"))
        lim: list[str] = []
        if is_sample:
            lim.append("Results describe a partial sample of the source file and may shift on the full data.")
        if missing_total > 0:
            lim.append(f"{pct(missing_total)} of cells were missing. If values are not missing at random, correlations and group comparisons may be biased; rows with missing values were excluded pairwise from each test.")
        lim.append(f"{total_tests} exploratory tests were run on hypotheses chosen after inspecting the data. Bonferroni adjustment guards against false positives but is conservative, so real effects in small samples may be missed; conversely, the rating system treats a significant result as reliable only if it survives the robustness checks.")
        if snapshot["duplicate_rows"]:
            lim.append(f"{snapshot['duplicate_rows']:,} duplicate rows were present; if they are not genuine repeats they overweight some observations.")
        if fresh.get("outlier_count"):
            lim.append(f"{fresh['outlier_count']:,} values fell outside 1.5 x IQR of their variable. They were retained, so means and Pearson correlations may be influenced by them (the robust correlations in the results show by how much).")
        lim.append("Only linear, rank-based and tree-based patterns were examined; interactions, time dependence within groups and unobserved confounders were not modelled, and none of the associations can be read as causal.")
        story += bullets(lim[:6])

    # ---------------------------------------------------------------- 6 Conclusions and recommendations
    if sec_on["conclusions"]:
        story.append(H1("conclusions", "Conclusions and Recommendations"))
        evidence: list[list[str]] = []
        rating_color = {"High": "#15803d", "Moderate": "#b45309", "Low": "#b91c1c", "Null result": "#475569"}

        def ev(name: str, stat: str, robust: str, rating: str) -> None:
            evidence.append([name, stat, robust, f'<font color="{rating_color.get(rating, "#000")}"><b>{rating}</b></font>'])

        for d in strongest[:3]:
            rob = []
            if d["r_robust"] is not None:
                rob.append(f"r {d['r_robust']:.2f} without outliers")
            if d.get("simpson"):
                rob.append(f"same sign in {d['simpson']['same_sign']}/{d['simpson']['total']} {d['simpson']['group']} groups")
            ev(f"{esc(d['a'])} ~ {esc(d['b'])}", f"r = {d['r']:.2f} ({d['ci'][0]:.2f} to {d['ci'][1]:.2f}); adj p {RS.fmt_p(d['p_adj'])}", esc("; ".join(rob) or "-"), d["rating"])
        for ph, g_row in deep["posthoc"][:2]:
            consistent = ph["n_sig"] >= 0.5 * ph["n_pairs"]
            ev(f"{esc(ph['measure'])} by {esc(ph['group'])}", f"{ETA2} = {g_row['eta2']:.2f}; adj p {RS.fmt_p(g_row['p_adj'])}", f"{ph['n_sig']}/{ph['n_pairs']} pairs differ (Holm)", D.rate_evidence(g_row["p_adj"], g_row["eta2"], g_row["n"], consistent, "eta"))
        for t_ in trends_deep[:2]:
            e = t_.get("entities")
            broad = (e is None) or max(e["share_up"], e["share_down"]) >= 0.7
            ev(f"{esc(t_['measure'])} over {esc((deep['trends'] or {}).get('time', 'time'))}", f"tau = {t_['tau']:.2f}; R² = {t_['r2']:.2f}; p {RS.fmt_p(t_['p_tau'])}", (f"{e['share_up']:.0%} of {esc(e['column'])} rising" if e else "period means only"), D.rate_evidence(t_["p_tau"], math.sqrt(t_["r2"]), t_["periods"] * 3, broad))
        if model:
            ev(f"Predicting {esc(model['target'])}", f"{'balanced accuracy' if model['metric'] == 'balanced_accuracy' else 'R²'} {model['model'][model['metric']]:.2f} vs baseline {model['baseline'][model['metric']]:.2f}", f"{model['test']:,} held-out rows",
               "High" if model["improvement"] > 0.15 and model["test"] >= 100 else "Moderate" if model["improvement"] > 0.05 else "Low")
        for n_ in dc["nulls"][:2]:
            ev(f"{esc(n_['a'])} ~ {esc(n_['b'])}", f"r = {n_['r']:.2f} ({n_['ci'][0]:.2f} to {n_['ci'][1]:.2f})", f"detects |r| &gt; {n_['min_detectable']:.2f} (80% power)", "Null result")
        if evidence and opt["include_evidence"]:
            story.append(H2("Evidence summary"))
            story.append(P("Each headline result with its key statistic, what it was checked against, and how far it can be trusted (see Evidence ratings in the methods)."))
            if "t_evidence" not in ref:
                counters["table"] += 1
                ref["t_evidence"] = counters["table"]
            story.append(table_block("t_evidence", "Evidence summary of headline findings", table(["Finding", "Key statistic", "Robustness / context", "Rating"], evidence, [0.24, 0.31, 0.33, 0.12]), None))
        story.append(H2("Conclusions"))
        concl = [f"This analysis set out to describe {esc(data_title)}, judge its reliability, and locate its strongest relationships, trends and differences. The data are of {esc(quality['label'].lower())} quality ({quality['score']}/100)"
                 + (f" and contain {corr['n_significant']} statistically reliable numeric relationship{'s' if corr['n_significant'] != 1 else ''} among {corr['tests']} pairs tested" if corr["tests"] else "") + "."]
        high = [d for d in strongest if d["rating"] == "High"]
        if high:
            concl.append("The best-supported conclusion is that " + "; and that ".join(f"{esc(d['a'])} and {esc(d['b'])} are {'positively' if d['r'] > 0 else 'negatively'} and {RS.strength_label(d['r'])}ly related (r = {d['r']:.2f}), a pattern that survived outlier removal, rank-based checks and subgroup tests" for d in high[:2]) + ".")
        elif strongest:
            concl.append(f"The leading association ({esc(strongest[0]['a'])} with {esc(strongest[0]['b'])}, r = {strongest[0]['r']:.2f}) is real but should be treated with caution because it was sensitive to at least one robustness check.")
        if trends_deep:
            t0 = trends_deep[0]
            concl.append(f"Over time, {esc(t0['measure'])} {'rose' if t0['slope'] > 0 else 'fell'} " + (f"by {abs(t0['change_pct']):.0f}% " if t0.get("change_pct") is not None else "") + f"with {'a reliable' if t0['p_tau'] < RS.ALPHA else 'no reliable'} monotonic trend"
                         + (f", and the movement is broad-based across {esc(t0['entities']['column'])} values" if t0.get("entities") and max(t0["entities"]["share_up"], t0["entities"]["share_down"]) >= 0.7 else "") + ".")
        if dc["nulls"] or dc["isolated"]:
            absent = [f"{esc(n_['a'])} and {esc(n_['b'])}" for n_ in dc["nulls"][:2]] + [esc(i_["column"]) for i_ in dc["isolated"][:1]]
            concl.append(f"Equally informative are the absences of association ({'; '.join(absent)}): these variables can be treated as independent dimensions, and no linear relationship of practical size between them was missed.")
        concl.append("All findings are exploratory and associative; they identify where to look next rather than establish causes.")
        story += [P(x_) for x_ in concl]
        recs_out: list[str] = []
        for i in fixable[:2]:
            recs_out.append(f"<b>Fix the data:</b> {esc(i['title'].rstrip('.'))} ({esc(', '.join(i.get('columns') or [])[:50] or 'whole table')}).")
        if top_corr and top_corr["significant"]:
            recs_out.append(f"<b>Follow up the strongest link</b> between {esc(top_corr['a'])} and {esc(top_corr['b'])} with a regression that adjusts for other variables" + (", and drop or combine redundant variables in the same cluster before modelling." if dc["clusters"] else "."))
        if top_group and top_group["significant"]:
            recs_out.append(f"<b>Segment by {esc(top_group['group'])}</b> in any further analysis: its groups differ enough in {esc(top_group['measure'])} that pooled averages may mislead.")
        if trends_deep and trends_deep[0].get("changepoint"):
            recs_out.append(f"<b>Investigate the change near {trends_deep[0]['changepoint']['at']:.0f}</b> in {esc(trends_deep[0]['measure'])}: it suggests the underlying process shifted.")
        if model:
            recs_out.append(f"<b>{'Improve' if model['improvement'] > 0.05 else 'Reconsider'} the predictive model</b> for {esc(model['target'])}: " + ("tune it and validate on a different period or source." if model["improvement"] > 0.05 else "the current variables add little, so seek additional predictors."))
        skewed = fresh.get("skewed_fields") or []
        if skewed:
            recs_out.append(f"<b>Transform skewed variables</b> ({esc(', '.join(skewed[:3]))}), for example with a log transform, before using them in models that assume symmetric inputs.")
        if is_sample:
            recs_out.append("<b>Repeat on the full data</b> before publishing or acting on any number in this report.")
        for q in (fresh.get("research_questions") or [])[:2]:
            recs_out.append(f"<b>Next question:</b> {esc(q)}")
        recs_out.append("<b>Validate with domain knowledge:</b> automated analysis shows what the data contain; explaining why needs context this report cannot supply.")
        story.append(H2("Recommendations"))
        story += bullets(recs_out[:8], numbered=True)

    # ---------------------------------------------------------------- 7 References
    if sec_on["references"]:
        story.append(H1("references", "References"))
        refs: list[str] = []
        if slug:
            refs.append(f"Kaggle (n.d.) {esc(display_title if not opt['title'] else data_title)}. https://www.kaggle.com/datasets/{esc(slug)}" + (f". License: {esc(lic)}." if lic else "."))
        refs += [
            "Breiman, L. (2001) Random forests. <i>Machine Learning</i> 45: 5-32.",
            "Cohen, J. (1988) <i>Statistical Power Analysis for the Behavioral Sciences</i>, 2nd edn. Lawrence Erlbaum, Hillsdale, NJ.",
            "Cram&eacute;r, H. (1946) <i>Mathematical Methods of Statistics</i>. Princeton University Press, Princeton, NJ.",
            "Dunn, O.J. (1961) Multiple comparisons among means. <i>Journal of the American Statistical Association</i> 56: 52-64.",
            "Holm, S. (1979) A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics</i> 6: 65-70.",
            "Kendall, M.G. (1938) A new measure of rank correlation. <i>Biometrika</i> 30: 81-93.",
            "Kruskal, W.H. and Wallis, W.A. (1952) Use of ranks in one-criterion variance analysis. <i>Journal of the American Statistical Association</i> 47: 583-621.",
            "McKinney, W. (2010) Data structures for statistical computing in Python. <i>Proceedings of the 9th Python in Science Conference</i>: 56-61.",
            "Pearson, K. (1895) Notes on regression and inheritance in the case of two parents. <i>Proceedings of the Royal Society of London</i> 58: 240-242.",
            "Pedregosa, F. et al. (2011) Scikit-learn: machine learning in Python. <i>Journal of Machine Learning Research</i> 12: 2825-2830.",
            "Shapiro, S.S. and Wilk, M.B. (1965) An analysis of variance test for normality. <i>Biometrika</i> 52: 591-611.",
            "Spearman, C. (1904) The proof and measurement of association between two things. <i>American Journal of Psychology</i> 15: 72-101.",
            "Tukey, J.W. (1977) <i>Exploratory Data Analysis</i>. Addison-Wesley, Reading, MA.",
            "Virtanen, P. et al. (2020) SciPy 1.0: fundamental algorithms for scientific computing in Python. <i>Nature Methods</i> 17: 261-272.",
        ]
        ref_style = ParagraphStyle("Ref", parent=small, fontSize=7.9, leading=10.4, leftIndent=12, firstLineIndent=-12, textColor=ink, spaceAfter=2.5)
        story += [P(r, ref_style) for r in refs]

    # ---------------------------------------------------------------- appendices
    if sec_on["appendices"]:
        story += [CondPageBreak(2.5 * inch), _Anchor("appendix_a", "Appendices"), P("Appendix A. Variable dictionary", h1)]
        documented = dataset_info.get("column_descriptions") or {}
        inferred = dataset_info.get("inferred_column_descriptions") or {}
        if not inferred and not documented:
            try:  # the dataset package may not be importable in every context; descriptions are a nicety
                from dataset.descriptions import infer_column_description

                inferred = {c_: infer_column_description(c_, df[c_]) for c_ in list(df.columns)[:45]}
            except Exception:
                inferred = {}
        arows = []
        for col in list(df.columns)[:45]:
            p = profiles[col]
            d_text = documented.get(col) or inferred.get(col) or ""
            mark = "" if documented.get(col) else (" (inferred)" if d_text else "")
            arows.append([esc(col), esc(ROLE_NAMES.get(p["role"], p["role"])), pct(float(df[col].isna().mean() * 100)), f"{p['unique_count']:,}", esc(_shorten(d_text, 120)) + esc(mark)])
        story.append(table(["Variable", "Role", "Missing", "Unique", "Description"], arows, [0.19, 0.15, 0.10, 0.08, 0.48], right={2, 3}))
        if cols_n > 45:
            story.append(P(f"Showing the first 45 of {cols_n} variables.", small))
        story.append(P("Descriptions marked (inferred) were deduced from the column name, not taken from the dataset's documentation.", small))
        story.append(P("Appendix B. Reproducibility record", h1))
        rec_rows = [["Data fingerprint (SHA-256 prefix of table content)", S["fingerprint"]], ["Rows x columns analysed", f"{rows_n:,} x {cols_n}"], ["Analysis date", datetime.now().strftime("%Y-%m-%d %H:%M")],
                    ["Software", esc(S["software"])], ["Significance level / adjustment", "0.05 / Bonferroni within each test family (Holm for post-hoc contrasts)"], ["Random seed", "42 (train/test split, random forest, sampling)"],
                    ["Report detail level", detail]]
        story.append(table(["Item", "Value"], [[a, b] for a, b in rec_rows], [0.42, 0.58]))
        if transform_log:
            story.append(P("Preparation steps applied:", ParagraphStyle("pl2", parent=body, spaceBefore=6, spaceAfter=2, keepWithNext=1)))
            story += bullets([esc(t["description"]) for t in transform_log[:40]], numbered=True)

    # ---------------------------------------------------------------- page furniture
    footer_text = _safe(opt["footer"]) or "Generated by DataScopia - exploratory analysis; see the limitations in the Discussion"

    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saved: list[dict[str, Any]] = []
            self._pending_bookmarks: list[tuple[str, str, int, int]] = []

        def showPage(self) -> None:  # defer so the total page count is known when drawing footers
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def registerBookmark(self, key: str, title: str, level: int) -> None:
            self._pending_bookmarks.append((key, title, level, self._pageNumber))

        def save(self) -> None:
            total = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                self._decorate(total)
                for key, title, level, page_no in self._pending_bookmarks:
                    if page_no == self._pageNumber:
                        self.bookmarkPage(key)
                        self.addOutlineEntry(title, key, level=level, closed=0)
                super().showPage()
            super().save()

        def _decorate(self, total: int) -> None:
            page = self._pageNumber
            self.saveState()
            self.setStrokeColor(rule)
            self.setLineWidth(0.5)
            if page > 1:
                self.line(0.85 * inch, PAGE[1] - 0.55 * inch, PAGE[0] - 0.85 * inch, PAGE[1] - 0.55 * inch)
                self.setFont("Helvetica", 7.6)
                self.setFillColor(muted)
                self.drawString(0.85 * inch, PAGE[1] - 0.48 * inch, _safe(display_title)[:70])
                self.drawRightString(PAGE[0] - 0.85 * inch, PAGE[1] - 0.48 * inch, "Data Analysis Report")
            self.line(0.85 * inch, 0.58 * inch, PAGE[0] - 0.85 * inch, 0.58 * inch)
            self.setFont("Helvetica", 7.6)
            self.setFillColor(muted)
            self.drawString(0.85 * inch, 0.42 * inch, footer_text[:110])
            self.drawRightString(PAGE[0] - 0.85 * inch, 0.42 * inch, f"Page {page} of {total}")
            self.restoreState()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=PAGE, leftMargin=0.85 * inch, rightMargin=0.85 * inch, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title=_safe(display_title), author=_safe(opt["author"]) or "DataScopia", subject="Data analysis report", creator="DataScopia")
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
