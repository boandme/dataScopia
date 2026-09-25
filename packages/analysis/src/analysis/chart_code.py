"""Turn a chart configuration into runnable pandas + Plotly code.

The workspace draws charts server-side (``charts.py``); the notebook export needs the same
picture as *code a reader can run and edit*. Each generator here emits a self-contained snippet
(assuming ``df``, ``pd``, ``np``, ``px``, ``go`` and ``stats`` exist) that reproduces the chart's
computation - aggregation, top-N folding, trend line, granularity - in plain pandas, plus the
statistic the workspace reported (correlation, group difference, association) as a printed line.
"""

from __future__ import annotations

from typing import Any

_FREQ = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
_MEASURE = {"numeric", "ordinal_numeric", "binary_numeric"}
_GROUPING = {"categorical", "binary", "ordinal", "ordinal_numeric", "geo_country", "geo_state"}


def _role(roles: dict[str, Any], column: str | None) -> str:
    if not column:
        return "none"
    entry = roles.get(column) or {}
    return entry.get("role", "unknown") if isinstance(entry, dict) else "unknown"


def _numeric_expr(roles: dict[str, Any], column: str) -> str:
    """Column as numbers, reading '$1,299'-style text if that's how it is stored."""
    entry = roles.get(column) or {}
    if isinstance(entry, dict) and (entry.get("notes") or {}).get("stored_as_text"):
        return (f"pd.to_numeric(df[{column!r}].astype(str).str.replace(r'[,$€£¥%\\s]', '', regex=True)"
                ", errors='coerce')")
    return f"df[{column!r}]"


def _agg(agg: str | None, has_measure: bool) -> str:
    if not agg or agg == "none":
        return "mean" if has_measure else "count"
    if agg == "count":
        return "count"
    return agg if has_measure else "count"


def _title(config: dict[str, Any], default: str) -> str:
    return (config.get("customTitle") or default).strip()


def chart_code(config: dict[str, Any], roles: dict[str, Any], measures: list[str] | None = None) -> str:
    """Runnable code for one chart. Never raises: an unsupported/incomplete config yields a
    comment explaining what's missing, so a notebook cell is never broken by one bad chart."""
    try:
        return _chart_code(config, roles, measures or [])
    except Exception as exc:  # pragma: no cover - defensive
        return f"# Could not generate code for this chart ({type(exc).__name__}: {exc}).\n"


def _chart_code(config: dict[str, Any], roles: dict[str, Any], measures: list[str]) -> str:
    ctype = config.get("type") or "histogram"
    x, y = config.get("x") or None, config.get("y") or None
    group = config.get("group") or None
    agg = config.get("agg")
    sort = config.get("sort") or "none"
    lines: list[str] = []

    def emit(*more: str) -> None:
        lines.extend(more)

    if ctype in ("histogram",):
        if not x:
            return "# Pick a field for the histogram.\n"
        if _role(roles, x) not in _MEASURE:
            return _chart_code({**config, "type": "bar", "agg": "count"}, roles, measures)
        emit(f"values = {_numeric_expr(roles, x)}.dropna()",
             f"fig = px.histogram(df.assign(_v=values), x='_v', nbins=40, {'color=' + repr(group) + ', barmode=' + repr('overlay') + ', opacity=0.65, ' if group else ''}"
             f"title={_title(config, f'Distribution of {x}')!r}, labels={{'_v': {x!r}}})",
             "print(f'n = {len(values):,}; mean {values.mean():.4g}, median {values.median():.4g}, skewness {values.skew():.2f}')")

    elif ctype in ("bar", "stacked_bar"):
        if not x:
            return "# Pick a field for the categories.\n"
        has_y = bool(y) and _role(roles, y) in _MEASURE
        func = _agg(agg, has_y)
        by = [x] + ([group] if group and group != x else [])
        if func == "count" or not has_y:
            emit(f"table = df.groupby({by!r}).size().rename('value').reset_index()")
            label = "Number of rows"
        else:
            emit(f"table = df.assign(_y={_numeric_expr(roles, y)}).groupby({by!r})['_y'].{func}().rename('value').reset_index()")
            label = f"{func.title()} of {y}"
        emit("totals = table.groupby(" + repr(x) + ")['value'].sum().sort_values(ascending=" + str(sort == 'asc') + ")",
             "top = totals.index[:15]   # keep the 15 largest categories so the chart stays readable",
             f"fig = px.bar(table[table[{x!r}].isin(top)], x={x!r}, y='value'"
             + (f", color={group!r}, barmode={'stack' if ctype == 'stacked_bar' else 'group'!r}" if group and group != x else "")
             + f", title={_title(config, f'{label} by {x}')!r}, labels={{'value': {label!r}}})",
             f"fig.update_xaxes(categoryorder='total {'ascending' if sort == 'asc' else 'descending'}')")

    elif ctype == "pie":
        if not x:
            return "# Pick a field for the slices.\n"
        emit(f"counts = df[{x!r}].value_counts()",
             "if len(counts) > 8:   # fold the long tail into 'Other'",
             "    counts = pd.concat([counts.iloc[:8], pd.Series({'Other': counts.iloc[8:].sum()})])",
             f"fig = px.pie(names=counts.index.astype(str), values=counts.values, hole=0.45, title={_title(config, f'Share of rows by {x}')!r})")

    elif ctype == "scatter":
        if not x or not y:
            return "# A scatter plot needs two numeric fields (X and Y).\n"
        if _role(roles, x) not in _MEASURE | {"geo_lat", "geo_lon", "year"} or _role(roles, y) not in _MEASURE | {"geo_lat", "geo_lon", "year"}:
            return _chart_code({**config, "type": "box"}, roles, measures)
        emit(f"plot_df = pd.DataFrame({{{x!r}: {_numeric_expr(roles, x)}, {y!r}: {_numeric_expr(roles, y)}"
             + (f", {group!r}: df[{group!r}]" if group else "") + "}).dropna()",
             "if len(plot_df) > 5000:   # draw a sample; the statistics below use every row",
             "    shown = plot_df.sample(5000, random_state=7)",
             "else:",
             "    shown = plot_df",
             f"fig = px.scatter(shown, x={x!r}, y={y!r}, opacity=0.65" + (f", color={group!r}" if group else "") + f", title={_title(config, f'{y} vs {x}')!r})",
             f"slope, intercept = np.polyfit(plot_df[{x!r}], plot_df[{y!r}], 1)",
             f"x_range = [plot_df[{x!r}].min(), plot_df[{x!r}].max()]",
             "fig.add_scatter(x=x_range, y=[intercept + slope * v for v in x_range], mode='lines', name='Linear trend', line=dict(dash='dash'))",
             f"r, p = stats.pearsonr(plot_df[{x!r}], plot_df[{y!r}])",
             f"rho, _ = stats.spearmanr(plot_df[{x!r}], plot_df[{y!r}])",
             "print(f'Pearson r = {r:.2f} (R^2 = {r*r:.2f}, p = {p:.3g}); Spearman rho = {rho:.2f}; n = {len(plot_df):,}')")

    elif ctype in ("line", "area", "timeseries"):
        if not x:
            return "# Pick a field for the horizontal axis.\n"
        is_time = _role(roles, x) in ("datetime", "year")
        has_y = bool(y)
        func = _agg(agg, has_y)
        freq = _FREQ.get(config.get("granularity") or "none")
        emit(f"plot_df = df[[{x!r}" + (f", {y!r}" if has_y else "") + (f", {group!r}" if group else "") + "]].dropna(subset=[" + repr(x) + "]).copy()")
        if _role(roles, x) == "datetime":
            emit(f"plot_df[{x!r}] = pd.to_datetime(plot_df[{x!r}], errors='coerce')", f"plot_df = plot_df.dropna(subset=[{x!r}])")
        keys = [f"pd.Grouper(key={x!r}, freq={freq!r})" if freq and _role(roles, x) == "datetime" else repr(x)]
        if group:
            keys.append(repr(group))
        target = f"['{y}']" if has_y else ""
        if func == "count" or not has_y:
            emit(f"series = plot_df.groupby([{', '.join(keys)}]).size().rename('value').reset_index()")
            label = "Number of rows"
        else:
            emit(f"series = plot_df.groupby([{', '.join(keys)}]){target}.{func}().rename('value').reset_index()")
            label = f"{func.title()} of {y}"
        if group:
            emit(f"keep = series.groupby({group!r})['value'].sum().abs().nlargest(8).index   # the 8 largest series",
                 f"series = series[series[{group!r}].isin(keep)]")
        emit(f"fig = px.{'area' if ctype == 'area' else 'line'}(series.sort_values({x!r}), x={x!r}, y='value'" + (f", color={group!r}" if group else "")
             + f", markers=len(series) < 60, title={_title(config, f'{label} over {x}' if is_time else f'{label} by {x}')!r}, labels={{'value': {label!r}}})")

    elif ctype in ("box", "violin"):
        if not y and x and _role(roles, x) in _MEASURE:
            x, y = None, x
        if not y:
            return "# Pick a numeric field for the values (Y).\n"
        emit(f"plot_df = pd.DataFrame({{'value': {_numeric_expr(roles, y)}" + (f", {x!r}: df[{x!r}]" if x else "") + "}).dropna()")
        if x:
            emit(f"top = plot_df[{x!r}].value_counts().index[:12]   # the 12 largest groups",
                 f"plot_df = plot_df[plot_df[{x!r}].isin(top)]")
        emit(f"fig = px.{ctype}(plot_df, " + (f"x={x!r}, " if x else "") + f"y='value', {'points=' + repr('outliers') + ', ' if ctype == 'box' else ''}title={_title(config, f'{y} by {x}' if x else f'Distribution of {y}')!r}, labels={{'value': {y!r}}})")
        if x:
            emit(f"groups = [g['value'].to_numpy() for _, g in plot_df.groupby({x!r}) if len(g) >= 2]",
                 "if len(groups) >= 2:",
                 "    h, p = stats.kruskal(*groups)",
                 f"    print(f'Kruskal-Wallis across {x} groups: H = {{h:.2f}}, p = {{p:.3g}}')")

    elif ctype == "heatmap":
        cols = [c for c in measures if _role(roles, c) in _MEASURE][:14]
        emit(f"cols = {cols!r}",
             "cols = [c for c in cols if c in df.columns]",
             "corr = df[cols].apply(pd.to_numeric, errors='coerce').corr()",
             f"fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', text_auto='.2f', title={_title(config, 'Correlation matrix')!r})",
             "# Strongest pairs, for reading the matrix:",
             "pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().sort_values(key=abs, ascending=False)",
             "print(pairs.head(5).round(2).to_string())")

    elif ctype == "distribution":
        if not x or not group:
            return _chart_code({**config, "type": "histogram"}, roles, measures)
        emit(f"plot_df = pd.DataFrame({{'value': {_numeric_expr(roles, x)}, {group!r}: df[{group!r}]}}).dropna()",
             f"top = plot_df[{group!r}].value_counts().index[:3]   # overlapping curves: keep 3 groups distinguishable",
             f"plot_df = plot_df[plot_df[{group!r}].isin(top)]",
             f"fig = px.histogram(plot_df, x='value', color={group!r}, histnorm='probability density', barmode='overlay', opacity=0.6, nbins=40, title={_title(config, f'{x} by {group}')!r}, labels={{'value': {x!r}}})",
             f"groups = [g['value'].to_numpy() for _, g in plot_df.groupby({group!r}) if len(g) >= 2]",
             "if len(groups) >= 2:",
             "    h, p = stats.kruskal(*groups)",
             f"    print(f'Kruskal-Wallis: p = {{p:.3g}} ({{\"groups differ\" if p < 0.05 else \"no significant difference\"}})')")

    elif ctype == "target":
        if not x or not y:
            return "# Pick a feature (X) and a target (Y).\n"
        t_role, f_role = _role(roles, y), _role(roles, x)
        t_class = t_role in {"binary", "binary_numeric", "categorical", "ordinal", "ordinal_numeric"}
        f_class = f_role in _GROUPING
        if not t_class:
            return _chart_code({**config, "type": "scatter" if not f_class else "box", "group": None}, roles, measures)
        if not f_class:
            return _chart_code({**config, "type": "box", "x": y, "y": x}, roles, measures)
        emit(f"plot_df = df[[{x!r}, {y!r}]].dropna()",
             f"share = pd.crosstab(plot_df[{x!r}], plot_df[{y!r}], normalize='index').mul(100).reset_index().melt(id_vars={x!r}, var_name={y!r}, value_name='% of rows')",
             f"fig = px.bar(share, x={x!r}, y='% of rows', color={y!r}, barmode='stack', title={_title(config, f'{y} by {x}')!r})",
             f"table = pd.crosstab(plot_df[{x!r}], plot_df[{y!r}])",
             "chi2, p, _, _ = stats.chi2_contingency(table, correction=False)",
             "v = (chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))) ** 0.5",
             "print(f\"Cramer's V = {v:.2f}; chi-square p = {p:.3g}\")")

    elif ctype == "treemap":
        if not x:
            return "# Pick a categorical field for the treemap tiles.\n"
        has_y = bool(y) and _role(roles, y) in _MEASURE
        path = ([group] if group and group != x else []) + [x]
        if has_y:
            emit(f"tiles = df.assign(_y={_numeric_expr(roles, y)}).groupby({path!r}).agg(rows=({x!r}, 'size'), value=('_y', {_agg(agg, True)!r})).reset_index()",
                 f"fig = px.treemap(tiles.nlargest(40, 'rows'), path={path!r}, values='rows', color='value', color_continuous_scale='Blues', title={_title(config, f'{x} breakdown')!r})")
        else:
            emit(f"tiles = df.groupby({path!r}).size().rename('rows').reset_index()",
                 f"fig = px.treemap(tiles.nlargest(40, 'rows'), path={path!r}, values='rows', title={_title(config, f'{x} breakdown')!r})")

    elif ctype == "map":
        has_y = bool(y) and _role(roles, y) in _MEASURE
        role = _role(roles, x)
        if role in ("geo_country", "geo_state"):
            func = _agg(agg, has_y)
            kind = "USA-states" if role == "geo_state" else None
            notes = (roles.get(x) or {}).get("notes", {}) if isinstance(roles.get(x), dict) else {}
            mode = "USA-states" if kind else ("ISO-3" if notes.get("code_kind") == "iso3" else "country names")
            if has_y and func != "count":
                emit(f"table = df.assign(_y={_numeric_expr(roles, y)}).groupby({x!r})['_y'].{func}().rename('value').reset_index()")
            else:
                emit(f"table = df.groupby({x!r}).size().rename('value').reset_index()")
            emit(f"fig = px.choropleth(table, locations={x!r}, locationmode={mode!r}, color='value', color_continuous_scale='Blues', "
                 + ("scope='usa', " if kind else "") + f"title={_title(config, f'{y or 'Rows'} by {x}')!r})")
        else:
            return "# Map: the workspace used latitude/longitude columns.\nfig = px.scatter_geo(df.dropna(subset=[" + repr(x or 'lat') + "]).head(3000), lat=" + repr(x or 'lat') + ", lon='lon')\n"

    elif ctype == "missing":
        emit("missing = df.isna().mean().mul(100).round(2)",
             "missing = missing[missing > 0].sort_values(ascending=False).head(25)",
             "fig = px.bar(missing[::-1], orientation='h', labels={'value': '% of rows missing', 'index': 'column'}, title='Missing values by column') if len(missing) else None",
             "print(missing.to_string() if len(missing) else 'No missing values.')")

    elif ctype == "pair":
        cols = [c for c in measures if _role(roles, c) == "numeric"][:5]
        emit(f"cols = [c for c in {cols!r} if c in df.columns]",
             "sample = df[cols].apply(pd.to_numeric, errors='coerce').dropna()",
             "sample = sample.sample(min(len(sample), 1200), random_state=7)",
             "fig = px.scatter_matrix(sample, dimensions=cols, title='Pair plot')",
             "fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.6))")
    else:
        return f"# Chart type {ctype!r} has no code generator.\n"

    if not lines or "fig =" not in "\n".join(lines) and ctype != "missing":
        return "\n".join(lines) + "\n"
    if ctype == "missing":
        lines.append("if fig is not None:\n    fig.show()")
    else:
        lines.append("fig.show()")
    return "\n".join(lines) + "\n"
