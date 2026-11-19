"""Deterministic data analysis: every number is computed in Python and labelled with the method used
(sample vs population formulas, quartile method, skewness definition). No LLM touches a statistic."""
import ast
import csv
import hashlib
import io
import json
import math
import operator
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.tools import tool

MISSING_TOKENS = {"", "n/a", "na", "nan", "null", "none", "-", "--", "?"}
SMALL_SAMPLE = 8            # below this many points shape statistics (skewness) mean very little


def _num(x, digits: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, digits)


def fmt(x, digits: int = 2) -> str:
    """1520.0 -> '1,520' ; 443.8468 -> '443.85' ; 0.99488 (digits=4) -> '0.9949'"""
    if x is None:
        return "n/a"
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------------------------------
# statistics with method metadata
# ------------------------------------------------------------------------------------------
def describe_series(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"error": "no values"}
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    uniq, counts = np.unique(arr, return_counts=True)
    mode = [float(v) for v, c in zip(uniq, counts) if c == counts.max()] if counts.max() > 1 else []
    centred = arr - arr.mean()
    m2, m3 = float((centred ** 2).mean()), float((centred ** 3).mean())
    g1 = m3 / m2 ** 1.5 if m2 > 0 else 0.0                       # population skewness
    skew_adjusted = g1 * math.sqrt(n * (n - 1)) / (n - 2) if n > 2 else None   # Fisher-Pearson G1
    return {
        "n": n,
        "sum": _num(arr.sum()), "mean": _num(arr.mean()), "median": _num(np.median(arr)),
        "mode": [_num(m) for m in mode],                         # [] = no value repeats
        "std_sample": _num(arr.std(ddof=1)) if n > 1 else None,
        "std_population": _num(arr.std(ddof=0)),
        "variance_sample": _num(arr.var(ddof=1)) if n > 1 else None,
        "variance_population": _num(arr.var(ddof=0)),
        "min": _num(arr.min()), "max": _num(arr.max()), "range": _num(arr.max() - arr.min()),
        "q1": _num(q1), "q3": _num(q3), "iqr": _num(iqr),
        "skewness_adjusted": _num(skew_adjusted), "skewness_population": _num(g1),
        "outliers": [_num(v) for v in arr if v < low or v > high],
        "methods": {
            "std_variance": "sample (n-1) is the headline value; population (n) given for comparison",
            "quartiles": "linear interpolation (numpy default)",
            "outliers": "IQR rule: below Q1-1.5*IQR or above Q3+1.5*IQR",
            "skewness": "sample-adjusted Fisher-Pearson G1 (headline); population g1 for comparison",
            "small_sample_warning": n < SMALL_SAMPLE,
        },
    }


def trend_report(values: List[float], labels: Optional[List[str]] = None, periods_ahead: int = 3) -> dict:
    """Ordinary least squares trend of an ORDERED series (x = 0,1,2,...), step-to-step changes, linear forecast."""
    if len(values) < 3:
        return {"error": "need at least 3 values"}
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res, ss_tot = float(((y - predicted) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    steps = []
    for i in range(1, len(y)):
        change = y[i] - y[i - 1]
        steps.append({
            "from": labels[i - 1] if labels else str(i - 1), "to": labels[i] if labels else str(i),
            "change": _num(change),
            "pct": _num(change / abs(y[i - 1]) * 100, 2) if y[i - 1] != 0 else None,
        })
    pcts = [s["pct"] for s in steps if s["pct"] is not None]
    periods_ahead = max(0, min(int(periods_ahead), 24))
    return {
        "n": int(len(y)),
        "slope_per_step": _num(slope), "intercept": _num(intercept), "r_squared": _num(r2),
        "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
        "steps": steps,
        "avg_step_pct": _num(np.mean(pcts), 2) if pcts else None,
        "total_change": _num(y[-1] - y[0]),
        "total_pct_change": _num((y[-1] - y[0]) / abs(y[0]) * 100, 2) if y[0] != 0 else None,
        "linear_forecast": [_num(slope * (len(y) + i) + intercept) for i in range(periods_ahead)],
        "methods": {
            "model": "ordinary least squares, x = 0,1,2,... for the 1st, 2nd, 3rd ... value (values assumed in time order)",
            "intercept": "fitted value at x = 0 (the first value)",
            "avg_step_pct": "arithmetic mean of the step-to-step percentage changes",
            "forecast": "linear extrapolation - an estimate, not a fact",
        },
    }


# ------------------------------------------------------------------------------------------
# dataset detection (provenance) and data-quality checks
# ------------------------------------------------------------------------------------------
def _to_number_or_text(tokens: List[str]):
    """-> (values, kind) where kind is 'numeric' | 'text' | 'mixed'. Missing tokens become None."""
    values, numeric, text = [], 0, 0
    for token in tokens:
        token = (token or "").strip()
        if token.lower() in MISSING_TOKENS:
            values.append(None)
            continue
        try:
            values.append(float(token))
            numeric += 1
        except ValueError:
            values.append(token)
            text += 1
    if numeric and not text:
        return values, "numeric"
    if numeric and numeric >= text:
        return [None if isinstance(v, float) else v for v in values], "mixed"
    return values, "text"


def parse_dataset(text: str) -> Optional[dict]:
    """Find tabular data in free text: JSON, 'Name: v1, v2, v3' lines, or CSV with/without header.
    Returns {'format', 'columns': {name: values}, 'kinds': {name: kind}} or None when there is no
    usable numeric dataset (at least one numeric column with 3+ numbers)."""
    text = (text or "").strip()
    if not text:
        return None
    raw_columns, fmt_name = {}, None

    try:                                                         # JSON
        data = json.loads(text)
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            raw_columns, fmt_name = {k: [str(x) if x is not None else "" for x in v] for k, v in data.items()}, "json"
        elif isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
            keys = list(data[0].keys())
            raw_columns = {k: [str(r.get(k, "")) for r in data] for k in keys}
            fmt_name = "json"
        elif isinstance(data, list) and data and all(isinstance(v, (int, float)) for v in data):
            raw_columns, fmt_name = {"values": [str(v) for v in data]}, "json"
    except (ValueError, TypeError):
        pass

    if not raw_columns:                                          # 'Name: v1, v2, v3'
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, values = line.split(":", 1)
            parts = [v.strip() for v in values.split(",")]
            if name.strip() and len([p for p in parts if p]) >= 2:
                raw_columns[name.strip()] = parts
        if raw_columns:
            fmt_name = "label: values"

    if not raw_columns:                                          # CSV
        rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
        if len(rows) >= 3:
            first = rows[0]
            has_header = not all(_to_number_or_text([c])[1] == "numeric" for c in first if c.strip())
            header, body = (first, rows[1:]) if has_header else ([f"column_{i+1}" for i in range(len(first))], rows)
            for j, name in enumerate(header):
                raw_columns[name.strip() or f"column_{j+1}"] = [row[j] if j < len(row) else "" for row in body]
            fmt_name = "csv"

    columns, kinds = {}, {}
    for name, tokens in raw_columns.items():
        columns[name], kinds[name] = _to_number_or_text(tokens)
    if not any(kinds[n] == "numeric" and sum(v is not None for v in columns[n]) >= 3 for n in columns):
        return None
    return {"format": fmt_name, "columns": columns, "kinds": kinds}


def locate_dataset(candidates: List[Tuple[str, str]]) -> Optional[dict]:
    """candidates: [(label, text)] in priority order. The first one that holds a numeric dataset wins,
    and the result records exactly where the data came from."""
    for label, text in candidates:
        dataset = parse_dataset(text)
        if dataset:
            body = (text or "").strip()
            dataset.update({"origin": label, "sha256_12": hashlib.sha256(body.encode("utf-8")).hexdigest()[:12],
                            "chars": len(body)})
            return dataset
    return None


def data_quality(dataset: dict) -> dict:
    cols, kinds = dataset["columns"], dataset["kinds"]
    n_rows = max(len(v) for v in cols.values())
    missing = {n: sum(v is None for v in vals) + (n_rows - len(vals)) for n, vals in cols.items()}
    rows = list(zip(*[vals + [None] * (n_rows - len(vals)) for vals in cols.values()]))
    duplicates = len(rows) - len(set(rows))
    numeric_cols = [n for n, k in kinds.items() if k == "numeric"]
    outliers = {}
    for n in numeric_cols:
        clean = [v for v in cols[n] if v is not None]
        if len(clean) >= 4:
            stats = describe_series(clean)
            outliers[n] = stats["outliers"]
    return {"rows": n_rows, "missing": missing, "duplicate_rows": duplicates,
            "mixed_columns": [n for n, k in kinds.items() if k == "mixed"],
            "equal_lengths": len({len(v) for v in cols.values()}) == 1, "outliers": outliers}


def build_data_report(dataset: dict) -> Tuple[str, dict]:
    """The full deterministic report (text) and the computed metrics. Starts with the data's provenance."""
    cols, kinds = dataset["columns"], dataset["kinds"]
    numeric = [n for n in cols if kinds[n] == "numeric" and sum(v is not None for v in cols[n]) >= 3][:4]
    label_col = next((n for n in cols if kinds[n] == "text" and len(cols[n]) == len(cols[numeric[0]])), None)
    labels = [str(v) for v in cols[label_col]] if label_col else None
    quality = data_quality(dataset)
    lines = [
        f"DATASET PROVENANCE: {dataset.get('origin', 'unknown')}; input hash {dataset.get('sha256_12', 'n/a')}; "
        f"{quality['rows']} rows; columns: " + ", ".join(f"{n} ({kinds[n]})" for n in cols),
        "",
        "DATA QUALITY (cleaning checks):",
        f"- Rows: {quality['rows']}; missing values: " + (", ".join(f"{n}={c}" for n, c in quality['missing'].items()) or "none")
        + f"; duplicate rows: {quality['duplicate_rows']}; all columns equally long: {'yes' if quality['equal_lengths'] else 'NO'}",
        "- Mixed number/text columns: " + (", ".join(quality['mixed_columns']) or "none"),
    ]
    metrics = {}
    for name in numeric:
        values = [v for v in cols[name] if v is not None]
        stats, trend = describe_series(values), trend_report(values, labels if labels and len(labels) == len(values) else None)
        metrics[name] = {"stats": stats, "trend": trend, "values": values}
        out = quality["outliers"].get(name)
        lines += [
            f"- Outliers in {name} (IQR rule): " + ("none" if not out else ", ".join(fmt(o) for o in out))
            + (" (small sample - treat with caution)" if stats["methods"]["small_sample_warning"] else ""),
            "",
            f'STATISTICS for "{name}" (n={stats["n"]}):',
            "- Values in order: " + "; ".join(fmt(v) for v in values),
            f"- Sum: {fmt(stats['sum'])}; Mean: {fmt(stats['mean'])}; Median: {fmt(stats['median'])}",
            "- Mode: " + ("no mode (all values unique)" if not stats["mode"] else ", ".join(fmt(m) for m in stats["mode"])),
            f"- Standard deviation (sample, n-1): {fmt(stats['std_sample'])} [population, n: {fmt(stats['std_population'])}]",
            f"- Variance (sample, n-1): {fmt(stats['variance_sample'])} [population: {fmt(stats['variance_population'])}]",
            f"- Min: {fmt(stats['min'])}; Max: {fmt(stats['max'])}; Range: {fmt(stats['range'])}",
            f"- Q1: {fmt(stats['q1'])}; Q3: {fmt(stats['q3'])}; IQR: {fmt(stats['iqr'])} (linear-interpolation quartiles)",
            f"- Skewness (sample-adjusted Fisher-Pearson G1): {fmt(stats['skewness_adjusted'])} "
            f"[population g1: {fmt(stats['skewness_population'])}]"
            + (f"; n={stats['n']} is too few points to judge the distribution shape" if stats["methods"]["small_sample_warning"] else ""),
        ]
        if "error" not in trend:
            lines += [
                "",
                f'TREND for "{name}" (assumes the values are in time order; x = 0,1,2,... for the 1st,2nd,3rd value):',
                f"- Ordinary least squares: slope {fmt(trend['slope_per_step'])} per step; intercept {fmt(trend['intercept'])} "
                f"(fitted value at the first step); R-squared {fmt(trend['r_squared'], 4)}; direction: {trend['direction']}",
                "- Step-to-step changes: " + "; ".join(
                    f"{s['from']}->{s['to']}: {'+' if (s['change'] or 0) >= 0 else ''}{fmt(s['change'])}"
                    + (f" ({fmt(s['pct'])}%)" if s['pct'] is not None else "") for s in trend["steps"]),
                f"- Average step-to-step % change (arithmetic mean of the step percentages): {fmt(trend['avg_step_pct'])}%",
                f"- Total change first to last: {'+' if (trend['total_change'] or 0) >= 0 else ''}{fmt(trend['total_change'])}"
                + (f" ({fmt(trend['total_pct_change'])}%)" if trend['total_pct_change'] is not None else ""),
                "- Linear forecast, next steps (extrapolation, an estimate not a fact): "
                + "; ".join(fmt(v) for v in trend["linear_forecast"]),
            ]
    return "\n".join(lines), metrics


def template_interpretation(metrics: dict) -> str:
    """Plain-language reading built in code from the computed numbers (used when the LLM is unavailable
    or writes a number that is not in the report)."""
    if not metrics:
        return ""
    name, m = next(iter(metrics.items()))
    stats, trend, values = m["stats"], m["trend"], m["values"]
    parts = [f"{name} goes from {fmt(values[0])} to {fmt(values[-1])} over {stats['n']} points"]
    if trend.get("total_pct_change") is not None:
        parts[0] += f", a total change of {fmt(trend['total_pct_change'])}%"
    parts[0] += "."
    if "error" not in trend:
        parts.append(f"A straight-line trend ({trend['direction']}) explains {fmt(trend['r_squared'], 4)} of the variation "
                     f"(R-squared); the slope is {fmt(trend['slope_per_step'])} per step.")
    if stats["methods"]["small_sample_warning"]:
        parts.append(f"With only {stats['n']} points the shape statistics (such as skewness) say little, "
                     f"and the forecast is only an extrapolation of the straight line.")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------
# LangChain tools (kept for tool-calling agents and direct use)
# ------------------------------------------------------------------------------------------
@tool
def analyse_numbers(numbers: List[float]) -> dict:
    """Descriptive statistics of a list of numbers with the METHOD of each statistic: count, sum, mean, median,
    mode (empty if no value repeats), sample AND population std/variance, min, max, range, quartiles, IQR,
    adjusted and population skewness, outliers (IQR rule)."""
    if not numbers:
        return {"error": "numbers is empty"}
    return describe_series(numbers)


@tool
def parse_table(text: str) -> dict:
    """Turn user-provided data (text) into named columns. Understands lines like 'Month: Jan, Feb' /
    'Downloads: 1000, 1200', CSV text with or without a header row, and JSON. Missing values become null."""
    dataset = parse_dataset(text)
    if not dataset:
        return {"error": "no numeric dataset found (need a column with at least 3 numbers)"}
    cols = dataset["columns"]
    return {"format": dataset["format"], "columns": cols, "lengths": {k: len(v) for k, v in cols.items()},
            "numeric_columns": [k for k, v in dataset["kinds"].items() if v == "numeric"],
            "equal_lengths": len({len(v) for v in cols.values()}) == 1}


@tool
def describe_csv(csv_text: str) -> dict:
    """Profile a CSV text: shape, column types, missing values, numeric summary and the first 5 rows."""
    import pandas as pd                                    # imported here: pandas is slow to load
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return {"error": f"could not read CSV: {e}"}
    return {
        "rows": int(df.shape[0]), "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
        "numeric_summary": json.loads(df.describe().to_json()) if not df.select_dtypes("number").empty else {},
        "head": json.loads(df.head(5).to_json(orient="records")),
    }


@tool
def trend_analysis(values: List[float], periods_ahead: int = 3) -> dict:
    """Trend of an ordered series: OLS slope/intercept/R-squared, direction, step-to-step % changes, average step %,
    total % change and a LINEAR extrapolation (an estimate). Each result carries its method description."""
    return trend_report(values, None, periods_ahead)


@tool
def correlation(x: List[float], y: List[float], method: str = "pearson") -> dict:
    """Correlation between two equally long numeric lists. method: pearson or spearman."""
    if len(x) != len(y):
        return {"error": f"lists have different lengths ({len(x)} vs {len(y)})"}
    if len(x) < 3:
        return {"error": "need at least 3 pairs"}
    if method not in ("pearson", "spearman"):
        return {"error": "method must be 'pearson' or 'spearman'"}
    if np.std(x) == 0 or np.std(y) == 0:
        return {"error": "correlation undefined (a list has no variation)"}
    if method == "pearson":
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        import pandas as pd
        r = float(pd.Series(x, dtype=float).corr(pd.Series(y, dtype=float), method="spearman"))
    strength = abs(r)
    label = ("very strong" if strength >= 0.8 else "strong" if strength >= 0.6
             else "moderate" if strength >= 0.4 else "weak" if strength >= 0.2 else "very weak")
    return {"method": method, "r": _num(r), "strength": label,
            "direction": "positive" if r > 0 else "negative" if r < 0 else "none", "n": len(x)}


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
          "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
          "pow": pow, "floor": math.floor, "ceil": math.ceil}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> dict:
    """Safely evaluate a math expression, e.g. '(2100-1000)/1000*100' or 'sqrt(144)+log10(1000)'.
    Supports + - * / // % **, parentheses, pi, e and sqrt/log/log10/exp/abs/round/min/max/sum/floor/ceil."""
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
        return {"expression": expression, "result": _num(result, 6) if isinstance(result, (int, float)) else result}
    except ZeroDivisionError:
        return {"error": "division by zero"}
    except Exception as e:
        return {"error": f"cannot evaluate '{expression}': {e}"}


DATA_TOOLS = [analyse_numbers, parse_table, describe_csv, trend_analysis, correlation, calculate]
