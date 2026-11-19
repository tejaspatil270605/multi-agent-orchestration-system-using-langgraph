"""Deterministic statistics with method metadata: python -m pytest tests/test_data_stats.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.data_analysis_tools import (describe_series, trend_report, parse_dataset, locate_dataset, data_quality,
                                       build_data_report, template_interpretation, fmt)
from tools.writing_tools import find_unsupported_numbers

DEMO = """ Python usage dataset:
 Month: Jan, Feb, Mar, Apr, May
 PyPI downloads: 1000, 1200, 1500, 1800, 2100"""


def test_sample_vs_population_and_methods():
    s = describe_series([1000, 1200, 1500, 1800, 2100])
    assert round(s["std_sample"], 2) == 443.85 and round(s["std_population"], 2) == 396.99
    assert s["variance_sample"] == 197000 and s["variance_population"] == 157600
    assert s["skewness_adjusted"] == 0.2081 and s["skewness_population"] == 0.1396
    assert s["q1"] == 1200 and s["q3"] == 1800 and s["mode"] == []             # no value repeats -> no mode
    assert s["methods"]["small_sample_warning"] is True and "sample (n-1)" in s["methods"]["std_variance"]
    assert describe_series([1, 2, 2, 3, 3, 3, 9, 9, 9, 9])["mode"] == [9.0]


def test_trend_has_steps_totals_and_method():
    t = trend_report([1000, 1200, 1500, 1800, 2100], ["Jan", "Feb", "Mar", "Apr", "May"])
    assert [s["pct"] for s in t["steps"]] == [20.0, 25.0, 20.0, 16.67] and t["steps"][0]["from"] == "Jan"
    assert t["slope_per_step"] == 280 and t["intercept"] == 960 and t["r_squared"] == 0.9949
    assert t["avg_step_pct"] == 20.42 and t["total_pct_change"] == 110.0 and t["linear_forecast"] == [2360, 2640, 2920]
    assert "ordinary least squares" in t["methods"]["model"] and "estimate" in t["methods"]["forecast"]
    assert "error" in trend_report([1, 2])


def test_dataset_detection_and_provenance():
    assert parse_dataset("Research LangGraph and analyse Python data independently, then write an article.") is None
    assert parse_dataset("") is None and parse_dataset("a: 1, 2") is None                 # fewer than 3 numbers
    ds = locate_dataset([("prose request", "analyse some data"), ("demo constant", DEMO)])
    assert ds["origin"] == "demo constant" and ds["format"] == "label: values" and len(ds["sha256_12"]) == 12
    assert ds["kinds"] == {"Month": "text", "PyPI downloads": "numeric"}
    csv = parse_dataset("month,downloads\nJan,1000\nFeb,1200\nMar,\nApr,1800")
    assert csv["columns"]["downloads"] == [1000.0, 1200.0, None, 1800.0]                  # missing value kept as None
    assert parse_dataset('[{"a": 1}, {"a": 2}, {"a": 3}]')["format"] == "json"


def test_data_quality_checks():
    ds = parse_dataset("m,v\nJan,1\nFeb,2\nFeb,2\nMar,\nApr,100\nMay,3\nJun,2")
    q = data_quality(ds)
    assert q["rows"] == 7 and q["missing"]["v"] == 1 and q["duplicate_rows"] == 1 and q["outliers"]["v"] == [100.0]


def test_report_is_labelled_and_template_uses_only_report_numbers():
    ds = locate_dataset([("demo constant", DEMO)])
    report, metrics = build_data_report(ds)
    assert report.startswith("DATASET PROVENANCE: demo constant; input hash ")
    for needle in ("Standard deviation (sample, n-1): 443.85 [population, n: 396.99]", "no mode (all values unique)",
                   "sample-adjusted Fisher-Pearson G1", "too few points", "Jan->Feb: +200 (20%)", "estimate not a fact",
                   "missing values: Month=0, PyPI downloads=0", "linear-interpolation quartiles"):
        assert needle in report, needle
    text = template_interpretation(metrics)
    assert find_unsupported_numbers(report, text) == [] and "110%" in text
    assert fmt(1520.0) == "1,520" and fmt(443.8468) == "443.85" and fmt(0.99488, 4) == "0.9949" and fmt(None) == "n/a"
