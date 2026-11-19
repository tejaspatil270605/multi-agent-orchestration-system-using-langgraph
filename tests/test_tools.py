"""Offline tool tests: python -m pytest tests/test_tools.py  (no API keys needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.data_analysis_tools import (analyse_numbers, parse_table, describe_csv,
                                       trend_analysis, correlation, calculate)
from tools.writing_tools import (format_artical, word_count, readability_score,
                                 verify_numbers, build_references)
from tools.coding_tools import check_python_syntax, check_imports, run_python_code
from tools.research_tools import (fetch_webpage, get_current_datetime, evidence_from_hits,
                                  clean_snippet, source_tier, topic_terms,
                                  is_relevant, best_excerpt, source_type)
from tools.writing_tools import (renumber_citations, available_source_ids, cited_source_ids,
                                 max_table_rows)


def test_analyse_numbers():
    r = analyse_numbers.invoke({"numbers": [1000, 1200, 1500, 1800, 2100]})
    assert r["mean"] == 1520 and r["median"] == 1500 and r["min"] == 1000 and r["n"] == 5
    assert analyse_numbers.invoke({"numbers": [1, 2, 3, 100]})["outliers"] == [100.0]
    assert "error" in analyse_numbers.invoke({"numbers": []})


def test_parse_table_label_format():
    text = """ Python usage dataset:
      Month: Jan, Feb, Mar, Apr, May
      PyPI downloads: 1000, 1200, 1500, 1800, 2100"""
    r = parse_table.invoke({"text": text})
    assert r["format"] == "label: values"
    assert r["columns"]["PyPI downloads"] == [1000, 1200, 1500, 1800, 2100]
    assert r["columns"]["Month"][0] == "Jan" and r["equal_lengths"]
    assert r["numeric_columns"] == ["PyPI downloads"]


def test_parse_table_csv_and_json():
    assert parse_table.invoke({"text": "a,b\n1,2\n3,4\n5,6"})["columns"]["b"] == [2, 4, 6]
    assert parse_table.invoke({"text": '{"x": [1,2,3]}'})["format"] == "json"
    assert "error" in parse_table.invoke({"text": ""})


def test_trend_analysis():
    r = trend_analysis.invoke({"values": [1000, 1200, 1500, 1800, 2100], "periods_ahead": 2})
    assert r["direction"] == "increasing" and r["r_squared"] > 0.98
    assert r["total_pct_change"] == 110.0 and len(r["linear_forecast"]) == 2
    assert "error" in trend_analysis.invoke({"values": [1, 2]})


def test_correlation():
    r = correlation.invoke({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8.5]})
    assert r["strength"] == "very strong" and r["direction"] == "positive"
    assert "error" in correlation.invoke({"x": [1, 2, 3], "y": [1, 2]})
    assert "error" in correlation.invoke({"x": [1, 1, 1], "y": [1, 2, 3]})


def test_calculate_safe():
    assert calculate.invoke({"expression": "(2100-1000)/1000*100"})["result"] == 110.0
    assert calculate.invoke({"expression": "sqrt(144)+log10(1000)"})["result"] == 15.0
    for bad in ["__import__('os').system('echo hi')", "open('x')", "9**9999", "1/0"]:
        assert "error" in calculate.invoke({"expression": bad}), bad


def test_describe_csv():
    r = describe_csv.invoke({"csv_text": "a,b\n1,x\n2,y\n3,"})
    assert r["rows"] == 3 and r["missing_values"]["b"] == 1


def test_writing_tools():
    out = format_artical.invoke({"title": "T", "content": "intro\n## H\nbody\n\n\n\nmore"})
    assert out.startswith("# T\n\nintro") and "## H\n\nbody" in out and "\n\n\n" not in out
    assert format_artical.invoke({"title": "T", "content": "# Mine\ntext"}).startswith("# Mine")
    assert word_count.invoke({"text": "One two three. Four five."})["words"] == 5
    assert readability_score.invoke({"text": "The cat sat on the mat. It was happy."})["flesch_reading_ease"] > 80
    refs = build_references.invoke({"sources": ["https://a.com", "https://a.com", "https://b.com"]})
    assert refs.count("https://a.com") == 1 and "2. https://b.com" in refs


def test_verify_numbers():
    src = "Downloads grew from 1,000 to 2100 (110% growth)."
    assert verify_numbers.invoke({"source_text": src, "draft_text": "It rose 1000 to 2,100, up 110%."})["ok"]
    bad = verify_numbers.invoke({"source_text": src, "draft_text": "Downloads hit 9999 and 5 months passed."})
    assert bad["unsupported_numbers"] == ["9999"]        # small ints / years ignored


def test_coding_tools():
    assert check_python_syntax.invoke({"code": "print(1)"})["ok"]
    assert check_python_syntax.invoke({"code": "def f(:\n  pass"})["ok"] is False
    imp = check_imports.invoke({"code": "import os, json\nimport notarealpkg123"})
    assert "os" in imp["stdlib"] and imp["missing"] == ["notarealpkg123"]
    ok = run_python_code.invoke({"code": "print(sum(range(5)))"})
    assert ok["ok"] and ok["stdout"].strip() == "10"
    assert run_python_code.invoke({"code": "raise ValueError('x')"})["ok"] is False
    assert "timed out" in run_python_code.invoke({"code": "while True: pass", "timeout_seconds": 1})["error"]
    for evil in ["import subprocess", "import os\nos.system('echo hi')", "eval('1')", "from shutil import rmtree"]:
        assert "rejected" in run_python_code.invoke({"code": evil})["error"], evil


def test_code_execution_switch():
    os.environ["ENABLE_CODE_EXECUTION"] = "0"
    try:
        assert "disabled" in run_python_code.invoke({"code": "print(1)"})["error"]
    finally:
        os.environ["ENABLE_CODE_EXECUTION"] = "1"


def test_fetch_webpage_blocks_private():
    for url in ["http://localhost/x", "http://127.0.0.1/", "http://192.168.1.5/", "file:///etc/passwd", "ftp://x.com"]:
        assert "rejected" in fetch_webpage.invoke({"url": url})[0]["error"], url
    assert len(get_current_datetime.invoke({})) >= 19


GITHUB_JUNK = """/

# README.md

Copy path

## File metadata and controls

28 lines (24 loc) · 857 Bytes

Raw

Copy raw file

Download raw file

Outline

Edit and raw actions

# Example of langgraph integration

## Feature

1. Using langgraph stream writer and custom mode of stream to integrate with the pipeline."""


def test_clean_snippet_removes_page_junk():
    out = clean_snippet(GITHUB_JUNK, 300)
    assert "Copy path" not in out and "Download raw file" not in out and "857 Bytes" not in out
    assert "stream writer" in out
    long = "First sentence is here and it is long enough. " * 20
    cut = clean_snippet(long, 100)
    assert len(cut) <= 101 and cut.endswith(".")
    assert clean_snippet(None) == "" and clean_snippet("Raw") == ""


def test_source_tier_prefers_official():
    assert source_tier("https://docs.langchain.com/oss/python/langgraph/overview") == 0
    assert source_tier("https://langchain-ai.github.io/langgraph/") == 0
    assert source_tier("https://en.wikipedia.org/wiki/X") == 0
    assert source_tier("https://blog.gopenai.com/some-post") == 1
    assert source_tier("https://github.com/open-webui/pipelines/blob/main/README.md") == 1
    assert source_tier("https://www.youtube.com/watch?v=abc") == 2
    assert source_tier("https://notyoutube.com/x") == 1


def test_evidence_fallback_from_raw_hits():
    good = "LangGraph is a framework for building stateful agents as graphs of nodes and edges."
    results = [
        {"query": "q", "source": "tavily", "result": [
            {"title": "Video", "url": "https://www.youtube.com/watch?v=1", "content": good, "score": 0.99},
            {"title": "Blog", "url": "https://blog.example.com/a", "content": good, "score": 0.5},
            {"title": "Blog dup", "url": "https://blog.example.com/a", "content": good, "score": 0.4},
            {"title": "Junk only", "url": "https://junk.example.com", "content": "Raw\n\nCopy raw file", "score": 0.9},
            {"title": "Docs", "url": "https://docs.langchain.com/x", "content": good, "score": 0.1},
            {"title": "Scribd", "url": "https://www.scribd.com/document/1", "content": good, "score": 0.99}]},
        {"query": "q", "source": "wikipedia", "result": [{"title": "W", "url": "https://en.wikipedia.org/?curid=1", "content": good}]},
        {"query": "q", "source": "arxiv", "result": []},
    ]
    items = evidence_from_hits(results)
    urls = [i["url"] for i in items]
    assert set(urls[:2]) == {"https://docs.langchain.com/x", "https://en.wikipedia.org/?curid=1"}   # official first
    assert "https://www.scribd.com/document/1" not in urls                                          # excluded site
    assert urls[-1] == "https://www.youtube.com/watch?v=1"                                       # video last
    assert "https://junk.example.com" not in urls and urls.count("https://blog.example.com/a") == 1
    assert evidence_from_hits([{"result": []}]) == [] and evidence_from_hits(None) == []


SOURCES = """Summary text [S1-1].

SOURCES:
[S1-1] LangGraph docs - https://docs.example.com/lg
    > LangGraph builds stateful agents.
[S1-2] Blog about graphs - https://blog.example.com/g
    > Graphs have nodes and edges.
[S1-3] Unused source - https://unused.example.com
    > Not cited anywhere."""


def test_citations_renumbered_and_only_cited_sources_listed():
    assert available_source_ids(SOURCES) == {"S1-1", "S1-2", "S1-3"}
    draft = "Intro uses graphs [S1-2]. Agents are stateful [S1-1, S1-2]. Fake [S9-9]. Again [S1-2]."
    assert cited_source_ids(draft) == ["S1-2", "S1-1", "S1-2", "S9-9", "S1-2"]
    text, refs = renumber_citations(draft, SOURCES)
    assert "[1]" in text and "[1, 2]" in text and "S1-" not in text and "S9-9" not in text
    assert refs.splitlines() == ["[1] Blog about graphs - https://blog.example.com/g",
                                 "[2] LangGraph docs - https://docs.example.com/lg"]
    assert "unused.example.com" not in refs


LANGGRAPH_QUERIES = [
    "LangGraph official documentation overview", "LangGraph core concepts and architecture",
    "LangGraph use cases in Python applications", "Recent developments and releases of LangGraph 2026",
    "LangGraph integration with LangChain and LLMs", "LangGraph tutorials for beginners",
    "LangGraph community contributions and GitHub repository"]


def test_topic_terms_and_relevance_filter():
    assert topic_terms(LANGGRAPH_QUERIES) == ["langgraph"]
    assert topic_terms([]) == [] and topic_terms(["python data pipelines", "how to use ai"]) == []
    terms = ["langgraph"]
    assert is_relevant({"title": "LangGraph overview", "url": "https://x.com", "content": ""}, terms)
    assert is_relevant({"title": "t", "url": "https://x.com/langgraph", "content": ""}, terms)
    assert not is_relevant({"title": "LLVM", "url": "https://en.wikipedia.org/?curid=1",
                            "content": "LLVM is a set of compiler technologies."}, terms)
    assert is_relevant({"title": "anything", "url": "u", "content": "c"}, [])      # no topic -> keep all


def test_clean_snippet_keeps_product_names_and_drops_ads():
    page = """While LangGraph can be used standalone, it also integrates with any LangChain product. To improve your application, pair LangGraph with:

LangSmith

Trace requests, evaluate outputs, and monitor deployments in one place.

Interrupt is coming to London on October 13! Get tickets

Fetch the complete documentation index at: </llms.txt Use this file to discover all available pages."""
    out = clean_snippet(page, 600)
    assert "LangSmith" in out and "pair LangGraph with:" in out and "Trace requests" in out
    assert "Get tickets" not in out and "London" not in out and "llms.txt" not in out


def test_best_excerpt_prefers_topic_sentences():
    text = ("Cookie settings and navigation are shown at the top of every page for visitors. "
            "LangGraph models agent workflows as graphs made of state, nodes and edges. "
            "Our newsletter is sent every week with many unrelated updates for subscribers. "
            "LangGraph supports durable execution and human-in-the-loop review of agent steps.")
    out = best_excerpt(text, ["langgraph"], 400)
    assert out.count("LangGraph") == 2 and "Cookie" not in out and "newsletter" not in out
    assert best_excerpt("", ["x"]) == "" and best_excerpt("short.", ["x"]) == ""


def test_evidence_prefers_deep_read_excerpt():
    base = "Short generic search snippet that says very little about the real subject at all."
    page = "LangGraph models agent workflows as graphs made of state, nodes and edges for stateful agents."
    items = evidence_from_hits([{"source": "tavily", "result": [
        {"title": "Docs", "url": "https://docs.example.com/a", "content": base, "page_excerpt": page}]}])
    assert items[0]["snippets"] == page


def test_source_type_labels():
    assert source_type("https://docs.langchain.com/oss/python/langgraph/overview") == "official documentation"
    assert source_type("https://docs.aws.amazon.com/x/langchain-langgraph.html", ["langgraph"]).startswith("third-party")
    assert source_type("https://www.ibm.com/docs/en/watsonx/orchestrate", ["langgraph"]).startswith("third-party")
    assert source_type("https://www.scribd.com/document/1/x").startswith("low-quality")
    assert source_type("https://arxiv.org/abs/2412.03801") == "research paper"
    assert source_type("https://en.wikipedia.org/wiki/X") == "encyclopedia"
    assert source_type("https://github.com/ProactiveAIAgents/LangGraph_Chatbot_Agent").startswith("code repository")
    assert source_type("https://medium.com/@x/y").startswith("blog post")
    assert source_type("https://www.truefoundry.com/blog/langchain-vs-langgraph").startswith("blog post")
    assert source_type("https://www.youtube.com/watch?v=1").startswith("video")
    assert source_type("https://www.coursera.org/articles/x") == "web article"


def test_clean_snippet_drops_broken_fragments():
    s1 = clean_snippet("Morgan, and more\u2014LangGraph is a low-level orchestration framework for stateful agents. It gives control.", 300)
    assert s1.startswith("LangGraph is a low-level") and "Morgan" not in s1
    s2 = clean_snippet("and more of the page text continues here without any start. LangGraph models workflows as graphs of nodes.", 300)
    assert s2 == "LangGraph models workflows as graphs of nodes."
    assert clean_snippet("Hello there. This is fine and complete sentence number two.").startswith("Hello there.")


def test_max_table_rows():
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | x |" for i in range(9))
    assert max_table_rows(table) == 9
    two = "| h |\n|--|\n| 1 |\n\ntext\n\n| h |\n|:-:|\n| 1 |\n| 2 |\n| 3 |"
    assert max_table_rows(two) == 3 and max_table_rows("no table") == 0


def test_evidence_ranks_snippets_that_mention_the_topic_first():
    on = "LangGraph is a framework for building stateful agents as graphs of nodes and edges."
    off = "With the rapid development of large model technology, agent technology is becoming widespread."
    items = evidence_from_hits([{"source": "arxiv", "result": [
        {"title": "LangGraph + CrewAI", "url": "https://arxiv.org/abs/1", "content": off},
        {"title": "LangGraph paper", "url": "https://arxiv.org/abs/2", "content": on}]}], terms=["langgraph"])
    assert [i["url"] for i in items] == ["https://arxiv.org/abs/2", "https://arxiv.org/abs/1"]


def test_mojibake_and_menu_junk():
    from tools.research_tools import fix_mojibake
    assert fix_mojibake("Updated on Sep 25, 2026 \u00e2\u0080\u00a2 Explore") == "Updated on Sep 25, 2026 \u2022 Explore"
    assert fix_mojibake("plain text stays") == "plain text stays"
    menu = ("For Individuals For Businesses For Universities For Governments Explore Degrees Log In Join for Free "
            "Written by Coursera Staff. LangGraph builds stateful agents as graphs of nodes and edges.")
    out = clean_snippet(menu, 300)
    assert out == "LangGraph builds stateful agents as graphs of nodes and edges." and "Log In" not in out
    assert clean_snippet("Skip to main content Open navigation menu 0 ratings 0% found this document useful (0 votes) 16 views") == ""


def test_number_formats_are_normalised():
    from tools.writing_tools import find_unsupported_numbers
    source = "Mean: 1,520; Sum: 7\u202f600; Variance 197 000; forecast 2\u00a0360, 2\u00a0640"
    draft = "Mean 1520, total 7,600, variance 197000, forecast 2 360 and 2,640."
    assert find_unsupported_numbers(source, draft) == []
    assert find_unsupported_numbers(source, draft + " Growth was 110%.") == ["110"]


def test_repeated_tables_and_hypothetical():
    from tools.writing_tools import repeated_table_labels, mentions_hypothetical
    t = ("| m | v |\n|--|--|\n| Mean | 1 |\n| Median | 2 |\n| Min | 3 |\n\ntext\n\n"
         "| Metric | Value |\n|--|--|\n| Mean | 1 |\n| Median | 2 |\n| Min | 3 |\n| Max | 4 |")
    assert repeated_table_labels(t) == ["mean", "median", "min"]
    assert repeated_table_labels("| a |\n|--|\n| x |\n\n| b |\n|--|\n| y |") == []
    assert mentions_hypothetical("for a hypothetical package") and not mentions_hypothetical("the supplied dataset")


def test_official_repository_label_and_off_topic_snippets_are_dropped():
    assert source_type("https://github.com/langchain-ai/langgraph-example", ["langgraph"]) == "official code repository"
    assert source_type("https://github.com/someone/awesome-langgraph", ["langgraph"]).startswith("code repository (ONE")
    on = "LangGraph is a framework for building stateful agents as graphs of nodes and edges."
    off = "With the rapid development of large model technology, agent technology is becoming widespread."
    hits = [{"source": "arxiv", "result": [
        {"title": "LangGraph paper A", "url": "https://arxiv.org/abs/1", "content": on},
        {"title": "LangGraph paper B", "url": "https://arxiv.org/abs/2", "content": on + " It supports cycles."},
        {"title": "LangGraph paper C", "url": "https://arxiv.org/abs/3", "content": on + " It supports persistence."},
        {"title": "LangGraph + CrewAI", "url": "https://arxiv.org/abs/4", "content": off}]}]
    urls = [i["url"] for i in evidence_from_hits(hits, terms=["langgraph"])]
    assert "https://arxiv.org/abs/4" not in urls and len(urls) == 3          # the generic abstract is noise
    few = evidence_from_hits(hits[:1][0:1] and [{"source": "arxiv", "result": hits[0]["result"][:1] + hits[0]["result"][3:]}], terms=["langgraph"])
    assert len(few) == 2                                                      # too few relevant ones: nothing is dropped
