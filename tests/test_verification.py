"""Claim-by-claim verification (pure code): python -m pytest tests/test_verification.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.verification_tools import (parse_sources, split_claims, cited_ids, verify_claims, format_issues,
                                      remove_claims, foreign_citation_tags, strip_foreign_tags, issue_key,
                                      compact_evidence, normalize_text)

SOURCES_TEXT = """Summary [S0-1].

SOURCES:
[S0-1] LangGraph overview - Docs by LangChain - https://docs.langchain.com/oss/python/langgraph/overview
    type: official documentation
    > LangGraph is a low-level orchestration framework for building stateful agents. It supports durable execution and human-in-the-loop.
[S0-2] A blog about graphs - https://blog.example.com/g
    type: blog post (one author's view)
    > LangGraph supports scheduled tasks, persistent storage and retry on failure in production.
[S0-3] IBM release notes - https://www.ibm.com/docs/x
    type: third-party / vendor documentation (NOT the project's own)
    > Checkpointers store conversation state in memory, SQLite and PostgreSQL."""


def test_parse_sources():
    s = parse_sources(SOURCES_TEXT)
    assert set(s) == {"S0-1", "S0-2", "S0-3"}
    assert s["S0-1"]["type"] == "official documentation" and "stateful agents" in s["S0-1"]["snippet"]
    assert s["S0-2"]["url"] == "https://blog.example.com/g"


def test_split_claims_moves_citation_after_period_back_to_its_sentence():
    claims = split_claims("LangGraph is a low-level orchestration framework for agents. [S0-1] It runs stateful agents [S0-1][S0-3].")
    assert cited_ids(claims[0]) == ["S0-1"] and cited_ids(claims[1]) == ["S0-1", "S0-3"]
    assert split_claims("| a | b |\n# Heading\nShort.\n```code```") == []


def test_supported_claim_passes_and_problems_are_found():
    s = parse_sources(SOURCES_TEXT)
    good = "LangGraph is a low-level orchestration framework for building stateful agents [S0-1]."
    assert verify_claims(good, s, ["langgraph"]) == []
    unknown = verify_claims("LangGraph is a framework for building stateful agents [S9-9].", s, ["langgraph"])
    assert unknown[0]["problem"] == "unknown_source"
    weak = verify_claims("LangGraph includes a visual drag and drop designer for business analysts [S0-1].", s, ["langgraph"])
    assert weak[0]["problem"] == "weak_support"
    missing = verify_claims("LangGraph gives developers fine grained control over agent workflows and state.", s, ["langgraph"])
    assert missing[0]["problem"] == "no_citation"
    assert verify_claims("Gaps: the sources do not cover recent releases of LangGraph at all.", s, ["langgraph"]) == []


def test_scope_claim_needs_documentation_not_a_blog():
    s = parse_sources(SOURCES_TEXT)
    blog_only = verify_claims("LangGraph supports scheduled tasks and persistent storage in production [S0-2].", s, ["langgraph"])
    assert [i["problem"] for i in blog_only] == ["scope"] and "external scheduler" in blog_only[0]["detail"]
    ok = verify_claims("LangGraph supports durable execution and human-in-the-loop review of stateful agents [S0-1].", s, ["langgraph"])
    assert ok == []
    lines = format_issues(blog_only)
    assert lines and lines[0].startswith("[scope]")


def test_unicode_hyphens_do_not_cause_false_weak_support():
    s = {"S0-3": {"title": "x", "url": "u", "type": "official documentation",
                  "snippet": "The main use cases are conversational agents, and long-running, multi-step LLM applications."}}
    claim = "Typical uses include long\u2011running, multi\u2011step LLM applications built on it [S0-3]."   # non-breaking hyphens
    assert verify_claims(claim, s, ["langgraph"]) == []
    assert normalize_text("a\u2011b\u202fc") == "a-b c"


def test_table_rows_with_citations_are_checked_and_labelled_blocks_are_skipped():
    s = parse_sources(SOURCES_TEXT)
    table = "| Concept | Meaning |\n|---|---|\n| Visual designer | A drag and drop designer for analysts [S0-1] |\n| Plain | no citation here |"
    issues = verify_claims(table, s, ["langgraph"])
    assert [i["problem"] for i in issues] == ["weak_support"]
    ill = "Illustration: LangGraph could orchestrate a data pipeline. It could store results. A human could review them."
    assert verify_claims(ill, s, ["langgraph"]) == [] and verify_claims("Gaps: recent releases of langgraph are not covered at all.", s, ["langgraph"]) == []


def test_unverifiable_sentences_can_be_removed():
    s = parse_sources(SOURCES_TEXT)
    text = ("Intro line that stays.\nLangGraph provides the building blocks for orchestrating complex workflows, including data processing. "
            "This sentence is fine and should stay in the text.\n"
            "- LangGraph supports scheduled tasks and persistent storage in production [S0-2]. [S0-2]\n"
            "LangGraph supports durable execution and human-in-the-loop review of stateful agents [S0-1].")
    issues = verify_claims(text, s, ["langgraph"])
    new, removed = remove_claims(text, issues)
    assert len(removed) == 2 and "building blocks" not in new and "scheduled tasks" not in new and "[S0-2]" not in new
    assert "This sentence is fine" in new and "durable execution" in new and "Intro line that stays." in new
    assert verify_claims(new, s, ["langgraph"]) == []
    assert issue_key(issues) == issue_key(list(reversed(issues))) and issue_key([]) == ""


def test_invented_citation_tags():
    assert foreign_citation_tags("see [T1], [D2], [S0-1] and [3] but not [link](x)") == ["[3]", "[D2]", "[T1]"]
    assert strip_foreign_tags("Five rows [T1]. Next.") == "Five rows. Next."
    assert foreign_citation_tags("only [S0-1] and [S12-3]") == []


def test_compact_evidence_for_the_llm_checker():
    out = compact_evidence(parse_sources(SOURCES_TEXT))
    assert out.count("\n") == 2 and out.startswith("[S0-1] (official documentation) LangGraph overview")
    assert len(out) < len(SOURCES_TEXT)
