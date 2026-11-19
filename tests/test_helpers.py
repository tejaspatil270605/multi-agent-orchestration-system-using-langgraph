"""Retry-delay logic for Groq errors: python -m pytest tests/test_helpers.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents
from agents import (retry_delay, parse_check_verdict, classify_error, make_validation, TokenWindow,
                    estimate_tokens, invoke_with_retry, describe_reply)


def test_uses_wait_time_from_groq_message():
    assert retry_delay(Exception("Rate limit reached. Please try again in 7.5s."), 1) == 8.5
    assert retry_delay(Exception("Please try again in 450ms"), 1) == 1.45
    assert retry_delay(Exception("Please try again in 1m5.2s"), 1) == 65          # capped
    assert retry_delay(Exception("Please try again in 2m3.5s. Need more tokens"), 1) == 65


def test_token_limit_errors_wait_for_the_window():
    err = Exception("Error code: 413 - Request too large ... tokens per minute (TPM)")
    assert retry_delay(err, 1) == 15 and retry_delay(err, 2) == 30 and retry_delay(err, 5) == 45
    assert retry_delay(Exception("Error code: 429 rate limit"), 1) == 15


def test_other_errors_retry_quickly():
    assert retry_delay(Exception("Failed to parse tool call arguments as JSON"), 1) == 2
    assert retry_delay(Exception("boom"), 4) == 5


def test_parse_check_verdict():
    assert parse_check_verdict("ALL SUPPORTED") == [] and parse_check_verdict("all supported.") == []
    assert parse_check_verdict("PROBLEM: first one\nPROBLEM: second: with colon") == ["first one", "second: with colon"]
    assert parse_check_verdict("problem: lower case works") == ["lower case works"]
    assert parse_check_verdict("") is None and parse_check_verdict(None) is None
    assert parse_check_verdict("I think it looks fine overall") is None        # unreadable -> check not performed


def test_error_classification():
    assert classify_error(Exception("Error code: 429 - Rate limit reached")) == {"category": "rate_limit", "transient": True, "max_retries": 2, "message": "Error code: 429 - Rate limit reached"}
    assert classify_error(Exception("Error code: 400 - Failed to validate JSON"))["category"] == "json_format"
    assert classify_error(Exception("HTTPSConnectionPool Read timed out"))["category"] == "timeout"
    for permanent in ("TAVILY_API_KEY is not set", "Error code: 401 - Invalid API Key", "no search results"):
        info = classify_error(Exception(permanent))
        assert info["transient"] is False and info["max_retries"] == 0, permanent
    unknown = classify_error(Exception("something odd"))
    assert unknown["category"] == "unknown" and unknown["max_retries"] == 1


def test_validation_status_is_separate_from_execution():
    assert make_validation({"a": "passed", "b": "passed"})["status"] == "passed"
    assert make_validation({"a": "passed", "b": "not_run"})["status"] == "review_required"   # a skipped check never counts as passed
    assert make_validation({"a": "passed", "b": "failed"}, ["x"])["status"] == "review_required"
    assert make_validation({"a": "not_run"})["status"] == "not_run" and make_validation({})["status"] == "not_run"
    assert make_validation({"a": "passed"}, ["note"])["notes"] == ["note"]


def test_token_window_waits_for_capacity():
    now = [0.0]
    window = TokenWindow(1000, window=60.0, safety=1.0, clock=lambda: now[0])
    assert window.wait_needed(900) == 0.0
    window.record(900)
    now[0] = 10.0
    assert window.wait_needed(200) == 50.0            # must wait until the first call leaves the 60 s window
    assert window.wait_needed(50) == 0.0
    now[0] = 61.0
    assert window.wait_needed(900) == 0.0
    assert TokenWindow(0).wait_needed(10 ** 6) == 0.0   # guard disabled
    assert estimate_tokens("x" * 3500, expected_output=0) == 1000


class _Flaky:
    def __init__(self, errors, then="ok"):
        self.errors, self.then, self.calls = list(errors), then, 0

    def invoke(self, payload):
        self.calls += 1
        if self.errors:
            raise Exception(self.errors.pop(0))
        return self.then


def test_retry_only_transient_errors_with_bounded_wait():
    agents.time.sleep = lambda s: None
    agents._window = TokenWindow(0)                    # no guard in this test
    flaky = _Flaky(["Error code: 400 - Failed to validate JSON", "Error code: 429 rate limit"])
    assert invoke_with_retry(flaky, "p", tries=3) == "ok" and flaky.calls == 3
    permanent = _Flaky(["Error code: 401 - Invalid API Key"] * 3)
    try:
        invoke_with_retry(permanent, "p", tries=3)
        assert False, "should raise"
    except Exception as e:
        assert "401" in str(e) and permanent.calls == 1          # a permanent error is NOT retried
    agents.MAX_TOTAL_WAIT = 20
    always = _Flaky(["Error code: 429 rate limit"] * 10)
    try:
        invoke_with_retry(always, "p", tries=10)
    except Exception:
        pass
    assert always.calls <= 3                                     # 15 s + 30 s waits exceed the 20 s budget


def test_natural_language_verdicts_are_understood():
    for ok in ["No unsupported statements were found.", "All statements are supported.", "No problems found.",
               "Nothing is unsupported."]:
        assert parse_check_verdict(ok) == [], ok
    assert parse_check_verdict("PROBLEM: x") == ["x"] and parse_check_verdict("Hmm, hard to say") is None


def test_describe_reply_explains_empty_answers():
    class Reply:
        response_metadata = {"finish_reason": "length"}
        usage_metadata = {"output_tokens": 2500, "output_token_details": {"reasoning": 2480}}
    assert describe_reply(Reply()) == "finish_reason=length, output_tokens=2500, reasoning_tokens=2480"
    assert describe_reply(object()) == "no metadata"


def test_real_token_usage_is_recorded_instead_of_the_estimate():
    agents.time.sleep = lambda s: None
    agents.MIN_GAP_SECONDS = 0
    agents._window = TokenWindow(10 ** 6)

    class Reply:
        usage_metadata = {"total_tokens": 123}

    class Runner:
        def invoke(self, payload):
            return Reply()
    invoke_with_retry(Runner(), "x" * 7000)              # the estimate would be about 2,600 tokens
    assert agents._window.events[-1][1] == 123


def test_make_llm_survives_clients_without_reasoning_effort():
    calls = []

    def fake_chat_groq(**kwargs):
        calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise TypeError("unexpected keyword")
        return "llm"
    original = agents.ChatGroq
    agents.ChatGroq = fake_chat_groq
    try:
        assert agents.make_llm(100, model="openai/gpt-oss-20b", reasoning_effort="low") == "llm"
        assert "reasoning_effort" in calls[0] and "reasoning_effort" not in calls[1]
        calls.clear()
        agents.make_llm(100, model="some-other-model", reasoning_effort="low")       # only gpt-oss models get it
        assert "reasoning_effort" not in calls[0]
    finally:
        agents.ChatGroq = original
