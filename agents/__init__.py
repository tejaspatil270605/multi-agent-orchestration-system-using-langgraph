"""Shared helpers used by all specialist agents."""
import os
import re
import json
import time
import threading
from typing import List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import audit

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# the fact checker can use a stronger model, e.g. GROQ_CHECKER_MODEL=openai/gpt-oss-120b
CHECKER_MODEL = os.getenv("GROQ_CHECKER_MODEL", MODEL_NAME)
RECURSION_LIMIT = 25          # hard stop for every agent's think -> tool -> think loop

# ---- rate-limit control (free-tier friendly) ----
LLM_CONCURRENCY = max(1, int(os.getenv("LLM_CONCURRENCY", "1")))     # LLM calls in flight at once
TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))                  # tokens per minute (0 = no guard)
MIN_GAP_SECONDS = float(os.getenv("LLM_MIN_GAP_SECONDS", "0.5"))      # pause between two LLM calls
MAX_GATE_WAIT = 30.0                                                  # never wait longer than this for capacity
MAX_TOTAL_WAIT = float(os.getenv("LLM_MAX_TOTAL_WAIT", "90"))         # total backoff budget per call


def make_llm(max_tokens=None, temperature=0, model=None, reasoning_effort=None):
    """reasoning_effort ("low"/"medium"/"high") only applies to gpt-oss models: a reasoning model spends part of
    max_tokens on thinking, and with too little budget the visible answer comes back EMPTY."""
    name = model or MODEL_NAME
    kwargs = {"model": name, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort and "gpt-oss" in name:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return ChatGroq(**kwargs)
    except Exception:                                   # older client without that parameter
        kwargs.pop("reasoning_effort", None)
        return ChatGroq(**kwargs)


# --------------------------------------------------------------------------------------
# error classification: transient errors are retried, permanent ones are not
# --------------------------------------------------------------------------------------
def classify_error(error) -> dict:
    """{category, transient, max_retries, message}. Used for retries (router) and the audit log."""
    text = str(error)
    low = text.lower()
    if any(k in low for k in ("rate limit", "429", "413", "tokens per minute", "request too large")):
        category, transient, retries = "rate_limit", True, 2
    elif any(k in low for k in ("timed out", "timeout", "connection", "temporarily unavailable")):
        category, transient, retries = "timeout", True, 2
    elif any(k in low for k in ("failed to validate json", "failed to generate json", "tool_use_failed",
                                "json_validate_failed", "failed to parse tool call", "empty response")):
        category, transient, retries = "json_format", True, 2
    elif any(k in low for k in ("502", "503", "504", "overloaded", "server error", "bad gateway")):
        category, transient, retries = "server_error", True, 2
    elif any(k in low for k in ("401", "403 client error: forbidden for url: https://api", "invalid api key",
                                "authentication", "incorrect api key")):
        category, transient, retries = "auth", False, 0
    elif "api_key" in low and any(k in low for k in ("not set", "missing", "required")):
        category, transient, retries = "missing_key", False, 0
    elif "no search results" in low or "no usable evidence" in low:
        category, transient, retries = "no_results", False, 0
    else:
        category, transient, retries = "unknown", True, 1           # one retry for unknown errors, then stop
    return {"category": category, "transient": transient, "max_retries": retries, "message": text[:240]}


def failure_record(error) -> dict:
    return classify_error(error)


def make_validation(checks: dict, notes: Optional[List[str]] = None) -> dict:
    """Quality of an output - kept separate from whether the task executed.
    status: passed | review_required | not_run   (failed is reserved for unusable output)
    A check that did not run (e.g. the fact checker was down) can never count as passed."""
    values = set(checks.values())
    if not checks or values == {"not_run"}:
        status = "not_run"
    elif values == {"passed"}:
        status = "passed"
    else:
        status = "review_required"
    return {"status": status, "checks": dict(checks), "notes": list(notes or [])}


# --------------------------------------------------------------------------------------
# rate-limit guard
# --------------------------------------------------------------------------------------
# counters for the end-of-run summary (how many LLM calls, how long we waited and why)
STATS = {"llm_calls": 0, "llm_errors": 0, "tokens": 0, "gate_wait_s": 0.0, "backoff_wait_s": 0.0}


def describe_reply(response) -> str:
    """finish_reason and token usage of a model reply - the first thing to look at when a reply is empty."""
    meta = getattr(response, "response_metadata", None) or {}
    usage = getattr(response, "usage_metadata", None) or {}
    parts = []
    if meta.get("finish_reason"):
        parts.append(f"finish_reason={meta['finish_reason']}")
    if usage.get("output_tokens") is not None:
        parts.append(f"output_tokens={usage['output_tokens']}")
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    if reasoning:
        parts.append(f"reasoning_tokens={reasoning}")
    return ", ".join(parts) or "no metadata"


class TokenWindow:
    """Sliding 60 s window of (estimated) tokens sent to the model."""

    def __init__(self, limit: int, window: float = 60.0, safety: float = 0.85, clock=time.monotonic):
        self.limit, self.window, self.safety, self.clock = limit, window, safety, clock
        self.events = []
        self._lock = threading.Lock()

    def wait_needed(self, tokens: int) -> float:
        if self.limit <= 0:
            return 0.0
        with self._lock:
            now = self.clock()
            self.events = [(t, k) for t, k in self.events if now - t < self.window]
            budget = self.limit * self.safety
            used = sum(k for _, k in self.events)
            if used + tokens <= budget or not self.events:
                return 0.0
            need, freed = used + tokens - budget, 0
            for t, k in sorted(self.events):
                freed += k
                if freed >= need:
                    return max(0.0, t + self.window - now)
            return max(0.0, self.events[-1][0] + self.window - now)

    def record(self, tokens: int):
        with self._lock:
            self.events.append((self.clock(), tokens))


def estimate_tokens(payload, expected_output: int = 800) -> int:
    if isinstance(payload, list):
        text = " ".join(str(m.get("content") if isinstance(m, dict) else getattr(m, "content", m)) for m in payload)
    else:
        text = str(payload)
    return int(len(text) / 3.5) + expected_output


_semaphore = threading.BoundedSemaphore(LLM_CONCURRENCY)
_window = TokenWindow(TPM_LIMIT)
_last_call = [0.0]


def retry_delay(error, attempt: int) -> float:
    """Seconds to wait before the next try. Groq rate-limit errors say how long to wait
    ("Please try again in 7.5s"); token-per-minute errors (413/429) need the window to reset."""
    text = str(error)
    match = re.search(r"try again in\s+((?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", text)
    if match and match.group(1):
        total = 0.0
        for amount, unit in re.findall(r"([\d.]+)(ms|m|s)", match.group(1)):
            total += float(amount) * {"ms": 0.001, "s": 1, "m": 60}[unit]
        return min(total + 1, 65)
    lowered = text.lower()
    if "413" in text or "429" in text or "rate limit" in lowered or "tokens per minute" in lowered:
        return min(15 * attempt, 45)
    return min(2 * attempt, 5)


def invoke_with_retry(runnable, payload, tries: int = 3, label: str = "LLM call", expected_output: int = 600):
    """One LLM call with: a shared rate-limit guard (one call at a time, token window),
    retries for TRANSIENT errors only (rate limit, timeout, malformed JSON ...), a bounded total
    backoff, and an audit-log line for every failed attempt."""
    last_error, total_wait = None, 0.0
    tokens = estimate_tokens(payload, expected_output)
    for attempt in range(1, tries + 1):
        try:
            wait = min(_window.wait_needed(tokens), MAX_GATE_WAIT)
            if wait > 0:
                print(f"  rate-limit guard: waiting {wait:.0f}s before '{label}'")
                STATS["gate_wait_s"] += wait
                time.sleep(wait)
            with _semaphore:
                gap = MIN_GAP_SECONDS - (time.monotonic() - _last_call[0])
                if gap > 0:
                    time.sleep(gap)
                STATS["llm_calls"] += 1
                try:
                    result = runnable.invoke(payload)
                finally:
                    _last_call[0] = time.monotonic()
            if result is None:
                raise ValueError("empty response")
            usage = getattr(result, "usage_metadata", None)
            used = usage.get("total_tokens") if isinstance(usage, dict) and usage.get("total_tokens") else tokens
            _window.record(used)                           # real usage when the API reports it, else the estimate
            STATS["tokens"] += used
            return result
        except Exception as error:
            last_error = error
            STATS["llm_errors"] += 1
            info = classify_error(error)
            audit.log_event("llm_error", label=label, attempt=attempt, tries=tries,
                            category=info["category"], transient=info["transient"], message=info["message"])
            print(f"{label} failed (attempt {attempt}/{tries}) [{info['category']}]: {str(error)[:160]}")
            if attempt >= tries or not info["transient"]:
                break
            wait = retry_delay(error, attempt)
            if total_wait + wait > MAX_TOTAL_WAIT:
                print(f"  backoff budget of {MAX_TOTAL_WAIT:.0f}s exhausted - giving up")
                break
            print(f"  waiting {wait:.0f}s before retrying")
            time.sleep(wait)
            total_wait += wait
            STATS["backoff_wait_s"] += wait
    raise last_error


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------
def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


def clip(text, limit: int = 8000) -> str:
    text = as_text(text)
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


_claim_checker = None
_claim_checker_big = None


def parse_check_verdict(text) -> Optional[List[str]]:
    """Plain-text verdict of the fact checker.
    'ALL SUPPORTED' -> []   |   lines 'PROBLEM: ...' -> the list   |   anything else -> None (unreadable)"""
    text = (text or "").strip()
    problems = [line.strip().split(":", 1)[1].strip()
                for line in text.splitlines()
                if line.strip().upper().startswith("PROBLEM:") and ":" in line]
    problems = [p for p in problems if p]
    if problems:
        return problems
    if "ALL SUPPORTED" in text.upper() or re.search(
            r"\ball\s+(statements\s+|claims\s+)?(are\s+)?supported\b|\bno\s+(unsupported|problems?|issues?)\b|"
            r"\bnothing\s+(is\s+)?unsupported\b", text, re.I):
        return []
    return None


def find_unsupported_claims(text: str, evidence: str, max_claims: int = 6) -> Optional[List[str]]:
    """Second-opinion fact check by a separate LLM call (optionally a stronger model).
    Returns the list of unsupported statements ([] = all supported) or None when the check
    could NOT be performed - callers must then record 'not_run', never pretend it passed.
    The answer is plain text (no JSON). A reasoning model can burn its whole budget on thinking and return an
    EMPTY answer: the first try uses low reasoning effort, the second a larger token budget, and the reply's
    finish_reason / token usage is logged so the cause is visible."""
    global _claim_checker, _claim_checker_big
    try:
        if _claim_checker is None:
            _claim_checker = make_llm(max_tokens=2500, model=CHECKER_MODEL, reasoning_effort="low")
        prompt = f"""You are a strict fact checker.

EVIDENCE (the only allowed source of facts; each source has a type):
{clip(evidence, 6000)}

TEXT TO CHECK:
{clip(text, 5000)}

Find every statement in TEXT that the EVIDENCE does not clearly support:
- details, features or components that appear nowhere in the evidence
- a cause-and-effect or "X brought Y" link between facts that the evidence does not connect
- something that only ONE source of type "blog post", "code repository" or "video" says about
  one project or tool, but TEXT states as a general property of the technology (it must be
  attributed, e.g. "one example project does ...")
- speculation written as fact ("could", "would allow", "can be used to ...") about uses the
  evidence does not mention, unless the passage is clearly labelled "Illustration:"
- scope confusion: describing a framework as if it schedules jobs, scales or is production-ready
  when the evidence only says that about a platform or an external scheduler
- wrong attribution (for example calling a blog "official", or code "from the docs" when the
  evidence shows another origin)
Ignore headings, ordinary wording and the numbers of the data table.

Answer in PLAIN TEXT only (no JSON), briefly. If everything is supported, answer exactly: ALL SUPPORTED
Otherwise write one line per problem, each starting with "PROBLEM:" followed by a short quote
(max 25 words) of the offending statement."""
        for attempt in (1, 2):
            runner = _claim_checker
            if attempt == 2:                                  # empty/unreadable first reply -> bigger budget
                if _claim_checker_big is None:
                    _claim_checker_big = make_llm(max_tokens=6000, model=CHECKER_MODEL, reasoning_effort="low")
                runner = _claim_checker_big
            response = invoke_with_retry(runner, [{"role": "user", "content": prompt}], tries=2,
                                         label="claim check", expected_output=300)
            raw = as_text(getattr(response, "content", response))
            verdict = parse_check_verdict(raw)
            if verdict is not None:
                return verdict[:max_claims]
            detail = describe_reply(response)
            print(f"Claim check gave an unreadable answer (attempt {attempt}/2) [{detail}]: {raw[:150]!r}")
            audit.log_event("claim_check_unreadable", attempt=attempt, detail=detail, raw=raw[:200])
        return None
    except Exception as error:
        print(f"Claim check could not run: {str(error)[:160]}")
        return None


def task_context(state) -> str:
    """The block every agent shows its LLM: request, task, inputs, dependency results."""
    task = state['task']
    dependency_results = state.get('result') or {}
    dependencies = "\n".join(
        f"[Task {index} result]\n{clip(value, 4500)}"
        for index, value in dependency_results.items()
    ) or "None"
    return f"""User request:
{state['user_input']}

Your assigned task:
{task.description}

Required inputs:
{task.required_input}

Expected output:
{task.expected_output}

Complexity:
{task.complexity}

Results from previous dependency tasks:
{dependencies}

Additional input provided by the user:
{state.get('additional_input')}"""


STATUS_RULES = """STATUS RULES:
1. If all required information is available and you successfully complete the
   assigned task -> status = "completed".
2. If required information from the user is missing and the task cannot be
   completed without it -> status = "waiting_for_input". Say clearly what is
   missing in `output` and why in `reason`. Never invent or assume it.
3. If you tried but hit a real failure (tool error you could not work around,
   impossible request) -> status = "failed" and explain in `reason`.
4. Producing a message or an explanation is NOT the same as completing the task.
5. Perform ONLY the assigned task; leave other agents' work to them.
6. Use dependency results as source material; never replace missing
   dependency results with invented information."""
