# Multi-Agent Orchestrator (LangGraph)

Flow: `supervisor -> task_router -> (research | data | coding | writing agents) -> task_router -> ... -> END`,
with human-in-the-loop (`hitl`) and SQLite checkpointing.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and TAVILY_API_KEY
python backend.py
python -m pytest tests      # offline checks, no keys needed
```
`DEMO_MODE=1` (default) answers the "please provide data" pause with a demo dataset that is **labelled as a demo
constant**. `DEMO_MODE=0` asks you to paste your own data.

## Two separate questions for every task
| | meaning | values |
|---|---|---|
| **execution** | did the task run? | completed / failed / waiting_for_input |
| **validation** | is the result trustworthy? | passed / review_required / not_run |

A result with failed or skipped checks is **never** reported as a normal success: `review_required` prints a
banner at the end of the run. A check that could not run (for example the fact checker was down) counts as `not_run`,
not as passed. The end-of-run **audit summary** shows execution, validation, attempts, failure type and input provenance.

## Agents
| Agent | What it does |
|---|---|
| research | queries (plain text) -> search (Tavily, Wikipedia, arXiv) -> drop off-topic and excluded sites (Scribd, LinkedIn ...) -> deep-read best pages -> evidence built in code (cleaned snippets, the project's own docs first, source-type labels, numbered ids) -> plain-text summary (no tables) with a `Gaps:` line -> **every sentence checked against the sources it cites** + LLM fact check -> one rewrite. Retries reuse the cached evidence (no repeated searches). |
| data | **all numbers computed in Python**: dataset detection with provenance, data-quality checks, statistics labelled with their method (sample vs population, quartile method, skewness definition, "no mode"), OLS trend with step changes. The LLM only writes a short reading, rejected if it contains a number that is not in the report. |
| coding | write -> test -> fix with syntax/import checks and a sandboxed runner; validation = the code ran without error. |
| writing | LLM draft with citations -> code checks (numbers in any format, citations, invented tags like [T1], length, table size, repeated tables, dataset description) -> per-sentence claim check, then LLM fact check on a clean draft -> ONE merged revision. If a rewrite repeats the same unsupported sentence the loop stops and those sentences are **removed automatically** (never delivered). Sources lists only cited links. |

## Retries
Errors are classified: **transient** (rate limit, timeout, malformed JSON) are retried with bounded backoff;
**permanent** (missing/invalid API key, no search results) are not retried; unknown errors get one retry.
Validation problems are never "retried blindly": the writer gets corrective feedback for a bounded number of drafts,
then the result is flagged `review_required`.

## Settings (`.env.example`)
`GROQ_CHECKER_MODEL` (stronger fact checker), `GROQ_TPM_LIMIT` / `LLM_CONCURRENCY` / `LLM_MIN_GAP_SECONDS` /
`LLM_MAX_TOTAL_WAIT` (rate-limit control), `OFFICIAL_SOURCES_FILE` (extra official domains), `AUDIT_LOG_DIR`.

## Troubleshooting a slow or failing run
- The end-of-run summary prints LLM calls, tokens, and **seconds spent waiting for rate limits**. Lower
  `GROQ_TPM_LIMIT` only if your plan is smaller than 8,000 tokens/minute; raise it if your plan is larger.
- `claim check gave an unreadable answer [finish_reason=length ...]` means the reasoning model used its whole token
  budget on thinking. The checker already uses low reasoning effort and retries once with a bigger budget; a stronger
  `GROQ_CHECKER_MODEL` also helps.
- Token use is read from the API's real usage numbers when available (the estimate is only a fallback).

## Audit log
Every run writes `logs/run_<id>.jsonl`: one JSON line per event with agent, task, attempt, error category,
input provenance (source, hash) and validation outcome.

## Limits (be honest in your portfolio)
- The claim check is a lexical-overlap heuristic plus an LLM second opinion - not proof of truth.
- Source labels come from the URL; an official blog on a company domain may be labelled "blog post".
- Dates and facts are only as good as the sources.
- `run_python_code` is not a sandbox; set `ENABLE_CODE_EXECUTION=0` to disable it.
