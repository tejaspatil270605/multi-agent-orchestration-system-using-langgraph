import re
from datetime import date
from state import State
from schemas import AgentResult, EvidenceItem, EvidencePack
from agents import (make_llm, task_context, clip, as_text, invoke_with_retry, find_unsupported_claims,
                    make_validation, failure_record, classify_error, describe_reply)
import audit
from tools.research_tools import (tavily_search, wikipedia_search, arxiv_search, fetch_webpage,
                                  evidence_from_hits, clean_snippet, source_tier, source_type,
                                  topic_terms, is_relevant, best_excerpt)
from tools.verification_tools import (parse_sources, verify_claims, format_issues, compact_evidence,
                                      remove_claims)
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, START, END

# Everything the LLM produces here is PLAIN TEXT (queries, summary): a malformed-JSON error cannot occur.
text_llm = make_llm(max_tokens=3500)

MAX_ATTEMPTS = 2          # query -> search -> evidence rounds when nothing useful is found
DEEP_READ_PAGES = 3       # how many top web pages are downloaded in full


class Research_State(TypedDict):
    research_prompt: str
    task_description: str
    queries: List[str]
    search_result: List[dict]
    terms: List[str]
    tool_errors: List[str]
    evidence: Optional[EvidencePack]
    attempts: int


def _usable(hits, errors: List[str] = None, source: str = "") -> List[dict]:
    """Drop tool error markers (remembered for the failure classification), keep real results."""
    clean = []
    for hit in hits or []:
        if isinstance(hit, dict) and "error" in hit:
            print(f"  (tool warning: {hit['error']})")
            if errors is not None:
                errors.append(f"{source}: {hit['error']}")
        elif isinstance(hit, dict):
            clean.append(hit)
    return clean


def parse_queries(text: str) -> List[str]:
    """One query per line; bullets, numbering and quotes are removed; duplicates dropped."""
    queries = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"\'`')
        if 3 <= len(line) <= 120 and line.lower() not in [q.lower() for q in queries]:
            queries.append(line)
    return queries[:7]


def query_generator(state: Research_State):
    attempts = state.get('attempts', 0)
    prompt = (f"{state['research_prompt']}\n\nWrite 5 to 7 focused, non-redundant web search queries for this "
              f"research task, ONE PER LINE, no numbering, no explanations. At least one query must target official "
              f"documentation or primary sources (for example include the words 'official documentation').")
    if attempts > 0:
        prompt += (f"\n\nThe previous queries {state.get('queries')} returned no usable evidence. "
                   f"Write NEW queries with different wording.")
    queries = []
    for _ in range(2):
        response = invoke_with_retry(text_llm, [{'role': 'user', 'content': prompt}], label="query generation",
                                     expected_output=200)
        queries = parse_queries(as_text(response.content))
        if len(queries) >= 3:
            break
        print(f"  query list unreadable: {as_text(response.content)[:120]!r}")
    if len(queries) < 3:                                   # deterministic fallback, never crash on this
        base = state.get('task_description', '')[:100]
        queries = [base, f"{base} official documentation", f"{base} overview", f"{base} use cases"]
        print("  using fallback queries built from the task description")
    for i, query in enumerate(queries):
        print(f"{i+1}. {query}")
    return {'queries': queries, 'attempts': attempts + 1}


def search_queries(state: Research_State):
    queries = state['queries']
    terms = topic_terms(queries)
    print(f"Topic terms used to filter irrelevant results: {terms or 'none found'}")
    errors, all_result = [], []
    for query in queries:
        print(f"Searching Query {query}")
        result = _usable(tavily_search.invoke(query), errors, "tavily")
        for hit in result:
            hit['content'] = clean_snippet(hit.get('content'), 600)
        all_result.append({'query': query, 'source': 'tavily', 'result': result})

    # extra sources searched with the TOPIC (not with the docs-style query), then filtered
    topic_query = " ".join(terms) if terms else queries[0]
    for source, tool_fn in (('wikipedia', wikipedia_search), ('arxiv', arxiv_search)):
        hits = _usable(tool_fn.invoke(topic_query), errors, source)
        all_result.append({'query': topic_query, 'source': source, 'result': hits})

    # drop everything that is off-topic or from an excluded site (Scribd, LinkedIn ...)
    dropped = 0
    for item in all_result:
        kept = [h for h in item['result'] if is_relevant(h, terms) and source_tier(h.get('url', ''), terms) < 3]
        dropped += len(item['result']) - len(kept)
        item['result'] = kept
    if dropped:
        print(f"Dropped {dropped} irrelevant or excluded search result(s)")

    # deep read: download the best pages and keep their most informative sentences
    web_hits = [h for item in all_result if item['source'] == 'tavily' for h in item['result']]
    web_hits.sort(key=lambda h: (source_tier(h.get('url', ''), terms), -(h.get('score') or 0)))
    seen = set()
    for hit in web_hits:
        url = hit.get('url')
        if not url or url in seen:
            continue
        seen.add(url)
        pages = _usable(fetch_webpage.invoke({"url": url, "max_chars": 6000}), errors, "fetch")
        if pages:
            excerpt = best_excerpt(clean_snippet(pages[0].get('content'), 6000), terms, 400)
            for same in web_hits:                      # a URL can appear in several queries
                if same.get('url') == url:
                    same['page_excerpt'] = excerpt
        if len(seen) >= DEEP_READ_PAGES:
            break
    return {'search_result': all_result, 'terms': terms, 'tool_errors': errors}


def final_research(state: Research_State):
    """Evidence is built in code from the (filtered, cleaned) search results: no LLM call,
    so it cannot return an empty list or invent a URL."""
    items = evidence_from_hits(state['search_result'], terms=state.get('terms'))
    print(f"Evidence built from search results: {len(items)} source(s)")
    return {'evidence': EvidencePack(evidence=[EvidenceItem(**item) for item in items])}


def after_evidence(state: Research_State):
    evidence = state.get('evidence')
    empty = evidence is None or not evidence.evidence
    if empty and state.get('attempts', 0) < MAX_ATTEMPTS:
        print("No usable evidence - generating new queries")
        return 'query_generator'
    return END


research_graph = StateGraph(Research_State)
research_graph.add_node('query_generator', query_generator)
research_graph.add_node('search_queries', search_queries)
research_graph.add_node('final_research', final_research)
research_graph.add_edge(START, 'query_generator')
research_graph.add_edge('query_generator', 'search_queries')
research_graph.add_edge('search_queries', 'final_research')
research_graph.add_conditional_edges('final_research', after_evidence,
                                     {'query_generator': 'query_generator', END: END})
research_graph = research_graph.compile()


def format_evidence(pack: EvidencePack, task_index: int, terms: List[str] = None) -> str:
    """Numbered source list. The ids ([S<task>-<n>]) are what the writer cites."""
    lines = []
    for i, item in enumerate(pack.evidence, 1):
        title = " ".join((item.title or item.url).split())
        lines.append(f"[S{task_index}-{i}] {title} - {item.url}\n"
                     f"    type: {source_type(item.url, terms)}\n"
                     f"    > {(item.snippets or '').strip()}")
    return "\n".join(lines)


SUMMARY_RULES = """Write a clear, beginner-friendly research summary as plain paragraphs and short bullet lists
(NO tables, NO headings).
RULES:
1. Use ONLY facts written in the evidence snippets. Cite sources with their ids right after each fact,
   for example [S{n}-1]. Never cite an id that does not exist.
2. Do NOT add components, features, class names, APIs or architecture details that are not in the evidence.
3. Every source has a "type" line. What a blog post, code repository, video or third-party page says describes
   ONE author's view, ONE project or ONE product: attribute it ("one example project does ...") - statements
   about what the technology is or does in general must rest on official documentation, an official repository
   or a research paper.
4. Running a graph/workflow is NOT the same as scheduling it. Say a framework schedules jobs, scales or is
   production-ready only if a documentation or paper source says exactly that; otherwise describe it as something
   an external scheduler or platform provides, or leave it out.
5. If part of the task is not covered by the evidence (for example recent developments), do not guess: finish
   with one sentence that starts with 'Gaps:' naming what the sources do not cover.
Reply with the summary text only."""


def generate_text(prompt: str, label: str) -> str:
    response = invoke_with_retry(text_llm, [{'role': 'system', 'content': "You are a careful research specialist."},
                                            {'role': 'user', 'content': prompt}], label=label, expected_output=1200)
    text = as_text(response.content).strip()
    if not text:
        detail = describe_reply(response)
        print(f"  empty model reply for '{label}' [{detail}]")
        audit.log_event("empty_model_reply", label=label, detail=detail)
        raise ValueError("empty response")
    return text


def research_agent(state: State):
    task_index = state['task_index']
    audit.set_context(agent='research_agent', task=task_index)
    print(f"Research agent is started working {task_index}")
    cache_entry = (state.get('cache') or {}).get(task_index)
    out = {}
    evidence_text, terms = "", []

    try:
        # ---------- 1. evidence (reused on retries: searches are not repeated) ----------
        if cache_entry:
            print("Retry: reusing the search results of the previous attempt (no new searches)")
            audit.log_event("reuse_cache")
            evidence = EvidencePack(evidence=[EvidenceItem(**i) for i in cache_entry['evidence']])
            terms = cache_entry.get('terms', [])
        else:
            result = research_graph.invoke({
                "research_prompt": f"Today's date is {date.today().isoformat()}.\n\n{task_context(state)}",
                "task_description": state['task'].description, "attempts": 0})
            evidence, terms = result.get('evidence'), result.get('terms', [])
            if evidence is not None and evidence.evidence:
                out['cache'] = {task_index: {'evidence': [i.model_dump() for i in evidence.evidence], 'terms': terms}}
            else:
                tavily_errors = [e for e in result.get('tool_errors', []) if e.startswith("tavily")]
                failure = failure_record("no search results. " + " ".join(tavily_errors[:2])
                                         if not tavily_errors else " ".join(tavily_errors[:2]))
                if failure['category'] not in ('missing_key', 'auth', 'rate_limit', 'timeout'):
                    failure = classify_error("no search results")
                audit.log_event("no_evidence", **failure)
                response = AgentResult(status='failed', output='',
                                       reason=f"The searches returned no usable results [{failure['category']}].")
                print("\nResearch Agent Result :")
                print(response)
                return {'result': {}, 'agent_result': {task_index: response}, 'failures': {task_index: failure}}

        # ---------- 2. summary (plain text, status decided in code) ----------
        evidence_text = format_evidence(evidence, task_index, terms)
        summary_prompt = (f"{task_context(state)}\n\nToday's date is {date.today().isoformat()}.\n\n"
                          f"Evidence gathered from research:\n{evidence_text}\n\n{SUMMARY_RULES.replace('{n}', str(task_index))}")
        summary = generate_text(summary_prompt, "research summary")

        # ---------- 3. verification ----------
        # (a) every sentence against the sources it cites (free)  (b) LLM second opinion on the final summary.
        # If (a) already found problems the summary is rewritten first, so the LLM check is spent on the version
        # that will actually be delivered.
        sources = parse_sources(evidence_text)
        notes = []
        per_claim = verify_claims(summary, sources, topic_terms=terms)
        llm_issues, rewrite_reason, llm_tried = None, None, False
        if per_claim:
            rewrite_reason = format_issues(per_claim)
        else:
            llm_issues, llm_tried = find_unsupported_claims(summary, compact_evidence(sources)), True   # None = could not run
            if llm_issues:
                rewrite_reason = llm_issues
        if rewrite_reason:
            print(f"Research summary has {len(rewrite_reason)} unsupported/unverified claim(s) - rewriting once")
            audit.log_event("research_rewrite", problems=rewrite_reason[:6])
            try:
                summary = generate_text(
                    summary_prompt + "\n\nYour previous summary contained statements the evidence does not support "
                    "or that cite the wrong source:\n- " + "\n- ".join(rewrite_reason) +
                    "\nRewrite it: remove or correct them, and cite only the sources that really contain each fact."
                    "\n\nPREVIOUS SUMMARY:\n" + summary, "research summary revision")
                per_claim = verify_claims(summary, sources, topic_terms=terms)
                if llm_issues:                                   # the rewrite must be confirmed: check it again below
                    notes.append("LLM fact check flagged claims; the summary was rewritten and checked again")
                    llm_issues, llm_tried = None, False
            except Exception as error:                                          # keep the first summary, say so
                info = failure_record(error)
                notes.append(f"rewrite failed [{info['category']}]; the first summary was kept")
                audit.log_event("research_rewrite_failed", **info)
        if per_claim:
            # last resort: do not deliver sentences that cannot be verified
            summary, removed = remove_claims(summary, per_claim)
            if removed:
                notes.append(f"{len(removed)} unverifiable sentence(s) removed automatically")
                audit.log_event("research_sentences_removed", removed=[r[:120] for r in removed])
                per_claim = verify_claims(summary, sources, topic_terms=terms)
        if not llm_tried:                            # not run yet: check the version that will be delivered
            llm_issues = find_unsupported_claims(summary, compact_evidence(sources))
            if llm_issues:
                notes.append(f"LLM fact check flags {len(llm_issues)} claim(s) in the final summary")
        checks = {
            'per_claim_citations': 'passed' if not per_claim else 'failed',
            'llm_fact_check': 'not_run' if llm_issues is None else ('failed' if llm_issues else 'passed'),
        }
        if llm_issues is None:
            notes.append("LLM fact check could not run")
            print("WARNING: LLM fact check of the research summary was NOT performed")
        notes += format_issues(per_claim, 3)
        validation = make_validation(checks, notes)
        audit.log_event("research_done", sources=len(evidence.evidence), validation=validation['status'])

        response = AgentResult(status='completed', output=summary,
                               reason=f"Research completed from {len(evidence.evidence)} source(s); validation: {validation['status']}.")
        print("\nResearch Agent Result :")
        print(response)
        out.update({'result': {task_index: f"{summary}\n\nSOURCES:\n{evidence_text}"},
                    'agent_result': {task_index: response}, 'validation': {task_index: validation}})
        return out

    except Exception as error:
        failure = failure_record(error)
        audit.log_event("agent_error", **failure)
        print(f"Research agent error [{failure['category']}]: {error}")
        out.update({'result': {},
                    'agent_result': {task_index: AgentResult(status='failed', output='',
                                                             reason=f"Research agent error [{failure['category']}]: {failure['message']}")},
                    'failures': {task_index: failure}})
        return out
