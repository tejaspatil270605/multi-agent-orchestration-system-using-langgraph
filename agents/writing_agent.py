import re
from state import State
from schemas import AgentResult
from agents import (make_llm, task_context, as_text, invoke_with_retry, find_unsupported_claims,
                    make_validation, failure_record, RECURSION_LIMIT)
import audit
from tools.writing_tools import (find_unsupported_numbers, word_count, readability_score, format_artical,
                                 available_source_ids, cited_source_ids, renumber_citations, max_table_rows,
                                 repeated_table_labels, mentions_hypothetical)
from tools.verification_tools import (parse_sources, verify_claims, format_issues, compact_evidence, issue_key,
                                      remove_claims, foreign_citation_tags, strip_foreign_tags)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# gpt-oss is a reasoning model: reasoning tokens count against max_tokens, so keep it generous
llm = make_llm(max_tokens=6000)

MIN_WORDS, MAX_WORDS = 350, 800
MAX_TABLE_ROWS = 8
MAX_DRAFTS = 3          # first draft + up to 2 revisions; a revision that repeats the same problem stops earlier


class WritingAgentState(TypedDict):
    prompt: str
    source_text: str          # the AUTHORIZED upstream material: declared dependency outputs + user-supplied input
    title: str
    draft: str
    feedback: str
    length_feedback: str
    drafts_written: int
    quality: dict
    checks: dict              # name -> passed | failed | not_run  (becomes the validation record)
    topic_terms: list
    dataset_origin: str
    open_issues: list         # sentence-level problems of the current draft
    last_problem_key: str     # fingerprint of the previous round's problems (detects "no progress")
    stagnant: bool
    final_notes: list
    final_text: str


SYSTEM_PROMPT = """You are the writing specialist of a multi-agent system. You receive
research and data-analysis results and turn them into one clear article. Your reply must be
ONLY the finished article in Markdown - no tool calls, no JSON, no comments about the task.
Use plain ASCII hyphens and quotes."""


def write_draft(state: WritingAgentState):
    prompt = state['prompt']
    if state.get('feedback'):
        prompt += f"\n\nREVISION REQUIRED:\n{state['feedback']}\n\nPREVIOUS DRAFT:\n{state.get('draft', '')}"
    response = invoke_with_retry(
        llm,
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': prompt}],
        label="writing draft", expected_output=1200,
    )
    return {
        'draft': as_text(response.content).strip(),
        'drafts_written': state.get('drafts_written', 0) + 1,
    }


def quality_check(state: WritingAgentState):
    """Deterministic checks done in code. HARD problems (empty, invented numbers, no citations, invented citation
    tags) trigger a revision at once; soft problems (length, tables, wrong dataset description) are merged into the
    same revision, and into the fact-check revision, so a single rewrite fixes everything."""
    draft = state['draft']
    checks = {k: 'not_run' for k in ('numbers', 'citations', 'length_and_tables', 'dataset_description',
                                     'claims_per_sentence', 'llm_fact_check')}
    base = {'open_issues': [], 'stagnant': False}
    if not draft:
        feedback = "Your previous reply was empty. Write the complete article now."
        print(f"Writing agent: revision needed -> {feedback}")
        return {**base, 'feedback': feedback, 'length_feedback': '', 'quality': {}, 'checks': checks}

    # numbers are validated against the authorized upstream material only (formats normalised: 1,520 = 1 520 = 1520)
    unsupported = find_unsupported_numbers(state['source_text'], draft)
    available = available_source_ids(state['source_text'])
    valid_citations = [c for c in cited_source_ids(draft) if c in available]
    foreign = foreign_citation_tags(draft)
    quality = {
        'unsupported_numbers': unsupported,
        'words': word_count.invoke({"text": draft})["words"],
        'readability': readability_score.invoke({"text": draft}),
        'citations': len(set(valid_citations)),
        'table_rows': max_table_rows(draft),
    }
    checks['numbers'] = 'failed' if unsupported else 'passed'
    checks['citations'] = 'failed' if ((available and not valid_citations) or foreign) else 'passed'

    hard = ""
    if unsupported:
        hard = (f"Your draft contains numbers that are not in the source material: {unsupported}. Rewrite the "
                f"article using only numbers that appear in the source material (the data report already lists "
                f"values, step changes, totals and forecasts - do not compute new ones).")
    elif foreign:
        hard = (f"Your draft contains citation tags that do not exist: {foreign}. Only the ids of the SOURCES list "
                f"(for example [S1-2]) may be used. The data report needs no citation: describe the data as the "
                f"supplied dataset without any tag.")
    elif available and not valid_citations:
        hard = ("Your draft has no citations. After each fact taken from the research add the source id from "
                "the SOURCES list, for example [S1-2]. Use only ids that exist.")

    soft = []
    if quality['words'] > MAX_WORDS:
        soft.append(f"Your draft has {quality['words']} words. Shorten it to about 600 words (maximum "
                    f"{MAX_WORDS}) without changing any fact or number.")
    elif quality['words'] < MIN_WORDS:
        soft.append(f"Your draft has only {quality['words']} words. Expand it to about 600 words using only the "
                    f"source material.")
    if quality['table_rows'] > MAX_TABLE_ROWS:
        soft.append(f"Your data table has {quality['table_rows']} rows. Reduce it to at most {MAX_TABLE_ROWS} rows.")
    repeated = repeated_table_labels(draft)
    if len(repeated) >= 3:
        soft.append(f"The same data appears in more than one table ({', '.join(repeated[:4])} ...). "
                    f"Show the data ONCE, in a single table.")
    checks['length_and_tables'] = 'failed' if soft else 'passed'
    origin = state.get('dataset_origin') or ""
    if origin and mentions_hypothetical(draft):
        soft.append(f"The dataset's origin is: {origin}. Do not call it hypothetical, fictional or made-up; "
                    f"describe it as the dataset supplied for this run.")
        checks['dataset_description'] = 'failed'
    else:
        checks['dataset_description'] = 'passed'

    soft_text = "\n".join(soft)
    feedback = (hard + ("\n\n" + soft_text if soft_text else "")) if hard else ""
    if feedback:
        print(f"Writing agent: revision needed -> {feedback[:200]}")
    return {**base, 'feedback': feedback, 'length_feedback': soft_text, 'quality': quality, 'checks': checks}


def after_quality(state: WritingAgentState):
    if state.get('feedback'):                     # hard problem: fix it first (no point fact-checking)
        return 'write_draft' if state.get('drafts_written', 0) < MAX_DRAFTS else 'finalize'
    return 'claim_check'                          # always fact-check a draft that has no hard problem


def claim_check(state: WritingAgentState):
    """1) every sentence against the sources it cites (free, code)  2) LLM second opinion - only when step 1 found
    nothing, because a draft that needs a rewrite anyway should not cost an LLM call.
    A check that cannot run is recorded as not_run - never as passed. If a rewrite repeats the same problems the
    loop stops (no progress) and finalize removes the unverifiable sentences."""
    sources = parse_sources(state['source_text'])
    per_claim = verify_claims(state['draft'], sources, topic_terms=state.get('topic_terms'))
    checks = dict(state.get('checks') or {})
    checks['claims_per_sentence'] = 'failed' if per_claim else 'passed'
    llm_problems = []
    if per_claim:
        checks['llm_fact_check'] = 'not_run'                       # skipped this round on purpose (rewrite first)
    else:
        origin = state.get('dataset_origin') or ""
        evidence = compact_evidence(sources) + (f"\n[DATA] dataset origin: {origin}" if origin else "")
        llm_problems = find_unsupported_claims(state['draft'], evidence)           # None = could not run
        checks['llm_fact_check'] = 'not_run' if llm_problems is None else ('failed' if llm_problems else 'passed')
        if llm_problems is None:
            print("WARNING: LLM fact check of the article was NOT performed")

    problems = format_issues(per_claim) + (llm_problems or [])
    parts = []
    if problems:
        parts.append("The fact check found statements that the cited sources do not support:\n- "
                     + "\n- ".join(problems)
                     + "\nFor EACH of them do one of: delete the sentence, cite the source that really contains the "
                       "fact, or move it into one short paragraph that starts with 'Illustration:'. If the sources do "
                       "not cover something, say so briefly instead of filling the gap. Attribute claims that come "
                       "from a blog post, code repository or third-party page ('one example project does ...'). "
                       "Running a graph is not scheduling: do not say the framework schedules jobs unless a "
                       "documentation source says so. Delete speculative sentences ('could', 'would allow').")
    if state.get('length_feedback'):
        parts.append(state['length_feedback'])
    feedback = "\n\n".join(parts)

    key = issue_key(per_claim)
    stagnant = bool(per_claim) and key == state.get('last_problem_key')
    if stagnant:
        print("Writing agent: the rewrite repeated the same unsupported sentences - stopping revisions")
    if feedback and not stagnant:
        print(f"Writing agent: revision needed -> {len(problems)} claim problem(s)"
              f"{', length/table/dataset' if state.get('length_feedback') else ''}")
    return {'feedback': feedback, 'checks': checks, 'open_issues': per_claim, 'last_problem_key': key,
            'stagnant': stagnant}


def after_claims(state: WritingAgentState):
    if state.get('stagnant'):
        return 'finalize'
    if state.get('feedback') and state.get('drafts_written', 0) < MAX_DRAFTS:
        return 'write_draft'
    return 'finalize'


def finalize(state: WritingAgentState):
    draft = state['draft']
    if not draft:
        return {'final_text': '', 'final_notes': []}
    checks, notes = dict(state.get('checks') or {}), []
    sources = parse_sources(state['source_text'])

    # deterministic last resort: sentences that could not be verified are NOT delivered
    if state.get('open_issues'):
        draft, removed = remove_claims(draft, state['open_issues'])
        if removed:
            notes.append(f"{len(removed)} unverifiable sentence(s) removed automatically: "
                         + " | ".join(r[:90] for r in removed[:3]))
            audit.log_event("writing_sentences_removed", removed=[r[:120] for r in removed])
        remaining = verify_claims(draft, sources, topic_terms=state.get('topic_terms'))
        checks['claims_per_sentence'] = 'failed' if remaining else 'passed'
        notes += format_issues(remaining, 2)
        if checks.get('llm_fact_check') == 'not_run':            # the LLM never saw the cleaned article: check it now
            origin = state.get('dataset_origin') or ""
            llm_problems = find_unsupported_claims(
                draft, compact_evidence(sources) + (f"\n[DATA] dataset origin: {origin}" if origin else ""))
            checks['llm_fact_check'] = 'not_run' if llm_problems is None else ('failed' if llm_problems else 'passed')
            if llm_problems:
                notes.append("LLM fact check still flags: " + " | ".join(p[:90] for p in llm_problems[:3]))

    if foreign_citation_tags(draft):                              # invented tags like [T1] never reach the reader
        notes.append(f"invented citation tags removed: {foreign_citation_tags(draft)}")
        draft = strip_foreign_tags(draft)

    text, references = renumber_citations(draft, state['source_text'])
    # the LLM must not write its own source list - ours only holds sources that were cited
    text = re.sub(r"\n#{1,6}\s*(sources|references)\b.*\Z", "", text, flags=re.I | re.S).rstrip()
    text = format_artical.invoke({"title": state['title'], "content": text})
    if references:
        text = text.rstrip() + "\n\n## Sources\n\n" + references + "\n"
    return {'final_text': text, 'checks': checks, 'final_notes': notes}


writing_agent_graph = StateGraph(WritingAgentState)
writing_agent_graph.add_node('write_draft', write_draft)
writing_agent_graph.add_node('quality_check', quality_check)
writing_agent_graph.add_node('claim_check', claim_check)
writing_agent_graph.add_node('finalize', finalize)
writing_agent_graph.add_edge(START, 'write_draft')
writing_agent_graph.add_edge('write_draft', 'quality_check')
writing_agent_graph.add_conditional_edges('quality_check', after_quality, {
    'write_draft': 'write_draft', 'claim_check': 'claim_check', 'finalize': 'finalize'})
writing_agent_graph.add_conditional_edges('claim_check', after_claims, {
    'write_draft': 'write_draft', 'finalize': 'finalize'})
writing_agent_graph.add_edge('finalize', END)
writing_agent_graph = writing_agent_graph.compile()


def writing_agent(state: State):
    task_index = state['task_index']
    task = state['task']
    audit.set_context(agent='writing_agent', task=task_index)
    print(f"Writing agent is started working {task_index}")

    dataset = (state.get('provenance') or {}).get('dataset') or {}
    origin = dataset.get('origin', '')
    terms = sorted({t for entry in (state.get('cache') or {}).values() for t in entry.get('terms', [])})

    prompt = f"""You are the writing specialist in a multi-agent system.
Complete ONLY the assigned writing task using the information from the previous agents.

{task_context(state)}

DATASET ORIGIN (state it accurately if you mention the data): {origin or 'no dataset was used'}

RULES:
1. Use the dependency results as the authoritative source of information.
2. Preserve all facts exactly as provided by the dependency agents. Do not change
   numbers, dataset values, statistics, percentages, dates, names or calculated results.
   Use ONLY numbers that appear in the dependency results - do not compute new ones (the data report already
   lists values, step changes, totals and forecasts). When you quote a statistic keep its label
   (for example "sample standard deviation"). If the report says the sample is small, do not interpret skewness.
3. Do not invent facts or fill missing information with guesses. Never connect two facts
   causally (for example "X brought Y") unless a source says exactly that. If the task asks
   for something the research does not cover (see its "Gaps:" sentence), say in one sentence
   that the sources do not cover it - never fill the gap from your own knowledge.
4. Do not perform new research or independent data analysis.
5. Do not modify, replace, or create a different dataset. Describe the dataset in a few plain words as its origin
   says (for example "a built-in demo dataset"); never call it hypothetical, fictional or "a real package" unless the
   origin says so, and never mention file names or code constants.
6. Do NOT include code unless the same code appears in the dependency results, and never
   invent library class names, functions or APIs. Attribute any code to the source it came
   from. Do not call a source "official" and do not describe HOW the numbers were produced
   (for example never say a script was written); say they were calculated from the dataset.
7. Use all required dependency results that are relevant to the task. Every source has a
   "type" line. What a "blog post", "code repository", "video" or "third-party / vendor documentation" says
   describes ONE author's view, ONE project or ONE product: attribute it ("one example project does ...",
   "a tutorial blog notes ...") and never state it as a general property of the technology. Statements about what
   the technology is or does in general must rest on official documentation, an official repository or a
   research paper; if only a blog or project supports a point, attribute it or leave it out.
8. If dependency results conflict, keep the provided values and point out the conflict.
9. Write a clear, beginner-friendly response of approximately 500-700 words (never more
   than 800), organised with a title (# heading), suitable ## headings and short paragraphs.
10. Show the data ONCE, in a single compact table of at most 8 rows holding the most
    important values - do not list every statistic and never repeat the table. Do not describe
    charts that were not created. Every section must be traceable to the sources or to the data.
11. CITATIONS: after each fact that comes from the research, add the source id from the
    SOURCES list in the dependency results, for example [S1-2]. Use only ids that exist, and cite the source that
    really contains the fact. Do not write a Sources section yourself. The data report needs NO citation: never
    write tags such as [T1], [D1] or [1].
12. SCOPE: running a graph/workflow is not the same as scheduling it. Do not say a framework schedules jobs, scales
    or is production-ready unless a documentation or research-paper source says exactly that.
13. CONNECTING THE TWO TOPICS: the sources do not describe combining them. Either say so in one sentence, or add ONE
    short paragraph that starts with "Illustration:" (at most 3 sentences, no library code, no claim about what the
    framework "provides" or "supports"). Never write an uncited statement about the framework outside such a paragraph.
14. Return ONLY the final written content. No JSON, no status fields, no notes about
    these instructions."""

    # the AUTHORIZED source material for the number and claim checks: the outputs of the declared
    # dependencies (state['result'] only holds those) + what the user supplied
    source_text = "\n".join(
        [as_text(value) for value in (state.get('result') or {}).values()]
        + [as_text(state.get('additional_input')), state['user_input']]
    )

    try:
        result = writing_agent_graph.invoke(
            {'prompt': prompt, 'source_text': source_text, 'title': task.description[:80],
             'draft': '', 'feedback': '', 'length_feedback': '', 'drafts_written': 0, 'quality': {}, 'checks': {},
             'topic_terms': terms, 'dataset_origin': origin, 'open_issues': [], 'last_problem_key': '',
             'stagnant': False, 'final_notes': [], 'final_text': ''},
            config={'recursion_limit': RECURSION_LIMIT},
        )
        text, quality, checks = result['final_text'], result.get('quality') or {}, result.get('checks') or {}
        if text.strip():
            notes = list(result.get('final_notes') or [])
            if result.get('feedback') and not result.get('open_issues'):
                notes.append(f"unresolved after {result['drafts_written']} drafts: {result['feedback'][:300]}")
            elif result.get('feedback'):
                notes.append(f"{result['drafts_written']} drafts written; remaining feedback was applied by "
                             f"removing the unverifiable sentences")
            if checks.get('llm_fact_check') == 'not_run':
                notes.append("the LLM fact check was not performed")
            if checks.get('claims_per_sentence') == 'not_run':
                notes.append("the per-sentence claim check never ran")
            validation = make_validation(checks, notes)
            response = AgentResult(
                status="completed", output=text,
                reason=f"Writing completed ({quality.get('words', '?')} words, {quality.get('citations', 0)} cited "
                       f"source(s), {result['drafts_written']} draft(s)); validation: {validation['status']}"
                       + (f" - {'; '.join(notes)}" if notes else ""))
            audit.log_event("writing_done", validation=validation['status'], checks=checks, drafts=result['drafts_written'])
            out = {'validation': {task_index: validation}}
        else:
            response = AgentResult(status="failed", output="",
                                   reason="The writing agent produced no text after several attempts.")
            out = {'failures': {task_index: failure_record("empty response")}}
    except Exception as error:
        failure = failure_record(error)
        audit.log_event("agent_error", **failure)
        print(f"Writing agent error [{failure['category']}]: {error}")
        response = AgentResult(status='failed', output='',
                               reason=f"Writing agent error [{failure['category']}]: {failure['message']}")
        out = {'failures': {task_index: failure}}

    print("\nWriting Agent Result:")
    print(response)
    out.update({'result': {task_index: response.output} if response.status == 'completed' else {},
                'agent_result': {task_index: response}})
    return out
