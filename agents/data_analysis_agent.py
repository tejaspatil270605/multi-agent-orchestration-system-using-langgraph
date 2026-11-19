"""Data agent. All numbers are computed in code (tools.data_analysis_tools); the LLM only writes a short
plain-language reading, and that reading is rejected if it contains a number that is not in the report.
No tool-calling loop and no JSON generation -> nothing here can fail on malformed model output."""
import os
from state import State
from schemas import AgentResult
from agents import (make_llm, invoke_with_retry, as_text, make_validation, failure_record)
import audit
from tools.data_analysis_tools import (locate_dataset, build_data_report, template_interpretation)
from tools.writing_tools import find_unsupported_numbers

INTERPRET_WITH_LLM = os.getenv("INTERPRET_WITH_LLM", "1") == "1"
llm = make_llm(max_tokens=1500)

SYSTEM_PROMPT = """You are a data analyst writing a short interpretation of a finished statistical report for a beginner.
Rules: 3 to 5 sentences of plain text (no headings, no tables). Use ONLY numbers that appear in the report - never
compute or invent a new number. Say that forecasts are estimates (linear extrapolation). If the report says the
sample is small, say once that conclusions about the distribution shape are weak and do NOT interpret
skewness or the shape of the distribution at all. Do not describe charts (none exist) and
do not talk about any topic other than the dataset."""

MISSING_DATA_MESSAGE = ("No numeric dataset was found in the request or in the supplied input. Please provide the data as "
                        "CSV text (with a header row), JSON, or lines such as 'Month: Jan, Feb, Mar' and "
                        "'Downloads: 1000, 1200, 1500' (at least 3 numbers in a column).")


def interpret(report: str, metrics: dict):
    """-> (text, note). LLM text only if every number in it appears in the report; otherwise a template built in code."""
    note = None
    if INTERPRET_WITH_LLM:
        try:
            response = invoke_with_retry(
                llm, [{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user', 'content': f"REPORT:\n{report}\n\nWrite the interpretation now."}],
                tries=2, label="data interpretation")
            text = as_text(response.content).strip()
            if not text:
                note = "the model returned an empty interpretation; a template was used"
            else:
                invented = find_unsupported_numbers(report, text)
                if not invented:
                    return text, None
                note = f"the model wrote numbers that are not in the report {invented}; a template was used"
        except Exception as error:
            note = f"interpretation LLM unavailable ({failure_record(error)['category']}); a template was used"
    return template_interpretation(metrics), note


def data_agent(state: State):
    task_index = state['task_index']
    audit.set_context(agent='data_agent', task=task_index)
    print(f'Data Agent is started working {task_index}')
    try:
        provenance = state.get('provenance') or {}
        candidates = []
        if state.get('additional_input'):
            info = provenance.get(f"input_for_task_{task_index}") or {}
            candidates.append((f"additional input - {info.get('source', 'supplied by the user')}",
                               state['additional_input']))
        for dep, value in (state.get('result') or {}).items():
            candidates.append((f"output of task {dep}", as_text(value)))
        candidates.append(("original user request", state['user_input']))

        dataset = locate_dataset(candidates)
        if dataset is None:
            audit.log_event("dataset_missing", candidates=[c[0] for c in candidates])
            response = AgentResult(status='waiting_for_input', output=MISSING_DATA_MESSAGE,
                                   reason='No dataset was supplied for analysis.')
            print("\nData analysis agent result :")
            print(response)
            return {'result': {}, 'agent_result': {task_index: response}}

        report, metrics = build_data_report(dataset)
        interpretation, note = interpret(report, metrics)
        output = f"{report}\n\nINTERPRETATION:\n{interpretation}"
        columns = list(dataset['columns'])
        audit.log_event("dataset_used", origin=dataset['origin'], sha256_12=dataset['sha256_12'],
                        rows=max(len(v) for v in dataset['columns'].values()), columns=columns)
        validation = make_validation(
            {'calculations': 'passed', 'interpretation_numbers': 'passed'},
            [note] if note else [])
        response = AgentResult(status='completed', output=output,
                               reason=f"Analysis completed deterministically from: {dataset['origin']}.")
        print("\nData analysis agent result :")
        print(response)
        return {
            'result': {task_index: output},
            'agent_result': {task_index: response},
            'validation': {task_index: validation},
            'provenance': {'dataset': {'task': task_index, 'origin': dataset['origin'],
                                       'sha256_12': dataset['sha256_12'], 'columns': columns}},
        }
    except Exception as error:
        failure = failure_record(error)
        audit.log_event("agent_error", **failure)
        print(f"Data agent error: {error}")
        return {'result': {},
                'agent_result': {task_index: AgentResult(status='failed', output='',
                                                         reason=f"Data agent error [{failure['category']}]: {failure['message']}")},
                'failures': {task_index: failure}}
