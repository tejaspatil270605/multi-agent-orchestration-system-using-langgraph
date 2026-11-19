from state import State
from schemas import AgentResult
from agents import (make_llm, invoke_with_retry, task_context, STATUS_RULES, RECURSION_LIMIT,
                    make_validation, failure_record, as_text)
import audit
from tools.coding_tools import CODING_TOOLS
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

llm = make_llm(max_tokens=3000)
llm_with_tools = llm.bind_tools(CODING_TOOLS)
structured_llm = llm.with_structured_output(AgentResult, method='json_schema')


class CodingAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    agent_result: Optional[AgentResult]


tool_node = ToolNode(CODING_TOOLS)

SYSTEM_PROMPT = """You are a professional coding specialist with these tools:
check_python_syntax, check_imports, run_python_code.

WORKING METHOD (write -> test -> fix)
1. Write the code.
2. Call check_python_syntax, then check_imports (report any missing package in your answer).
3. If the code can run on its own (no API keys, no network, no user files), run it with
   run_python_code and read the output. If it fails, fix it and run again.
   Do not run code that needs secrets, the internet or files you do not have - say so instead.
4. Never claim code works unless you ran it successfully. If you could not run it,
   say "not executed" and why.
5. Never invent data, API responses, research findings or test results."""

FINAL_INSTRUCTION = """Now return the final AgentResult.
- status: completed / waiting_for_input / failed (follow the status rules).
- output: the COMPLETE final code in a ```python block, then a short beginner-friendly
  explanation, any required `pip install` packages, any assumptions (labelled as
  assumptions) and what you actually tested.
- reason: one sentence explaining the status."""


def coding_node(state: CodingAgentState):
    response = invoke_with_retry(llm_with_tools, state['messages'], label="coding agent LLM")
    return {'messages': [response]}


def final_result(state: CodingAgentState):
    messages = state['messages'] + [HumanMessage(content=FINAL_INSTRUCTION)]
    try:
        response = invoke_with_retry(structured_llm, messages, label="final result")
    except Exception as error:                     # structured output failed: fall back to the last plain reply
        info = failure_record(error)
        audit.log_event("structured_output_failed", **info)
        last = ""
        for message in reversed(state['messages']):
            if getattr(message, 'type', '') == 'ai' and not getattr(message, 'tool_calls', None) and message.content:
                last = as_text(message.content)
                break
        if not last.strip():
            raise
        response = AgentResult(status='completed', output=last,
                               reason=f"Structured output failed [{info['category']}]; the last model message was used.")
    return {'agent_result': response}


coding_agent_graph = StateGraph(CodingAgentState)
coding_agent_graph.add_node('coding_node', coding_node)
coding_agent_graph.add_node('final_result_node', final_result)
coding_agent_graph.add_node('tools', tool_node)
coding_agent_graph.add_edge(START, 'coding_node')
coding_agent_graph.add_conditional_edges('coding_node', tools_condition, {
    'tools': 'tools',
    '__end__': 'final_result_node'
})
coding_agent_graph.add_edge('tools', 'coding_node')
coding_agent_graph.add_edge('final_result_node', END)
coding_agent_graph = coding_agent_graph.compile()


def coding_agent(state: State):
    task_index = state['task_index']
    audit.set_context(agent='coding_agent', task=task_index)
    print(f'Coding agent is started working {task_index}')

    prompt = f"""{task_context(state)}

CODING RULES:
- Do NOT invent data, research findings, analysis results, test results,
  requirements or API responses. Use dependency results as source material.
- Optional assumptions are allowed only when they do not block the task and must
  be listed as assumptions.
- Provide correct, runnable Python unless another language is requested.
- Explain the important parts briefly and clearly (beginner-friendly).

{STATUS_RULES}"""

    out = {}
    try:
        result = coding_agent_graph.invoke(
            {'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ]},
            config={'recursion_limit': RECURSION_LIMIT},
        )
        response = result['agent_result']
        # quality: did the final code run successfully inside run_python_code?
        ran_ok = any(getattr(m, 'type', '') == 'tool' and getattr(m, 'name', '') == 'run_python_code'
                     and "'ok': True" in str(m.content) for m in result.get('messages', []))
        checks = {'executed_without_error': 'passed' if ran_ok else 'failed'}
        notes = [] if ran_ok else ["the code was not executed successfully in the sandboxed runner - test it yourself"]
        out['validation'] = {task_index: make_validation(checks, notes)}
    except Exception as error:
        failure = failure_record(error)
        audit.log_event("agent_error", **failure)
        print(f"Coding agent error [{failure['category']}]: {error}")
        response = AgentResult(status='failed', output='',
                               reason=f"Coding agent error [{failure['category']}]: {failure['message']}")
        out['failures'] = {task_index: failure}

    print("\nCoding Agent Result:")
    print(response)
    out.update({
        "result": {task_index: response.output} if response.status == 'completed' else {},
        "agent_result": {task_index: response}
    })
    return out
