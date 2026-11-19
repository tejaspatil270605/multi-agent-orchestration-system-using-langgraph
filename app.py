"""Streamlit UI for the Multi-Agent Orchestrator (LangGraph).

Run:   streamlit run app.py

What this file does (and nothing more):
  * imports the existing StateGraph from workflow/graph.py and compiles it with a SQLite checkpointer
  * runs it directly (graph.stream / graph.get_state) - no FastAPI, no second orchestration layer
  * shows the supervisor plan, task status, validation, failures, provenance and the final answer from the REAL state
  * answers the existing HITL interrupt (workflow/hitl.py) with Command(resume=...)

Facts taken from the source code that shape this UI:
  * The State holds ONE request (user_input / plan / completed_tasks). A second request in the same thread would
    find every task already completed, so every new message starts a NEW thread_id.
  * hitl.py only supports "input required" interrupts. There is no approval gate in the graph, so there is no real
    "approve / reject" action; the UI offers "provide input" and "decline (keep paused)".
  * The checkpoint stores the full state, so a run that stopped half-way can be continued with stream(None, config).
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", str(APP_DIR / "checkpoints.sqlite"))
HISTORY_FILE = APP_DIR / "ui_history.json"          # only the list of (thread_id, title) shown in the sidebar
# additional_input is shown unclipped to the writer LLM (agents.task_context), so keep uploads small enough for it
MAX_UPLOAD_CHARS = int(os.getenv("UI_MAX_UPLOAD_CHARS", "10000"))
RECURSION_LIMIT = 60
ATTACH_MARKER = "\n\n[Attached dataset: "

st.set_page_config(page_title="Agent Orchestration System", page_icon="🤖", layout="wide")

AGENT_INFO = {
    "research_agent": ("🔎", "Research"),
    "data_agent": ("📊", "Data analysis"),
    "coding_agent": ("💻", "Coding"),
    "writing_agent": ("✍️", "Writing"),
}
STATUS_BADGE = {
    "pending": "🕒 pending", "running": "⏳ running", "retrying": "🔁 retrying",
    "waiting_for_input": "⏸️ waiting for input", "completed": "✅ completed",
    "failed": "❌ failed", "blocked": "🚫 blocked",
}
VALIDATION_BADGE = {"passed": "✅ passed", "review_required": "⚠️ review required", "not_run": "⚪ not run"}


# ----------------------------------------------------------------------------------------------
# loading the existing workflow
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the LangGraph workflow ...")
def load_workflow():
    """-> (compiled graph, error text). The StateGraph comes from workflow/graph.py; the checkpointer is the
    SQLite saver from requirements.txt (langgraph-checkpoint-sqlite)."""
    try:
        from workflow.graph import graph
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as error:                                   # e.g. GROQ_API_KEY missing at import time
        return None, f"{type(error).__name__}: {error}"
    try:
        connection = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        return graph.compile(checkpointer=SqliteSaver(connection)), None
    except Exception as error:
        return None, f"Could not open the checkpoint database '{CHECKPOINT_DB}': {error}"


def load_meta():
    """Model names and the LLM counters kept by agents/__init__.py (only valid inside this Streamlit process)."""
    try:
        from agents import MODEL_NAME, CHECKER_MODEL, STATS
        return {"model": MODEL_NAME, "checker": CHECKER_MODEL, "stats": STATS}
    except Exception:
        return {}


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def get(obj, key, default=None):
    """Read a key from a dict or an attribute from a pydantic object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def int_keys(mapping) -> dict:
    """Checkpoint round-trips may turn int keys into strings; normalise them back."""
    out = {}
    for key, value in (mapping or {}).items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            out[key] = value
    return out


def new_thread_id() -> str:
    return uuid.uuid4().hex


def load_history() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def register_thread(thread_id: str, title: str):
    history = load_history()
    if all(item.get("thread_id") != thread_id for item in history):
        history.append({"thread_id": thread_id, "title": " ".join(title.split())[:45] or "Untitled",
                        "created": datetime.now().isoformat(timespec="seconds")})
        try:
            HISTORY_FILE.write_text(json.dumps(history, indent=1), encoding="utf-8")
        except Exception:
            pass


def read_snapshot(app, thread_id: str):
    try:
        return app.get_state({"configurable": {"thread_id": thread_id}}), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def pending_interrupts(snapshot) -> list:
    payloads = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            payloads.append(getattr(item, "value", item))
    return payloads


def split_prompt(user_input: str):
    """The UI appends '[Attached dataset: name]' to the request; show the original text and the file separately."""
    if ATTACH_MARKER in (user_input or ""):
        text, rest = user_input.split(ATTACH_MARKER, 1)
        return text, rest.split("]", 1)[0]
    return user_input or "", None


# ----------------------------------------------------------------------------------------------
# dataset upload -> text the existing data agent can read (CSV text / JSON), via the HITL resume value
# ----------------------------------------------------------------------------------------------
def fit_dataframe(df: pd.DataFrame):
    csv_text = df.to_csv(index=False)
    if len(csv_text) <= MAX_UPLOAD_CHARS:
        return csv_text, len(df), False
    low, high, best = 1, len(df), 0
    while low <= high:
        mid = (low + high) // 2
        if len(df.head(mid).to_csv(index=False)) <= MAX_UPLOAD_CHARS:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    if best < 3:
        raise ValueError(f"The rows are too wide: not even 3 rows fit into {MAX_UPLOAD_CHARS} characters.")
    return df.head(best).to_csv(index=False), best, True


def prepare_upload(uploaded):
    if uploaded is None:
        return None
    name, raw = uploaded.name, uploaded.getvalue()
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    try:
        df, original_text = None, None
        if extension == "csv":
            original_text = raw.decode("utf-8-sig", errors="replace")
            df = pd.read_csv(io.StringIO(original_text))
        elif extension in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(raw))                      # first sheet
        elif extension == "json":
            original_text = raw.decode("utf-8-sig", errors="replace")
            data = json.loads(original_text)                         # validates the JSON
            try:
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame(data)
            except Exception:
                try:
                    df = pd.json_normalize(data)
                except Exception:
                    df = None
        else:
            return {"ok": False, "name": name, "error": "Unsupported file type (use CSV, XLSX, XLS or JSON)."}

        if df is not None and df.empty:
            return {"ok": False, "name": name, "error": "The file contains no rows."}
        if original_text is not None and len(original_text) <= MAX_UPLOAD_CHARS:
            text, rows, truncated = original_text, (len(df) if df is not None else None), False
        elif df is not None:
            text, rows, truncated = fit_dataframe(df)
        else:
            return {"ok": False, "name": name,
                    "error": f"The JSON is larger than {MAX_UPLOAD_CHARS} characters and is not tabular, so it "
                             f"cannot be shortened safely."}
        return {"ok": True, "name": name, "text": text, "rows": rows,
                "total_rows": len(df) if df is not None else None, "truncated": truncated,
                "columns": list(df.columns) if df is not None else [],
                "preview": df.head(5) if df is not None else None}
    except ImportError as error:
        package = "openpyxl" if extension == "xlsx" else "xlrd" if extension == "xls" else "a required package"
        return {"ok": False, "name": name, "error": f"Reading this file needs {package}: pip install {package} ({error})"}
    except Exception as error:
        return {"ok": False, "name": name, "error": f"Could not read the file: {error}"}


# ----------------------------------------------------------------------------------------------
# running the graph
# ----------------------------------------------------------------------------------------------
def describe_update(node: str, update) -> str:
    if not isinstance(update, dict):
        return f"▶️ `{node}` finished"
    if node == "supervisor":
        count = len(get(update.get("plan"), "tasks", []) or [])
        return f"🧭 Supervisor created a plan with {count} task(s)"
    if node == "task_router":
        statuses = int_keys(update.get("task_status"))
        return ("🔀 Task router - " + ", ".join(f"task {i}: {s}" for i, s in statuses.items())
                if statuses else "🔀 Task router updated")
    if node in AGENT_INFO:
        icon, label = AGENT_INFO[node]
        results = {i: r for i, r in int_keys(update.get("agent_result")).items() if r is not None}
        detail = "; ".join(f"task {i} -> {get(r, 'status')}" for i, r in results.items())
        return f"{icon} {label} agent finished" + (f" ({detail})" if detail else "")
    if node == "human_input":
        return "🙋 Input received - resuming the workflow"
    return f"▶️ `{node}` finished"


def run_graph(app, thread_id: str, payload, status_box) -> bool:
    """Stream one run (or one resume). Returns True if the graph paused at an interrupt."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    log = st.session_state["activity"].setdefault(thread_id, [])
    interrupted = False
    for chunk in app.stream(payload, config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupted = True
                message = "⏸️ The workflow is waiting for your input"
            else:
                message = describe_update(node, update)
            log.append(f"{datetime.now():%H:%M:%S}  {message}")
            status_box.write(message)
    return interrupted


def safe_run(app, thread_id: str, payload, status_box):
    """Run the graph and turn any exception into a message instead of a crashed page."""
    try:
        interrupted = run_graph(app, thread_id, payload, status_box)
        status_box.update(label="Paused - waiting for your input" if interrupted else "Run finished",
                          state="complete")
        st.session_state["flash_error"] = None
    except Exception as error:
        text = f"{type(error).__name__}: {error}"
        hint = ""
        if "api_key" in text.lower() or "authentication" in text.lower():
            hint = " Check GROQ_API_KEY / TAVILY_API_KEY in your .env file."
        elif "rate limit" in text.lower() or "429" in text or "413" in text:
            hint = " This looks like a rate limit - wait a minute and use 'Continue from checkpoint'."
        st.session_state["flash_error"] = text[:600] + hint
        status_box.update(label="The run stopped with an error", state="error")


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def pick_final(tasks, statuses, results):
    for index in reversed(range(len(tasks))):
        if statuses.get(index) == "completed" and str(results.get(index) or "").strip():
            return index
    return None


def render_plan(tasks, statuses, retries):
    rows = []
    for index, task in enumerate(tasks):
        agent = get(task, "assigned_agent", "?")
        icon, label = AGENT_INFO.get(agent, ("🤖", agent))
        rows.append({
            "Task": index, "Agent": f"{icon} {label}", "Description": get(task, "description", ""),
            "Depends on": ", ".join(map(str, get(task, "dependencies", []) or [])) or "-",
            "Complexity": get(task, "complexity", ""),
            "Status": STATUS_BADGE.get(statuses.get(index, "pending"), statuses.get(index, "pending")),
            "Retries": retries.get(index, 0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_task_cards(tasks, statuses, results, agent_results, validation, failures):
    for index, task in enumerate(tasks):
        agent = get(task, "assigned_agent", "?")
        icon, label = AGENT_INFO.get(agent, ("🤖", agent))
        status = statuses.get(index, "pending")
        with st.expander(f"{icon} Task {index} - {label} agent - {STATUS_BADGE.get(status, status)}"):
            st.markdown(f"**Task:** {get(task, 'description', '')}")
            st.markdown(f"**Expected output:** {get(task, 'expected_output', '')}")
            agent_result = agent_results.get(index)
            if agent_result is not None:
                st.markdown(f"**Agent reason:** {get(agent_result, 'reason', '')}")
            check = validation.get(index)
            if check:
                st.markdown(f"**Validation:** {VALIDATION_BADGE.get(check.get('status'), check.get('status'))}")
                if check.get("checks"):
                    st.dataframe(pd.DataFrame([{"Check": k, "Result": v} for k, v in check["checks"].items()]),
                                 hide_index=True)
                for note in check.get("notes") or []:
                    st.caption(f"• {note}")
            failure = failures.get(index)
            if failure:
                st.error(f"Failure type: {failure.get('category')} (transient: {failure.get('transient')}) - "
                         f"{failure.get('message')}")
            output = results.get(index)
            if output:
                st.markdown("**Output:**")
                st.markdown(str(output))
            elif status in ("completed",):
                st.caption("No output stored for this task.")


def render_audit(tasks, statuses, retries, validation, failures, provenance):
    rows = []
    for index, task in enumerate(tasks):
        inputs = []
        given = provenance.get(f"input_for_task_{index}")
        if given:
            inputs.append(f"{given.get('source')} (sha {given.get('sha256_12')}, {given.get('chars')} chars)")
        dataset = provenance.get("dataset")
        if dataset and dataset.get("task") == index:
            inputs.append(f"dataset: {dataset.get('origin')}")
        rows.append({
            "Task": index, "Agent": get(task, "assigned_agent", ""),
            "Execution": statuses.get(index, "pending"),
            "Validation": (validation.get(index) or {}).get("status", "not_run"),
            "Attempts": retries.get(index, 0) + 1 if statuses.get(index) in ("completed", "failed", "retrying") else 0,
            "Failure type": (failures.get(index) or {}).get("category", "-"),
            "Input provenance": " | ".join(inputs) or "-",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def build_run_summary(thread_id, values, tasks, statuses, retries, validation, failures, provenance, results):
    return {
        "thread_id": thread_id, "request": split_prompt(values.get("user_input", ""))[0],
        "tasks": [{"index": i, "agent": get(t, "assigned_agent"), "description": get(t, "description"),
                   "dependencies": get(t, "dependencies"), "execution": statuses.get(i),
                   "validation": validation.get(i), "failure": failures.get(i), "retries": retries.get(i, 0)}
                  for i, t in enumerate(tasks)],
        "provenance": provenance, "results": {str(k): v for k, v in results.items()},
    }


def render_hitl(thread_id: str, payload, upload):
    ss = st.session_state
    info = payload if isinstance(payload, dict) else {"message": str(payload)}
    task_index = info.get("task_index")
    st.warning(f"🙋 **Input required** - {info.get('message', 'More information is required.')}")
    if info.get("task"):
        st.caption(f"Task: {info['task']}")
    if info.get("details"):
        st.markdown(str(info["details"]))
    st.caption("hitl.py only asks for missing information; the graph has no approval step. "
               "Provide the information below, or decline to keep the run paused.")

    if upload and upload.get("ok"):
        label = f"📎 Use the uploaded file ({upload['name']}"
        label += f", {upload['rows']} rows)" if upload.get("rows") else ")"
        if st.button(label, key=f"use_upload_{thread_id}_{task_index}"):
            ss["pending_resume"] = {"thread_id": thread_id, "kind": "answer",
                                    "value": {"text": upload["text"], "source": f"uploaded file {upload['name']}"}}
            st.rerun()
    elif upload and not upload.get("ok"):
        st.error(f"The uploaded file cannot be used: {upload.get('error')}")

    with st.form(f"hitl_form_{thread_id}_{task_index}"):
        text = st.text_area("Or type / paste the information the task needs", height=150,
                            placeholder="Month: Jan, Feb, Mar\nDownloads: 1000, 1200, 1500")
        submitted = st.form_submit_button("➡️ Submit and resume")
    if submitted:
        if not text.strip():
            st.error("Please enter some information first (or upload a file in the sidebar).")
        else:
            ss["pending_resume"] = {"thread_id": thread_id, "kind": "answer", "value": text}
            st.rerun()
    if st.button("⛔ Decline (keep paused)", key=f"decline_{thread_id}_{task_index}"):
        ss["declined"].append(thread_id)
        st.rerun()


def render_assistant(thread_id: str, snapshot, values: dict, upload):
    ss = st.session_state
    plan = values.get("plan")
    tasks = list(get(plan, "tasks", []) or [])
    statuses = int_keys(values.get("task_status"))
    retries = int_keys(values.get("retry_count"))
    results = int_keys(values.get("result"))
    agent_results = int_keys(values.get("agent_result"))
    validation = int_keys(values.get("validation"))
    failures = int_keys(values.get("failures"))
    provenance = values.get("provenance") or {}
    interrupts = pending_interrupts(snapshot)
    finished = not snapshot.next
    done = sum(1 for i in range(len(tasks)) if statuses.get(i) == "completed")
    all_done = bool(tasks) and done == len(tasks)

    if ss.get("flash_error"):
        st.error(f"⚠️ {ss['flash_error']}")

    # ---- overall status ----
    if not tasks:
        st.info("The supervisor has not produced a plan yet.")
    elif interrupts:
        st.info(f"⏸️ Paused - {done}/{len(tasks)} task(s) completed, waiting for your input.")
    elif not finished:
        st.warning(f"The run did not reach the end ({done}/{len(tasks)} task(s) completed). "
                   f"The checkpoint is saved; you can continue from it.")
        if st.button("▶️ Continue from checkpoint", key=f"continue_{thread_id}"):
            ss["pending_resume"] = {"thread_id": thread_id, "kind": "continue", "value": None}
            st.rerun()
    elif all_done:
        st.success(f"All {len(tasks)} task(s) completed.")
    else:
        st.error(f"The workflow ended with {done}/{len(tasks)} task(s) completed.")
        for index in range(len(tasks)):
            if statuses.get(index) in ("failed", "blocked"):
                reason = get(agent_results.get(index), "reason", "")
                st.markdown(f"- Task {index}: **{statuses.get(index)}** {('- ' + reason) if reason else ''}")
    if tasks:
        st.progress(done / len(tasks), text=f"{done}/{len(tasks)} tasks completed")

    # ---- workflow monitor ----
    if tasks:
        with st.expander("🧭 Workflow monitor - supervisor plan and task progress", expanded=True):
            render_plan(tasks, statuses, retries)
            render_task_cards(tasks, statuses, results, agent_results, validation, failures)
    log = ss["activity"].get(thread_id)
    if log:
        with st.expander("🛰️ Agent activity (this browser session)"):
            st.code("\n".join(log), language="text")

    # ---- human in the loop ----
    if interrupts:
        if thread_id in ss["declined"]:
            st.info("You declined to provide the information. The run stays paused in the checkpoint.")
            if st.button("Reopen the request", key=f"reopen_{thread_id}"):
                ss["declined"].remove(thread_id)
                st.rerun()
        else:
            render_hitl(thread_id, interrupts[0], upload)

    # ---- final answer ----
    if finished and tasks:
        final_index = pick_final(tasks, statuses, results)
        if final_index is not None:
            st.markdown("### ✅ Final answer")
            reviews = [i for i, v in validation.items() if v.get("status") == "review_required"]
            not_run = [i for i, v in validation.items() if v.get("status") == "not_run"]
            if reviews:
                st.warning("⚠️ REVIEW REQUIRED - validation found problems in task(s) "
                           f"{reviews}. Do not treat this result as verified; see the notes in the workflow monitor.")
            if not_run:
                st.info(f"Validation did not run for task(s) {not_run}; those results are unchecked.")
            st.markdown(str(results[final_index]))
            summary = build_run_summary(thread_id, values, tasks, statuses, retries, validation, failures,
                                        provenance, results)
            col1, col2 = st.columns(2)
            col1.download_button("⬇️ Download answer (.md)", data=str(results[final_index]),
                                 file_name=f"result_{thread_id[:8]}.md", mime="text/markdown",
                                 key=f"dl_md_{thread_id}")
            col2.download_button("⬇️ Download run summary (.json)",
                                 data=json.dumps(summary, indent=2, default=str, ensure_ascii=False),
                                 file_name=f"run_{thread_id[:8]}.json", mime="application/json",
                                 key=f"dl_json_{thread_id}")
        with st.expander("🧾 Audit summary (execution vs validation)"):
            render_audit(tasks, statuses, retries, validation, failures, provenance)
            st.caption("Execution = did the task run. Validation = is the result trustworthy. "
                       "The full event log is written by audit.py (see AUDIT_LOG_DIR in your .env).")


def render_welcome():
    st.markdown("#### Ask the agents anything")
    st.markdown(
        "Type a request below, for example *\"Research LangGraph and write a beginner-friendly article\"* or "
        "*\"Analyze my dataset and explain the trend\"*. The supervisor plans the tasks, the research / data / coding / "
        "writing agents run them, and every result is validated before it is shown."
    )
    st.caption("Each message starts a new run (the graph state holds exactly one request per thread).")


def render_sidebar(meta: dict):
    ss = st.session_state
    with st.sidebar:
        st.title("🤖 Agent Orchestration")
        if st.button("➕ New Chat", key="new_chat"):
            ss["thread_id"] = new_thread_id()
            ss["flash_error"] = None
            st.rerun()

        st.subheader("Conversations")
        history = load_history()
        if not history:
            st.caption("No conversations yet.")
        for item in reversed(history[-25:]):
            current = item["thread_id"] == ss["thread_id"]
            if st.button(("▶ " if current else "💬 ") + item["title"], key=f"hist_{item['thread_id']}"):
                ss["thread_id"] = item["thread_id"]
                ss["flash_error"] = None
                st.rerun()

        st.divider()
        st.subheader("📎 Dataset")
        uploaded = st.file_uploader("CSV, XLSX, XLS or JSON", type=["csv", "xlsx", "xls", "json"], key="dataset_file")
        upload = prepare_upload(uploaded)
        if upload:
            if upload["ok"]:
                st.success(f"{upload['name']}: {upload['rows'] or '?'} row(s) ready")
                if upload["truncated"]:
                    st.warning(f"Only the first {upload['rows']} of {upload['total_rows']} rows are sent "
                               f"(limit {MAX_UPLOAD_CHARS} characters, set UI_MAX_UPLOAD_CHARS to change it). "
                               f"Statistics will describe these rows only.")
                if upload["preview"] is not None:
                    st.dataframe(upload["preview"], hide_index=True)
                st.caption("The file is handed to the data agent when it asks for data (input request).")
            else:
                st.error(upload["error"])

        st.divider()
        with st.expander("ℹ️ Project information"):
            st.markdown(
                "**Flow:** supervisor → task router → research / data / coding / writing agents → task router → end, "
                "with human-in-the-loop and SQLite checkpointing.\n\n"
                "**Execution vs validation:** every task has an execution status and a separate validation status; "
                "results with failed or skipped checks are flagged *review required*.\n\n"
                f"**Checkpoint DB:** `{Path(CHECKPOINT_DB).name}`"
            )
            if meta.get("model"):
                st.markdown(f"**Model:** `{meta['model']}`  \n**Fact checker:** `{meta['checker']}`")
            st.caption("Not in the source code, so not offered here: long-term memory, approval gates, charts.")
        stats = meta.get("stats")
        if stats:
            with st.expander("⏱️ LLM usage (this process)"):
                st.write({"LLM calls": stats.get("llm_calls"), "LLM errors": stats.get("llm_errors"),
                          "tokens": stats.get("tokens"),
                          "seconds waiting for rate limits": round(stats.get("gate_wait_s", 0)
                                                                   + stats.get("backoff_wait_s", 0), 1)})
    return upload


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    ss = st.session_state
    ss.setdefault("thread_id", new_thread_id())
    ss.setdefault("activity", {})
    ss.setdefault("declined", [])
    ss.setdefault("pending_resume", None)
    ss.setdefault("flash_error", None)

    st.title("🤖 Agent Orchestration System")
    st.caption("Supervisor · Research · Data analysis · Coding · Writing - powered by LangGraph")

    app, load_error = load_workflow()
    meta = load_meta() if app is not None else {}
    upload = render_sidebar(meta)

    if app is None:
        st.error("The workflow could not be loaded, so nothing can run.")
        st.code(load_error or "unknown error", language="text")
        st.markdown("Typical causes: `GROQ_API_KEY` missing in `.env` (the agents create their LLMs when they are "
                    "imported), a missing package, or running `streamlit run app.py` from a different folder than "
                    "the project root.")
        return

    prompt = st.chat_input("Describe what you want the agents to do ...")
    pending = ss.get("pending_resume")

    if prompt:
        prompt = prompt.strip()
        snapshot, _ = read_snapshot(app, ss["thread_id"])
        if snapshot is not None and snapshot.values:                  # this thread already holds a request
            ss["thread_id"] = new_thread_id()
        thread_id = ss["thread_id"]
        user_input = prompt
        if upload and upload.get("ok"):
            user_input += f"{ATTACH_MARKER}{upload['name']}]"
        register_thread(thread_id, prompt)
        ss["flash_error"] = None
        with st.chat_message("user"):
            st.markdown(prompt)
            if upload and upload.get("ok"):
                st.caption(f"📎 {upload['name']}")
        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("Supervisor is planning ...", expanded=True)
            safe_run(app, thread_id, {"user_input": user_input}, status_box)
        st.rerun()

    elif pending:
        ss["pending_resume"] = None
        thread_id = pending["thread_id"]
        snapshot, _ = read_snapshot(app, thread_id)
        values = dict(snapshot.values or {}) if snapshot is not None else {}
        text, attachment = split_prompt(values.get("user_input", ""))
        with st.chat_message("user"):
            st.markdown(text)
            if attachment:
                st.caption(f"📎 {attachment}")
        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("Resuming the workflow ...", expanded=True)
            payload = None if pending["kind"] == "continue" else Command(resume=pending["value"])
            safe_run(app, thread_id, payload, status_box)
        st.rerun()

    else:
        thread_id = ss["thread_id"]
        snapshot, error = read_snapshot(app, thread_id)
        if error:
            st.error(f"Could not read the saved state of this conversation: {error}")
            return
        values = dict(snapshot.values or {})
        if not values:
            render_welcome()
            return
        text, attachment = split_prompt(values.get("user_input", ""))
        with st.chat_message("user"):
            st.markdown(text)
            if attachment:
                st.caption(f"📎 {attachment}")
        with st.chat_message("assistant", avatar="🤖"):
            render_assistant(thread_id, snapshot, values, upload)


main()