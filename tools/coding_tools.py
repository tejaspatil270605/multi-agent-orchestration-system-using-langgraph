import ast
import os
import subprocess
import sys
import tempfile
import importlib.util

from langchain_core.tools import tool

BLOCKED_IMPORTS = {"subprocess", "socket", "shutil", "ctypes", "multiprocessing", "pty", "signal"}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}
BLOCKED_OS_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "removedirs", "kill",
                    "fork", "execv", "execl", "execvp", "execlp", "spawnl", "spawnv", "rename"}


def _clip(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _unsafe_reason(code: str):
    """Best-effort static guard. This is NOT a sandbox - see README."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORTS:
                    return f"import of '{alias.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BLOCKED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BLOCKED_CALLS:
                return f"call to '{f.id}()' is not allowed"
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "os" and f.attr in BLOCKED_OS_ATTRS):
                return f"call to 'os.{f.attr}()' is not allowed"
    return None


@tool
def check_python_syntax(code: str) -> dict:
    """Check that Python code parses. Returns ok=True or the line number and message of the error."""
    try:
        ast.parse(code)
        return {"ok": True}
    except SyntaxError as e:
        return {"ok": False, "line": e.lineno, "error": f"{e.msg}"}


@tool
def check_imports(code: str) -> dict:
    """List the modules imported by the code and whether each is standard library,
    installed, or missing (missing ones need `pip install`)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"syntax error line {e.lineno}: {e.msg}"}
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    stdlib = sorted(r for r in roots if r in sys.stdlib_module_names)
    others = sorted(roots - set(stdlib))
    installed = [r for r in others if importlib.util.find_spec(r) is not None]
    missing = [r for r in others if r not in installed]
    return {"stdlib": stdlib, "installed": installed, "missing": missing}


@tool
def run_python_code(code: str, timeout_seconds: int = 10) -> dict:
    """Run a Python script in a separate process (temp folder, time limit, output
    truncated) and return returncode, stdout and stderr. Use it to TEST code before
    giving it to the user. Dangerous imports/calls (subprocess, socket, shutil, eval,
    exec, os.system ...) are rejected. Set ENABLE_CODE_EXECUTION=0 to disable."""
    if os.getenv("ENABLE_CODE_EXECUTION", "1") != "1":
        return {"ok": False, "error": "code execution is disabled (ENABLE_CODE_EXECUTION=0)"}
    try:
        reason = _unsafe_reason(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"syntax error line {e.lineno}: {e.msg}"}
    if reason:
        return {"ok": False, "error": f"rejected: {reason}"}

    timeout = max(1, min(int(timeout_seconds), 30))
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG")}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run([sys.executable, "-I", path], capture_output=True,
                                  text=True, timeout=timeout, cwd=folder, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout}s"}
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": _clip(proc.stdout), "stderr": _clip(proc.stderr)}


CODING_TOOLS = [check_python_syntax, check_imports, run_python_code]
