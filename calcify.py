#!/usr/bin/env python3
"""
calcify v0.1 — Thermodynamic extraction of deterministic invariants from LLM traces.
EUPL-1.2

A zero-dependency stdlib decorator that logs agent I/O, synthesizes a pure Python
kernel once a pattern stabilizes, mechanically verifies it against holdout traces,
and locks the function at 0 tokens if compression > 1.0x.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import urllib.request

# --- Configuration & Setup ---
DB_PATH = "calcify.db"
CALCIFIED_DIR = "calcified"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calcify")

os.makedirs(CALCIFIED_DIR, exist_ok=True)

# Ensure SQLite is WAL mode for concurrency
with sqlite3.connect(DB_PATH) as conn:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY,
            task_name TEXT,
            input_json TEXT,
            output_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS locked_tasks (
            task_name TEXT PRIMARY KEY,
            kernel_sha256 TEXT,
            traces_digest TEXT,
            compression_ratio REAL,
            tokens_observing INTEGER DEFAULT 0,
            tokens_synthesizing INTEGER DEFAULT 0,
            calls_saved INTEGER DEFAULT 0,
            status TEXT
        )
    """)

# --- Core Exceptions ---
class CalcifyDriftError(Exception):
    """Raised by a locked kernel when an input violates its structural invariant."""
    pass

class CalcifySecurityError(Exception):
    """Raised when a kernel file's SHA-256 hash does not match the database lock."""
    pass


# --- Canonical JSON & Hashing ---
def _canonical_json(obj) -> str:
    """Strict canonical JSON stringification to prevent tuple/list drift and NaN leakage."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except Exception as e:
        raise TypeError(f"Calcify trace payload must be strictly JSON-serializable: {e}")

def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _file_sha256(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# --- The Decorator ---
def calcify_task(task_name: str):
    """
    Wraps an LLM-calling function. If locked, executes the verified local kernel.
    If stale/observing, calls the original function and logs the trace.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # We only support single data dicts/strings for traces in v0.1
            input_data = args[0] if args else kwargs.get('data')
            
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute("SELECT status, kernel_sha256 FROM locked_tasks WHERE task_name=?", (task_name,)).fetchone()
            
            if row and row[0] == "LOCKED":
                kernel_sha256 = row[1]
                filepath = os.path.join(CALCIFIED_DIR, f"{task_name}.py")
                
                if not os.path.exists(filepath):
                    raise CalcifySecurityError(f"Kernel file missing for {task_name}")
                if _file_sha256(filepath) != kernel_sha256:
                    raise CalcifySecurityError(f"Tamper detected: SHA-256 mismatch for {task_name}. Refusing to run.")
                
                # Load content-addressed module to defeat sys.modules cache
                spec = importlib.util.spec_from_file_location(f"calcified_{kernel_sha256[:8]}", filepath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                
                try:
                    # Pass a deepcopy to prevent in-place mutation drift
                    res = mod.run(copy.deepcopy(input_data))
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("UPDATE locked_tasks SET calls_saved = calls_saved + 1 WHERE task_name=?", (task_name,))
                    return res
                except Exception as e:
                    logger.warning(f"[calcify] {task_name} | DRIFT DETECTED: {e} | Falling back to LLM.")
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("UPDATE locked_tasks SET status='STALE' WHERE task_name=?", (task_name,))
                    # Fall through to original func
            
            # --- Observe Phase (Original LLM Call) ---
            output_data = func(*args, **kwargs)
            
            # Log canonical trace
            in_json = _canonical_json(input_data)
            out_json = _canonical_json(output_data)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO traces (task_name, input_json, output_json) VALUES (?, ?, ?)",
                    (task_name, in_json, out_json)
                )
                conn.execute(
                    "INSERT OR IGNORE INTO locked_tasks (task_name, status) VALUES (?, 'OBSERVING')",
                    (task_name,)
                )
            
            return output_data
        return wrapper
    return decorator


# --- AST Purity Gate ---
def _check_ast_purity(code: str) -> bool:
    """Static purity lint. Rejects anything outside a strict stdlib allowlist."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
        
    ALLOWED_IMPORTS = {'re', 'json', 'math', 'decimal', 'string', 'collections', 'itertools', 'functools', 'unicodedata', 'datetime'}
    BANNED_BUILTINS = {'open', 'exec', 'eval', '__import__', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars'}
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names]
            mod = getattr(node, 'module', None)
            if mod and mod.split('.')[0] not in ALLOWED_IMPORTS: return False
            for name in names:
                if name.split('.')[0] not in ALLOWED_IMPORTS: return False
        
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            return False
            
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                return False
                
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') or node.attr.endswith('__'):
                return False
                
        elif isinstance(node, ast.FunctionDef):
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    return False # mutable defaults
                    
    return True

# --- Synthesizer (HTTP to LLM) ---
def _call_llm_for_code(prompt: str) -> tuple[str, int]:
    """Basic OpenAI-compatible HTTP client for synthesis. Returns (code, tokens)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required for synthesis")
        
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(payload).encode('utf-8'), headers=headers)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        
    raw_content = data['choices'][0]['message']['content']
    tokens = data['usage']['total_tokens']
    
    # Strip markdown fences
    lines = raw_content.split('\n')
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].startswith('```'):
        lines = lines[:-1]
        
    return '\n'.join(lines).strip(), tokens


# --- Compilation & Verification ---
def compile_task(task_name: str, force: bool = False, review: bool = True):
    with sqlite3.connect(DB_PATH) as conn:
        traces = conn.execute("SELECT input_json, output_json FROM traces WHERE task_name=?", (task_name,)).fetchall()
        
    if len(traces) < 5:
        logger.error(f"Not enough traces for {task_name} (need 5, have {len(traces)}).")
        return
        
    # 1. Blind Holdout Split
    split_idx = int(len(traces) * 0.8)
    train_traces = traces[:split_idx]
    holdout_traces = traces[split_idx:]
    
    traces_str = "\n".join(f"Input: {in_j}\nOutput: {out_j}\n" for in_j, out_j in train_traces)
    
    prompt = f"""
    Write a pure Python function `run(data)` that maps the following inputs to their outputs deterministically.
    
    RULES:
    1. Use ONLY the standard library (re, json, math, datetime, collections, etc).
    2. Do NOT use lookup tables, if/elif chains matching specific inputs, or memorization. Extract the structural invariant.
    3. If an input violates the expected structure, raise a `ValueError`.
    4. Return ONLY the python code. No explanations.
    
    TRACES:
    {traces_str}
    """
    
    logger.info(f"[calcify] {task_name} | Synthesizing kernel from {len(train_traces)} traces...")
    candidate_code, syn_tokens = _call_llm_for_code(prompt)
    
    # 2. Static AST Purity Check
    if not _check_ast_purity(candidate_code):
        logger.error(f"[calcify] {task_name} | AST: FAILED | Code uses banned builtins or non-stdlib imports.")
        return
        
    # 3. Hermetic Hash-Invariance Subprocess
    # Write a temporary verification script that tests BOTH train and holdout traces
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(candidate_code)
        f.write("\n\n")
        f.write("import json, sys\n")
        f.write(f"traces = {json.dumps(traces)}\n")
        f.write("for i, (in_j, expected_out_j) in enumerate(traces):\n")
        f.write("    try:\n")
        f.write("        res = run(json.loads(in_j))\n")
        f.write("    except Exception as exc:\n")
        f.write("        print(f'trace {i}: kernel raised {type(exc).__name__}: {exc}')\n")
        f.write("        sys.exit(1)\n")
        # Assert canonical JSON equivalence to prevent float/int or tuple/list drift
        f.write("    res_j = json.dumps(res, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)\n")
        f.write("    if res_j != expected_out_j:\n")
        f.write("        print(f'trace {i}: expected {expected_out_j}')\n")
        f.write("        print(f'trace {i}: got      {res_j}')\n")
        f.write("        sys.exit(1)\n")
        f.write("sys.exit(0)\n")
        verifier_path = f.name
        
    passed_hash_invariance = True
    for seed in ["1", "42"]:
        env = {"PYTHONHASHSEED": seed, "LC_ALL": "C"}
        # python3 -I: Isolated mode (no PYTHONPATH, no user site-packages)
        res = subprocess.run([sys.executable, "-I", verifier_path], env=env, capture_output=True, text=True)
        if res.returncode != 0:
            passed_hash_invariance = False
            detail = (res.stdout or "") + (res.stderr or "")
            logger.error(f"[calcify] {task_name} | verifier (seed {seed}): {detail.strip().splitlines()[-1] if detail.strip() else 'no output'}")
            break
            
    os.unlink(verifier_path)
    
    if not passed_hash_invariance:
        logger.error(f"[calcify] {task_name} | HASH-INVARIANCE: FAILED | Subprocess verification failed or leaked entropy.")
        return

    # 4. Compression Ratio Gate (MDL)
    canonical_traces_bytes = _canonical_json(traces).encode('utf-8')
    code_bytes = candidate_code.encode('utf-8')
    ratio = len(canonical_traces_bytes) / len(code_bytes)
    
    if len(canonical_traces_bytes) >= 1024:
        mdl_str = f"MDL: {ratio:.1f}x"
        if ratio < 1.0:
            logger.error(f"[calcify] {task_name} | {mdl_str} | FAILED | Code is larger than traces. Lookup-table cheat detected.")
            return
    else:
        mdl_str = "MDL: N/A (sample too small)"

    if review:
        print("\n" + "="*50)
        print(candidate_code)
        print("="*50 + "\n")
        ans = input(f"Lock {task_name} with this kernel? [y/N]: ")
        if ans.lower() != 'y':
            logger.info("Aborted by user.")
            return
            
    # 5. Lock and Seal
    kernel_sha256 = _sha256(candidate_code)
    traces_digest = _sha256(canonical_traces_bytes.decode('utf-8'))
    
    header = textwrap.dedent(f"""\
    # CALCIFICATION RECEIPT
    # Task: {task_name}
    # Kernel body SHA-256: {kernel_sha256}
    # Traces digest: {traces_digest}
    # Compression: {mdl_str}
    # The sealed file's own SHA-256 is recorded in calcify.db and checked on every call.

    """)
    
    final_code = header + candidate_code
    kernel_path = os.path.join(CALCIFIED_DIR, f"{task_name}.py")
    
    with open(kernel_path, 'w') as f:
        f.write(final_code)
        
    # Re-hash the file to ensure the DB lock matches disk exactly
    final_sha256 = _file_sha256(kernel_path)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO locked_tasks 
            (task_name, kernel_sha256, traces_digest, compression_ratio, tokens_synthesizing, status) 
            VALUES (?, ?, ?, ?, ?, 'LOCKED')
        """, (task_name, final_sha256, traces_digest, ratio, syn_tokens))
        
    logger.info(f"[calcify] {task_name} | AST: PURE | HASH-INVARIANCE: PASS (1,42) | {mdl_str} | KERNEL: sha256:{final_sha256[:8]}... | LOCKED")


# --- CLI ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: calcify <compile|status> [task_name]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    
    if cmd == "status":
        with sqlite3.connect(DB_PATH) as conn:
            tasks = conn.execute("SELECT task_name, status, calls_saved, tokens_synthesizing, compression_ratio FROM locked_tasks").fetchall()
            print(f"{'TASK':<20} | {'STATUS':<10} | {'SAVED':<6} | {'SYN_TOKS':<8} | {'COMPRESSION'}")
            print("-" * 65)
            for t, st, saved, syn, comp in tasks:
                comp_str = f"{comp:.1f}x" if comp else "N/A"
                print(f"{t:<20} | {st:<10} | {saved:<6} | {syn:<8} | {comp_str}")
                
    elif cmd == "compile":
        if len(sys.argv) < 3:
            print("Usage: calcify compile <task_name>")
            sys.exit(1)
        compile_task(sys.argv[2], review=True)
