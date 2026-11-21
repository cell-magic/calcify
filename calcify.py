#!/usr/bin/env python3
"""
calcify v0.2 — Thermodynamic extraction with Null-Space Cartilage (NSRC).
EUPL-1.2

A standard-library + numpy decorator that logs agent I/O, synthesizes a pure Python
kernel, mechanically verifies it, and implements an NSRC thermodynamic surplus gate 
to predictively route drift and permanently halt on uncompressible distributions.
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
import zlib
import numpy as np

# --- Configuration & Setup ---
# Resolved against this file, not the caller's CWD: otherwise running the demo
# from another directory silently creates a fresh, empty ledger and every task
# looks unobserved. Override with CALCIFY_DB / CALCIFY_DIR.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("CALCIFY_DB", os.path.join(BASE_DIR, "calcify.db"))
CALCIFIED_DIR = os.environ.get("CALCIFY_DIR", os.path.join(BASE_DIR, "calcified"))

# Own handler, propagate=False: importing calcify must not reconfigure the host app's logging.
logger = logging.getLogger("calcify")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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
            source TEXT DEFAULT 'observe',
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
            status TEXT,
            scar REAL DEFAULT 0.0,
            s_0 TEXT,
            b_depth REAL DEFAULT 0.0
        )
    """)
    
    # NSRC migration for existing databases
    try:
        conn.execute("ALTER TABLE locked_tasks ADD COLUMN scar REAL DEFAULT 0.0")
        conn.execute("ALTER TABLE locked_tasks ADD COLUMN s_0 TEXT")
        conn.execute("ALTER TABLE locked_tasks ADD COLUMN b_depth REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass # columns already exist

    try:
        conn.execute("ALTER TABLE traces ADD COLUMN source TEXT DEFAULT 'observe'")
    except sqlite3.OperationalError:
        pass

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

def _embed(obj, dim=1024) -> np.ndarray:
    """NSRC state projection: character trigram hashing into R^d.

    crc32, not hash(): Python's str hash is salted per process, which would
    make the well centre drift between runs and silently corrupt the gate.
    """
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    vec = np.zeros(dim)
    if len(text) < 3:
        vec[zlib.crc32(text.encode("utf-8")) % dim] += 1.0
    else:
        for i in range(len(text) - 2):
            vec[zlib.crc32(text[i:i + 3].encode("utf-8")) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _damage(task_name: str, reason: str, b_depth: float = 0.0) -> str:
    """NSRC scar: damage is additive and permanent (A3).

    A failed compression attempt — kernel crash, AST rejection, verification
    failure — means the substrate did not take the invariant. It is not a
    transient error and it is not healed by trying again tomorrow.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT scar, b_depth FROM locked_tasks WHERE task_name=?", (task_name,)).fetchone()
    scar = float(row[0] or 0.0) if row else 0.0
    depth = float(row[1] or 0.0) if row else 0.0
    depth = depth or b_depth or 1.0

    new_scar = scar + depth / 3.0 + 0.01
    if new_scar >= depth:
        status = "UNCOMPRESSIBLE"
        logger.error(
            f"[calcify] {task_name} | NSRC SHUT-DOWN: {reason} | scar {new_scar:.3f} >= depth {depth:.3f} "
            f"| the archive is permanently closed. This task will never be synthesized again."
        )
    else:
        status = "STALE"
        logger.warning(
            f"[calcify] {task_name} | {reason} | scar {new_scar:.3f}/{depth:.3f} "
            f"| {int(round((depth - new_scar) / (depth / 3.0)))} strike(s) before shut-down."
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE locked_tasks SET status=?, scar=? WHERE task_name=?", (status, new_scar, task_name))
    return status


# --- The Decorator ---
def calcify_task(task_name: str):
    """
    Wraps an LLM-calling function. If locked, executes the verified local kernel.
    If stale/observing, calls the original function and logs the trace.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # One data argument per trace. Fall back to the kwargs dict so a
            # call like func(x=1, y=2) is traced instead of silently recording null.
            if args:
                input_data = args[0]
            elif "data" in kwargs:
                input_data = kwargs["data"]
            else:
                input_data = dict(kwargs) or None
            
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute("SELECT status, kernel_sha256, scar, s_0, b_depth FROM locked_tasks WHERE task_name=?", (task_name,)).fetchone()
            
            if row and row[0] == "UNCOMPRESSIBLE":
                # The archive has permanently closed. Run LLM directly, do not bloat the trace DB.
                return func(*args, **kwargs)
                
            elif row and row[0] == "LOCKED":
                kernel_sha256, scar, s_0_json, b_depth = row[1], row[2], row[3], row[4]
                
                # --- NSRC Surplus Gate (Predictive Routing) ---
                if s_0_json is not None:
                    s_0 = np.array(json.loads(s_0_json))
                    try:
                        s = _embed(input_data)
                    except Exception:
                        # Unjudgeable input (non-JSON types). The gate cannot decide,
                        # so ask the model rather than crash a locked task.
                        return _observe_and_run(func, task_name, input_data, args, kwargs, source="escaped")
                    # Deficit psi (how far outside the well are we?)
                    psi = b_depth - 0.5 * np.linalg.norm(s - s_0)**2
                    # NSRC Margin M (with beta=1, e=0, x=scar)
                    M = psi - scar
                    
                    if M <= 0:
                        logger.warning(f"[calcify] {task_name} | NSRC SURPLUS GATE CLOSED (M={M:.3f}). Predictive drift. Bypassing kernel.")
                        # Bypass the kernel, run LLM to log trace, but keep status LOCKED.
                        return _observe_and_run(func, task_name, input_data, args, kwargs, source="escaped")
                
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
                    # Kernel crashed on an input that passed the surplus gate.
                    _damage(task_name, f"DRIFT DETECTED: {e}")
                    # Fall through to original func
            
            return _observe_and_run(func, task_name, input_data, args, kwargs, source="observe")
        return wrapper
    return decorator

def _observe_and_run(func, task_name, input_data, args, kwargs, source="observe"):
    # --- Observe Phase (Original LLM Call) ---
    output_data = func(*args, **kwargs)
    
    # Log canonical trace
    in_json = _canonical_json(input_data)
    out_json = _canonical_json(output_data)
    
    # Rough token estimate for observation
    obs_tokens = (len(in_json) + len(out_json)) // 4
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO traces (task_name, input_json, output_json, source) VALUES (?, ?, ?, ?)",
            (task_name, in_json, out_json, source)
        )
        conn.execute(
            "INSERT OR IGNORE INTO locked_tasks (task_name, status) VALUES (?, 'OBSERVING')",
            (task_name,)
        )
        conn.execute(
            "UPDATE locked_tasks SET tokens_observing = tokens_observing + ? WHERE task_name = ?",
            (obs_tokens, task_name)
        )
    
    return output_data


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
    timeout = float(os.environ.get("CALCIFY_TIMEOUT", "300"))
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required for synthesis")
        
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(payload).encode('utf-8'), headers=headers)
    # Without a timeout a dead endpoint hangs the compile forever.
    with urllib.request.urlopen(req, timeout=timeout) as response:
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
def compile_task(task_name: str, review: bool = True):
    with sqlite3.connect(DB_PATH) as conn:
        # Traces the gate routed away are evidence, not training data: they are
        # out-of-distribution by construction and would poison the well.
        rows = conn.execute(
            "SELECT input_json, output_json FROM traces "
            "WHERE task_name=? AND COALESCE(source,'observe') != 'escaped' ORDER BY id",
            (task_name,),
        ).fetchall()
        prev = conn.execute(
            "SELECT status, scar, tokens_observing, tokens_synthesizing, calls_saved "
            "FROM locked_tasks WHERE task_name=?",
            (task_name,),
        ).fetchone()
    traces = rows

    # NSRC B2: past the shut-down threshold the archive never opens again, whatever the budget.
    if prev and prev[0] == "UNCOMPRESSIBLE":
        logger.error(
            f"[calcify] {task_name} | UNCOMPRESSIBLE (scar {prev[1]:.3f}) | refusing to synthesize. "
            f"The distribution does not admit a stable invariant; route it to the model."
        )
        return

    if len(traces) < 5:
        logger.error(f"Not enough traces for {task_name} (need 5, have {len(traces)}).")
        return
        
    # 1. Blind Holdout Split
    split_idx = int(len(traces) * 0.8)
    train_traces = traces[:split_idx]
    holdout_traces = traces[split_idx:]
    
    # --- NSRC Well Boundary Calculation ---
    embeddings = np.array([_embed(json.loads(in_j)) for in_j, _ in train_traces])
    s_0 = np.mean(embeddings, axis=0)
    dists = [np.linalg.norm(s - s_0)**2 for s in embeddings]
    # Well depth is the max distance + 5% margin to encompass the training set
    b_depth = float(max(dists)) * 1.05 if dists else 0.1
    if b_depth < 0.01:
        b_depth = 0.01
    s_0_json = json.dumps(s_0.tolist())
    
    traces_str = "\n".join(f"Input: {in_j}\nOutput: {out_j}\n" for in_j, out_j in train_traces[-40:])
    if len(train_traces) > 40:
        traces_str = f"(most recent 40 of {len(train_traces)} traces)\n" + traces_str
    
    prompt = f"""
    Write a pure Python function `run(data)` that maps the following inputs to their outputs deterministically.
    
    RULES:
    1. Use ONLY these stdlib modules, no others: re, json, math, decimal, string, collections, itertools, functools, unicodedata, datetime.
    2. Do NOT use lookup tables, if/elif chains matching specific inputs, or memorization. Extract the structural invariant.
    3. If an input violates the expected structure, raise a `ValueError`.
    4. No mutable default arguments. No dunder attribute access.
    5. Return ONLY the python code. No explanations.
    
    TRACES:
    {traces_str}
    """
    
    logger.info(f"[calcify] {task_name} | Synthesizing kernel from {len(train_traces)} traces...")
    candidate_code, syn_tokens = _call_llm_for_code(prompt)
    
    # 2. Static AST Purity Check
    if not _check_ast_purity(candidate_code):
        _damage(task_name, "AST: FAILED | impure or non-stdlib candidate", b_depth)
        return
        
    # 3. Hermetic Hash-Invariance Subprocess
    # Write a temporary verification script that tests BOTH train and holdout traces
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(candidate_code)
        f.write("\n\n")
        f.write("import json, sys\n")
        # ensure_ascii keeps the embedded literal safe whatever the console encoding is
        f.write(f"traces = {json.dumps(traces, ensure_ascii=True)}\n")
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
        res = subprocess.run([sys.executable, "-I", verifier_path], env=env, capture_output=True, text=True,
                             encoding='utf-8', errors='replace')
        if res.returncode != 0:
            passed_hash_invariance = False
            detail = (res.stdout or "") + (res.stderr or "")
            logger.error(f"[calcify] {task_name} | verifier (seed {seed}): {detail.strip().splitlines()[-1] if detail.strip() else 'no output'}")
            break
            
    os.unlink(verifier_path)
    
    if not passed_hash_invariance:
        _damage(task_name, "HASH-INVARIANCE: FAILED | candidate does not reproduce the traces", b_depth)
        return

    # 4. Compression Ratio Gate (MDL)
    canonical_traces_bytes = _canonical_json(traces).encode('utf-8')
    code_bytes = candidate_code.encode('utf-8')
    ratio = len(canonical_traces_bytes) / len(code_bytes)
    
    if len(canonical_traces_bytes) >= 1024:
        mdl_str = f"MDL: {ratio:.1f}x"
        if ratio < 1.0:
            _damage(task_name, f"{mdl_str} | kernel is larger than the traces it explains", b_depth)
            return
    else:
        mdl_str = "MDL: N/A (sample too small)"

    if review:
        print("\n" + "="*50)
        print(candidate_code)
        print("="*50 + "\n")
        try:
            ans = input(f"Lock {task_name} with this kernel? [y/N]: ")
        except EOFError:
            ans = ""   # non-interactive shell: no consent is not consent
        if ans.lower() != 'y':
            logger.info("Aborted by user.")
            return
            
    # 5. Lock and Seal
    kernel_sha256 = _sha256(candidate_code)
    traces_digest = _sha256(canonical_traces_bytes.decode('utf-8'))
    
    # NSRC: scar is permanent and additive (A3). A recompile does not heal it.
    # The ledger is lifetime accounting, so nothing is reset by re-sealing either.
    carried_scar = float(prev[1]) if prev and prev[1] else 0.0
    carried_saved = int(prev[4] or 0) if prev else 0
    carried_obs = int(prev[2] or 0) if prev else 0
    total_syn = int(prev[3] or 0) + int(syn_tokens) if prev else int(syn_tokens)
    
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
    
    with open(kernel_path, 'w', encoding='utf-8') as f:
        f.write(final_code)
        
    # Re-hash the file to ensure the DB lock matches disk exactly
    final_sha256 = _file_sha256(kernel_path)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO locked_tasks 
            (task_name, kernel_sha256, traces_digest, compression_ratio,
             tokens_observing, tokens_synthesizing, calls_saved, status, scar, s_0, b_depth) 
            VALUES (?, ?, ?, ?, ?, ?, ?, 'LOCKED', ?, ?, ?)
        """, (task_name, final_sha256, traces_digest, ratio,
              carried_obs, total_syn, carried_saved, carried_scar, s_0_json, b_depth))
        
    scar_str = f" | SCAR: {carried_scar:.3f}/{b_depth:.3f}" if carried_scar > 0 else ""
    logger.info(f"[calcify] {task_name} | AST: PURE | HASH-INVARIANCE: PASS (1,42) | {mdl_str} | WELL: b={b_depth:.4f}{scar_str} | KERNEL: sha256:{final_sha256[:8]}... | LOCKED")


# --- CLI ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: calcify <compile|status> [task_name]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    
    if cmd == "status":
        with sqlite3.connect(DB_PATH) as conn:
            tasks = conn.execute("SELECT task_name, status, calls_saved, tokens_observing, tokens_synthesizing, compression_ratio, scar, b_depth FROM locked_tasks").fetchall()
            print(f"{'TASK':<20} | {'STATUS':<15} | {'SAVED':<6} | {'OBS_TOKS':<8} | {'SYN_TOKS':<8} | {'COMPRESSION':<11} | {'SCAR/DEPTH'}")
            print("-" * 95)
            for t, st, saved, obs, syn, comp, scar, b in tasks:
                comp_str = f"{comp:.1f}x" if comp else "N/A"
                print(f"{t:<20} | {st:<15} | {saved:<6} | {obs:<8} | {syn:<8} | {comp_str:<11} | {scar:.2f}/{b:.2f}")
                
    elif cmd == "compile":
        if len(sys.argv) < 3:
            print("Usage: calcify compile <task_name>")
            sys.exit(1)
        compile_task(sys.argv[2], review=True)
