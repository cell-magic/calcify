"""test_ledger.py — offline regression test for the accounting bugs.

No network, no model: the synthesis call is replaced by the kernel already on
disk, so the test exercises the real verify → lock → seal path.

    python test_ledger.py
"""
import json
import os
import sqlite3
import sys
import tempfile

# Point the module at a scratch ledger *before* importing it.
_tmp = tempfile.mkdtemp(prefix="calcify-test-")
os.environ["CALCIFY_DB"] = os.path.join(_tmp, "calcify.db")
os.environ["CALCIFY_DIR"] = os.path.join(_tmp, "calcified")

import calcify as C          # noqa: E402
import example               # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{f' — {detail}' if detail and not cond else ''}")
    if not cond:
        FAIL.append(name)


print(f"scratch ledger: {C.DB_PATH}")

# 1. Observe: eight documents through the decorator, no kernel yet.
for doc in example.EXAMPLES:
    example.extract_invoice(doc)

with sqlite3.connect(C.DB_PATH) as conn:
    n = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    row = conn.execute("SELECT status, tokens_observing FROM locked_tasks").fetchone()
check("eight traces recorded", n == 8, f"got {n}")
check("task is OBSERVING", row[0] == "OBSERVING", row[0])
check("observation tokens counted", row[1] > 0, f"got {row[1]}")

# 2. First lock, with the trace-derived kernel as the candidate.
import importlib.util  # noqa: E402

_kernel_path = os.path.join(os.path.dirname(os.path.abspath(C.__file__)), "calcified", "invoice_extract.py")
if not os.path.exists(_kernel_path):
    print("  SKIP  no reference kernel on disk; run the live compile once first")
    sys.exit(0)

body = open(_kernel_path, encoding="utf-8").read()
_lines = body.splitlines()
_i = 0
while _i < len(_lines) and (not _lines[_i].strip() or _lines[_i].lstrip().startswith("#")):
    _i += 1
body = "\n".join(_lines[_i:])   # strip the receipt header only, keep module constants
C._call_llm_for_code = lambda prompt: (body, 1000)
C.compile_task("invoice_extract", review=False)

with sqlite3.connect(C.DB_PATH) as conn:
    st, syn, obs, saved = conn.execute(
        "SELECT status, tokens_synthesizing, tokens_observing, calls_saved FROM locked_tasks"
    ).fetchone()
check("locked after first compile", st == "LOCKED", st)
check("synthesis tokens recorded", syn == 1000, syn)

# 3. Serve traffic, then recompile: the ledger must survive a re-seal.
for doc in example.EXAMPLES:
    example.extract_invoice(doc)
with sqlite3.connect(C.DB_PATH) as conn:
    before = conn.execute("SELECT calls_saved, tokens_observing FROM locked_tasks").fetchone()
check("kernel served eight calls", before[0] == 8, before[0])

C._call_llm_for_code = lambda prompt: (body, 500)
C.compile_task("invoice_extract", review=False)
with sqlite3.connect(C.DB_PATH) as conn:
    after = conn.execute(
        "SELECT calls_saved, tokens_observing, tokens_synthesizing FROM locked_tasks"
    ).fetchone()
check("calls_saved survives recompile", after[0] == before[0], f"{before[0]} -> {after[0]}")
check("observation tokens survive recompile", after[1] == before[1], f"{before[1]} -> {after[1]}")
check("synthesis tokens accumulate", after[2] == 1500, after[2])

# 4. The gate: a novel document must not reach the kernel and must not kill the lock.
saved_before = after[0]
out = example.extract_invoice({"text": "Balance: EUR 100.00, no dollar sign at all"})
with sqlite3.connect(C.DB_PATH) as conn:
    st, saved_after = conn.execute("SELECT status, calls_saved FROM locked_tasks").fetchone()
    src = conn.execute("SELECT source FROM traces ORDER BY id DESC LIMIT 1").fetchone()[0]
check("model answered the novel input", out.get("currency") == "UNKNOWN", out)
check("status still LOCKED", st == "LOCKED", st)
check("no call credited to the kernel", saved_after == saved_before, saved_after)
check("escaped trace tagged", src == "escaped", src)

# 5. Escaped traces are evidence, not training data.
with sqlite3.connect(C.DB_PATH) as conn:
    trainable = conn.execute(
        "SELECT count(*) FROM traces WHERE COALESCE(source,'observe') != 'escaped'"
    ).fetchone()[0]
check("escaped trace excluded from training set", trainable == 8, trainable)

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("all checks passed")
