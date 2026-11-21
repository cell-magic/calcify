# calcify v0.1

Thermodynamic extraction of deterministic invariants from LLM traces.

A zero-dependency, standard-library-only Python decorator that intercepts agent I/O, synthesizes a
pure Python kernel once a pattern stabilises, verifies it against a blind holdout split, and locks
the function at zero tokens.

Built as public proof-of-work for the AGA thermodynamic thesis. EUPL-1.2. No telemetry, no account,
no network calls except the one you configure for synthesis.

## The hardening

To prevent silent non-determinism and LLM lookup-table cheating, the extraction primitive is
mechanically guarded:

- **Static AST-purity gate** — bans `eval`, `open`, globals, `getattr`, mutable default arguments
  and non-stdlib imports. Purity is checked statically, not requested in the prompt. This is a purity
  lint against accidental I/O and entropy, not a security sandbox for hostile code: `--review` is
  the trust gate, and it is mandatory.
- **Compression-ratio gate (MDL)** — requires the candidate kernel to be byte-smaller than the
  canonical traces it explains, enforcing a real invariant instead of a lookup table. Below 1 KB of
  trace payload the ratio is meaningless, so it prints `MDL: N/A (sample too small)` and the gate
  falls back to the blind holdout split alone.
- **Blind holdout** — 20% of traces (minimum one) are withheld from synthesis and used only to
  verify.
- **Hermetic verification** — the compile-time check runs in a subprocess with `python3 -I` and a
  stripped environment, twice, under `PYTHONHASHSEED=1` and `42`, asserting byte-identical canonical
  output. Isolation applies to the *verification* step; at runtime the kernel is imported in-process
  for speed.
- **Canonical JSON** — every trace write, comparison and digest goes through one serializer
  (`sort_keys`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`), so tuples, int
  keys and NaN cannot drift across the SQLite boundary.
- **Content-addressed kernels** — loaded via `spec_from_file_location` under a module name keyed by
  SHA-256, defeating stale `sys.modules` bytecode.
- **Tamper and drift are different failures** — a hash mismatch between `./calcified/<task>.py` and
  the database raises `CalcifySecurityError` and refuses to run. A runtime exception from inside
  `run()` trips the circuit breaker, marks the task `STALE`, and falls back to the model.
- **Honest accounting** — `tokens_observing` and `tokens_synthesizing` are stored separately, so
  `status` never hides what the compile cost.

## Usage

```python
from calcify import calcify_task

@calcify_task("extract_invoice")
def extract_invoice(data: dict) -> dict:
    return call_llm(system_prompt="Extract invoice data", user_prompt=data["text"])
```

Run your agent to build the trace ledger, then compile once you have five or more traces:

```bash
python calcify.py compile extract_invoice     # synthesise, verify, review, lock
python calcify.py status                      # ledger, tokens, calls saved, compression
```

Synthesis talks to any OpenAI-compatible endpoint. No vendor lock:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python calcify.py compile extract_invoice

# Ollama, OpenRouter, vLLM, LM Studio — the same three variables
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_MODEL=llama3.1:8b
python calcify.py compile extract_invoice
```

If it passes every gate you get a receipt and a locked kernel:

```text
[calcify] invoice_extract | AST: PURE | HASH-INVARIANCE: PASS (1,42) | MDL: 2.4x | KERNEL: sha256:26bfd67a... | LOCKED
```

All later calls to `extract_invoice` execute the kernel in-process at zero tokens. An input outside
the kernel's domain raises inside `run()`, the breaker trips, and traffic falls back to the model —
the kernel is a partial function with a sharp edge, by design.

## Try it

```bash
python example.py                 # eight messy invoices, traced
python calcify.py compile invoice_extract
python example.py                 # now served by the kernel: 0 tokens
python example.py --drift         # EUR-only document → DRIFT DETECTED → fallback
```

Tamper check, by hand:

```bash
cp calcified/invoice_extract.py /tmp/k.bak
sed -i 's/"USD"/"EUR"/' calcified/invoice_extract.py
python example.py                 # CalcifySecurityError: tamper detected. Refuses to run.
cp /tmp/k.bak calcified/invoice_extract.py
```

## Requirements

Python 3.10+, standard library only. SQLite in WAL mode, created next to the script.
