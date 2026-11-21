# calcify v0.2

Thermodynamic extraction of deterministic invariants from LLM traces, with null-space cartilage.

Wrap an LLM call. Once a pattern stabilises, `calcify` synthesizes a pure Python kernel, verifies it
against a blind holdout, and locks it at zero tokens. v0.2 adds the part v0.1 got wrong: an explicit
answer to *when to stop trying*.

EUPL-1.2. Local files, local SQLite. No telemetry. Synthesis talks to any OpenAI-compatible endpoint,
or none at all once a task is locked.

Dependency: `numpy` (one, for the state projection). Everything else is standard library.

## Why v0.2 exists

v0.1 was a crystalline lock, and crystals shatter. On a stream where formats mutate, the sequence was:

```
compile → format drifts → kernel raises → fall back → recompile → format drifts → ...
```

Each cycle burns a full synthesis. On a genuinely high-entropy family that loop never terminates, and
the tool costs more than the model it was meant to replace. The fix is not a better kernel. It is a
control law that knows when there is no kernel to be had.

## The gate, and the scar

A task's state is projected into a fixed-dimensional space (character-trigram hashing, `crc32` — not
Python's salted `hash`). The centre of the mass of the traces that produced the kernel is the **well
centre** $s_0$; the largest distance any of them sat from it, plus 5%, is the **well depth** $b$.

Before the kernel is ever imported:

$$\psi = b - \tfrac12\lVert s - s_0\rVert^2, \qquad M = \psi - x$$

- $M > 0$ — the input is inside the well. Serve it locally, at zero tokens.
- $M \le 0$ — **the surplus gate closes.** The input is routed to the model *before* the kernel runs.
  Status stays `LOCKED`; nothing is smashed, nothing is spent on an exception.

$x$ is the **scar**: accumulated, additive, permanent. It rises on every failed compression attempt —
a kernel crash on in-domain input, an AST rejection, a verification failure, a failed compression
ratio. A recompile does not heal it. When

$$x \ge b$$

the task goes **`UNCOMPRESSIBLE`** and the archive closes permanently: `compile` refuses without
calling the model at all, and every call is routed to the LLM forever. The limiting resource was never
memory or budget. It is undamaged substrate.

## The hardening (unchanged from v0.1)

- **Static AST-purity gate** — bans `eval`, `open`, `globals`, `getattr`, mutable defaults, non-stdlib
  imports. A purity lint against accidental I/O, *not* a security sandbox for hostile code.
  `--review` is the trust gate and it is mandatory.
- **Compression-ratio gate (MDL)** — the kernel must be byte-smaller than the traces it explains.
  Below 1 KB of trace payload the ratio is meaningless and prints `MDL: N/A (sample too small)`.
- **Blind holdout** — 20% of traces withheld from synthesis, used only to verify.
- **Hermetic verification** — `python3 -I`, stripped environment, twice, under `PYTHONHASHSEED=1` and
  `42`, asserting byte-identical canonical JSON. Isolation applies to verification; at runtime the
  kernel is imported in-process for speed.
- **Canonical JSON** — one serializer for every write, comparison and digest.
- **Content-addressed kernels** — loaded by SHA-256-keyed module name, defeating `sys.modules` cache.
- **Tamper ≠ drift** — a hash mismatch raises `CalcifySecurityError` and refuses to run. A runtime
  exception trips the breaker and routes to the model.
- **Honest accounting** — `tokens_observing` and `tokens_synthesizing` are separate, and `status`
  shows both with the scar.

## Usage

```python
from calcify import calcify_task

@calcify_task("extract_invoice")
def extract_invoice(data: dict) -> dict:
    return call_llm(system_prompt="Extract invoice data", user_prompt=data["text"])
```

```bash
python calcify.py compile extract_invoice     # synthesise, verify, review, lock
python calcify.py status                      # ledger, tokens, compression, scar/depth
```

Synthesis takes any OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=sk-...                                    # OpenAI
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1                # Ollama, OpenRouter, vLLM, LM Studio
export OPENAI_MODEL=llama3.1:8b
python calcify.py compile extract_invoice
```

A successful lock prints:

```text
[calcify] invoice_extract | AST: PURE | HASH-INVARIANCE: PASS (1,42) | MDL: 2.9x | WELL: b=0.6323 | KERNEL: sha256:45a05a50... | LOCKED
```

A closed gate prints:

```text
[calcify] invoice_extract | NSRC SURPLUS GATE CLOSED (M=-0.052). Predictive drift. Bypassing kernel.
```

And the end of the road prints:

```text
[calcify] invoice_extract | NSRC SHUT-DOWN: HASH-INVARIANCE: FAILED | scar 0.662 >= depth 0.632 | the archive is permanently closed.
[calcify] invoice_extract | UNCOMPRESSIBLE (scar 0.662) | refusing to synthesize.
```

## Measured, not asserted

Eight messy invoices, six synthesised, two withheld. The gate was checked against every training
document and four probes:

| input | ψ | gate |
|---|---|---|
| the eight demo documents (six synthesised, two withheld) | +0.33 … +0.52 | open (all served) |
| same invoice, one amount edited | +0.42 | open |
| same invoice, `$` replaced by `USD` | +0.34 | open → kernel rejects → scar |
| EUR-only document | −0.05 | **closed**, model answers, status stays `LOCKED` |
| unrelated short string | −0.08 | **closed** |

Then, on a contaminated family, the whole shut-down ran live: crash → scar 0.221 → `STALE`;
failed verification → 0.442, "1 strike before shut-down"; failed verification → 0.662 ≥ 0.632 →
`UNCOMPRESSIBLE`; a fourth `compile` returned in milliseconds without generating a token.

## What v0.2 does not do

**It does not repair the input.** The null-space retrieval in the underlying algebra projects a
*state*, and a trigram projection here has no inverse — you cannot decode a vector back into a
document, so there is nothing to hand the kernel. The gate is a router, not a healer. It decides
"serve locally" or "ask the model" earlier and more cheaply than an exception would; it does not
widen the kernel's domain. A learned repair operator is a separate, harder object.

**The well is a heuristic.** Depth is max training distance × 1.05 in a 1024-dim hashed space. It is
not a calibrated density model, and 1024 dimensions of trigrams will collide. It is deliberately
cheap — no model, no training — but not free: the projection is one pass over the input's trigrams,
so it costs a few milliseconds on a multi-kilobyte document, not microseconds. Treat the radius as a
knob to tune against your own false-route rate, not a theorem.

## Try it

```bash
python example.py                 # eight invoices, traced
python calcify.py compile invoice_extract
python example.py                 # served by the kernel, zero tokens
python example.py --drift         # gate closes, model answers
python nsrc_probe.py gate         # ψ and M for every document and probe
python test_ledger.py             # offline: the accounting survives a re-seal
```

Tamper check:

```bash
cp calcified/invoice_extract.py /tmp/k.bak
sed -i 's/"USD"/"EUR"/' calcified/invoice_extract.py
python example.py                 # CalcifySecurityError: refuses to run
cp /tmp/k.bak calcified/invoice_extract.py
```

## Requirements

Python 3.10+, `numpy`. SQLite in WAL mode.

The ledger and the kernels live next to `calcify.py`, not next to your script — otherwise running
from two directories silently creates two ledgers and every task looks unobserved. Override them,
and the synthesis timeout, with:

```bash
export CALCIFY_DB=/path/to/calcify.db
export CALCIFY_DIR=/path/to/calcified
export CALCIFY_TIMEOUT=300          # seconds, synthesis only
```

`tokens_observing` is an estimate (characters ÷ 4), not a provider reading. `tokens_synthesizing` is
the provider's own count and accumulates across recompiles — nothing in the ledger is ever reset by
a re-seal. Traces the gate routed away are stored with `source='escaped'` and excluded from training:
they are out-of-distribution by construction, and feeding them back would poison the well.
