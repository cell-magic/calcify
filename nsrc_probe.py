"""nsrc_probe.py — exercise the NSRC gate and the scar.

    python nsrc_probe.py gate      # show psi / M for training docs and novelty probes
    python nsrc_probe.py crash     # feed an in-well input the kernel must reject
"""
import json
import sqlite3
import sys

import numpy as np

import calcify as C
import example


def _well():
    with sqlite3.connect(C.DB_PATH) as conn:
        row = conn.execute(
            "select s_0, b_depth, scar, status from locked_tasks where task_name='invoice_extract'"
        ).fetchone()
    return np.array(json.loads(row[0])), row[1], row[2], row[3]


def _psi(doc, s_0, b):
    return b - 0.5 * np.linalg.norm(C._embed(doc) - s_0) ** 2


PROBES = {
    "in-well, both $ and EUR": {"text": example.EXAMPLES[0]["text"] + "\nPaid in € by transfer."},
    "in-well, invoice no amounts": {"text": "INVOICE\nStudio Ferrand\nProject: identity work\nAmount to be confirmed.\nThanks."},
    "novel, EUR only": {"text": "Balance: EUR 100.00, no dollar sign at all"},
    "novel, unrelated": {"text": "hello world"},
}

if __name__ == "__main__":
    s_0, b, scar, status = _well()
    if sys.argv[1:] == ["gate"]:
        print(f"status={status} scar={scar:.3f} depth={b:.4f}")
        print("-- training docs --")
        for i, d in enumerate(example.EXAMPLES):
            p = _psi(d, s_0, b)
            print(f"  doc {i}: psi={p:+.4f} M={p - scar:+.4f} {'open' if p - scar > 0 else 'CLOSED'}")
        print("-- probes --")
        for name, p in PROBES.items():
            v = _psi(p, s_0, b)
            print(f"  {name:28s} psi={v:+.4f} M={v - scar:+.4f} {'open' if v - scar > 0 else 'CLOSED'}")
    elif sys.argv[1:] == ["crash"]:
        doc = PROBES["in-well, both $ and EUR"]
        print(f"psi={_psi(doc, s_0, b):+.4f} scar={scar:.3f}/{b:.3f}")
        out = example.extract_invoice(doc)
        print("returned:", out)
        print("after:", _well()[2:])
