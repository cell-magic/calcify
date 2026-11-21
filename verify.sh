#!/usr/bin/env bash
# One-shot proof, v0.2: the kernel serves, the gate routes, tampering refuses, the ledger stays honest.
set -u
lock() { python -c "import sqlite3;c=sqlite3.connect('calcify.db');c.execute(\"update locked_tasks set status='LOCKED' where task_name='invoice_extract'\");c.commit()"; }

echo "== 1. ledger (status · saved · synthesis tokens · compression · scar/depth) =="
python calcify.py status
echo
echo "== 2. eight invoices, served by the locked kernel (0 tokens) =="
python example.py | sed -n '2,4p'
echo
echo "== 3. the well: every training document and probe, measured =="
python nsrc_probe.py gate
echo
echo "== 4. one character changed on disk -> must refuse, not fall back =="
cp calcified/invoice_extract.py /tmp/k.bak
sed -i 's/"USD"/"EUR"/' calcified/invoice_extract.py
python example.py 2>&1 | tail -1
cp /tmp/k.bak calcified/invoice_extract.py && rm -f /tmp/k.bak
lock
echo
echo "== 5. novel format (EUR only) -> gate closes, model answers, lock holds =="
python example.py --drift | head -1
python calcify.py status | tail -1
echo
echo "== 6. restored to locked =="
lock
python calcify.py status | tail -1
