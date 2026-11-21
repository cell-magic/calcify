#!/usr/bin/env bash
# One-shot proof: locked kernel serves traffic, both breakers fire, state restores.
set -u
lock() { python -c "import sqlite3;c=sqlite3.connect('calcify.db');c.execute(\"update locked_tasks set status='LOCKED' where task_name='invoice_extract'\");c.commit()"; }

echo "== 1. ledger =="
python calcify.py status
echo
echo "== 2. eight invoices, served by the locked kernel (0 tokens) =="
python example.py | sed -n '2,4p'
echo
echo "== 3. one character changed on disk → must refuse, not fall back =="
cp calcified/invoice_extract.py /tmp/k.bak
sed -i 's/"USD"/"EUR"/' calcified/invoice_extract.py
python example.py 2>&1 | tail -1
cp /tmp/k.bak calcified/invoice_extract.py && rm -f /tmp/k.bak
lock
echo
echo "== 4. novel format (EUR only) → breaker trips, model answers =="
python example.py --drift | head -1
python calcify.py status | tail -1
echo
echo "== 5. restored to locked =="
lock
python calcify.py status | tail -1
