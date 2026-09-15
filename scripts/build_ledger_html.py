"""Inject data/ledger_history.json into the ledger page template and write
data/ledger.html, ready to publish as the "TWN Fear & Greed Ledger"
artifact. The template is the last published page with its
<script id="ledger-data"> payload swapped out; nothing else changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEMPLATE = DATA_DIR / "ledger_template.html"
LEDGER = DATA_DIR / "ledger_history.json"
OUT = DATA_DIR / "ledger.html"

DATA_BLOCK = re.compile(
    r'(<script id="ledger-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")
    rows = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    new_html, n = DATA_BLOCK.subn(lambda m: f"{m.group(1)}{payload}{m.group(3)}", html)
    if n != 1:
        print(f"expected exactly one ledger-data block in the template, found {n}")
        return 1
    OUT.write_text(new_html, encoding="utf-8")
    print(f"wrote {OUT} with {len(rows)} rows ({rows[0]['date']} -> {rows[-1]['date']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
