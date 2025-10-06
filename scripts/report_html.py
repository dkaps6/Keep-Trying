# scripts/report_html.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import html

CSS = """
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
h1, h2, h3 { margin: 0.2em 0; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #eee; padding: 6px 8px; font-size: 14px; }
th { background: #fafafa; position: sticky; top: 0; }
tr.green { background: #e6ffed; }
tr.amber { background: #fff5e6; }
tr.red { background: #fff; }
small.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""

def _class(color: str) -> str:
    c = (color or "").lower()
    if c == "green": return "green"
    if c == "amber": return "amber"
    return "red"

def main():
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    clean = out / "props_priced_clean.csv"
    md    = out / "SUMMARY.md"
    report = out / "report.html"

    if not clean.exists():
        report.write_text("<h1>No props_priced_clean.csv found</h1>", encoding="utf-8")
        print(f"[report] missing {clean}")
        return

    df = pd.read_csv(clean)
    for c in ("edge_abs","p_over_blend","p_market_over"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [
        "event_id","player","team","defense_team","market","line",
        "bet_side","edge_abs","color","bookmaker",
        "over_odds","under_odds","p_market_over","p_over_blend","fair_over_odds",
        "mu_model","wind_mph","temp_f","precip"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df.sort_values("edge_abs", ascending=False)
    rows = []
    for _, r in df.iterrows():
        cls = _class(str(r.get("color","")))
        cells = []
        for k in keep:
            v = r.get(k, "")
            if isinstance(v, float):
                if k == "edge_abs": v = f"{v:.2%}"
                elif "p_" in k:    v = f"{v:.2%}"
                elif k == "mu_model": v = f"{v:.1f}"
            cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")

    # summary md (if present)
    md_html = ""
    if md.exists():
        md_html = f"<pre class='mono'>{html.escape(md.read_text())}</pre>"

    table = (
        f"<table><thead><tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in keep)
        + "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NFL Props Report</title>
<style>{CSS}</style></head><body>
<h1>NFL Props — Report</h1>
<p><small class="mono">props_priced_clean.csv & SUMMARY.md</small></p>
{md_html}
{table}
</body></html>"""

    report.write_text(html_doc, encoding="utf-8")
    print(f"[report] wrote {report.resolve()}")

if __name__ == "__main__":
    main()
