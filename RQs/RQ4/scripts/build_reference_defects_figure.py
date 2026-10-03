"""Build and export the reference-review figure with one candidate reference defect per kind.

Run: conda run --no-capture-output -n wd python RQs/RQ4/scripts/build_reference_defects_figure.py
The card layout follows the earlier draw.io figure. Counts come from results/v5/screen_tasks.csv, and the code
excerpts come from the hash-checked reference sources listed in results/v5/reference_sources.csv. Export uses the
draw.io viewer in Playwright's Chromium and writes RQs/RQ4/fig/rq4_reference_defects.{drawio,pdf}.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re
import shutil
import subprocess
import tempfile
from threading import Thread
import urllib.request
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "RQs/RQ4/results/reference_screen"
ASSETS = ROOT / "RQs/RQ4/figures/assets"
PAPER_FIG = ROOT / "RQs/RQ4/figures"
VIEWER_URL = "https://viewer.diagrams.net/js/viewer-static.min.js"
VIEWER_CACHE = ASSETS / "vendor/viewer-static.min.js"

HAND = "fontFamily=Comic Sans MS;"
SANS = "fontFamily=Helvetica;"
MONO_CSS = "font-family:Menlo,Monaco,&quot;Courier New&quot;,monospace;"
INK, HDR_BLUE, CARD_STROKE = "#1d2b3a", "#1261C5", "#666666"
KW_BLUE, NUM_GREEN = "#0000ff", "#098658"
DEFECT = "#c0457a"              # count badge, as in the earlier figure
SLOT_RED = "#d9534f"
CF, LH = 15, 20
CHAR = 0.602 * CF
KW = re.compile(r"\b(fn|mut)\b")
NUM = re.compile(r"(?<![\w.])(\d+)(?![\w])")

# Kind key in screen_tasks.csv, card title, icon, task, contract rows, implementation rows.
# A row is (text, highlight, tag): ens = reference guarantee, bad = defective line, ok = behavior the contract
# should state, note = intended property kept elsewhere, slot = what the contract does not state.
CASES = [
    ("reference_under_specified", "Missing guarantee", "puzzle", "VeriCoding_VD0072_vericoded", [
        ("fn min_array(a: &[i32])", None, None),
        ("  -> (r: i32)", None, None),
        ("  requires a.len() > 0,", None, None),
        ("  ensures forall|i| … r <= a[i],", "ens", None),
        ("+ exists|i| … r == a[i]", "slot", "missing"),
    ], [
        ("  let mut min_val: i32 = a[0];", "ok", None),
        ("  while j < a.len() … {", None, None),
        ("    let old = min_val;", None, None),
        ("    let ai: i32 = a[j];", None, None),
        ("    if ai < old {", None, None),
        ("      min_val = ai;", "ok", "element of a"),
        ("    } …", None, None),
        ("  }", None, None),
        ("  min_val", None, None),
    ]),
    ("reference_vacuous", "Vacuous contract", "empty", "VerusBench_Misc_havoc_inline_post", [
        ("fn havoc_inline_post(", None, None),
        ("    v: &mut Vec<u32>, a: u32,", None, None),
        ("    b: bool)", None, None),
        ("  requires … old(v)[k] == 1,", None, None),
        ("    10 < a < 20, b == false,", None, None),
        ("", "slot", "no ensures clause"),
    ], [
        ("  let mut idx: usize = v.len();", None, None),
        ("  while (idx > 0) … {", None, None),
        ("    idx = idx - 1;", None, None),
        ("    v.set(idx, v[idx] + a);", None, None),
        ("  }", None, None),
        ("  proof {", "note", "intended post"),
        ("    assert(… v[k] == 1 + a);", "note", None),
        ("  }", None, None),
    ]),
    ("reference_encoding_bug", "Incorrect constraint", "bug", "VeriCoding_VA0637_vericoded", [
        ("fn solve(n: i8)", "bad", "−128 … 127"),
        ("  -> (result: bool)", None, None),
        ("  requires", None, "unsatisfiable"),
        ("    valid_input(n as int),", None, None),
        ("    // 1000 <= n <= 9999", "bad", None),
        ("  ensures", None, None),
        ("    result <==> is_good(…),", None, None),
    ], [
        ("  let m: i32 = n as i32;", None, None),
        ("  let d1 = m / 1000;", "bad", "always 0"),
        ("  let d2 = (m / 100) % 10;", None, None),
        ("  let d3 = (m / 10) % 10;", None, None),
        ("  let d4 = m % 10;", None, None),
        ("  (d1 == d2 && d2 == d3) ||", None, None),
        ("    (d2 == d3 && d3 == d4)", None, None),
    ]),
    ("encoding_limitation", "Inexpressible property", "lock", "VeriCoding_VT0332_vericoded", [
        ("fn numpy_log10(x: Vec<f32>)", None, None),
        ("  -> (result: Vec<f32>)", None, None),
        ("  requires x@.len() > 0,", None, None),
        ("  ensures", "ens", None),
        ("    result@.len() == x@.len(),", "ens", None),
        ("+ log10(x[i])", "slot", "not expressible"),
    ], [
        ("  let n: usize = x.len();", None, None),
        ("  let mut result = Vec::new();", None, None),
        ("  let mut i: usize = 0;", None, None),
        ("  while i < n … {", None, None),
        ("    let xi = x[i];", None, None),
        ("    result.push(xi);", "bad", "copies x[i]"),
        ("    i = i + 1;", None, None),
        ("  }", None, None),
        ("  result", None, None),
    ]),
]
BAR = {"ens": "#ffffcc", "bad": "#ffd6d6", "ok": "#d4f3dd", "note": "#fff1c7"}
TAG_COLOR = {"ok": "#1e7b3c", "note": "#8a5a00"}
CW, GAP, HDR, PAD = 316, 14, 40, 10
CODE_X = PAD + 4
N_SPEC = max(len(c[4]) for c in CASES)
N_IMPL = max(len(c[5]) for c in CASES)
SPEC_Y = HDR + 28
DIV_Y = SPEC_Y + N_SPEC * LH + 4
IMPL_Y = DIV_Y + 22
CH = IMPL_Y + N_IMPL * LH + 8
assert all(len(t) * CHAR < CW - CODE_X - PAD for c in CASES for rows in c[4:] for t, _, _ in rows)

RENDER_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:#fff}</style>
<script>window.DRAW_MATH_URL='';</script>
<script src="viewer-static.min.js"></script>
<style>
@font-face{font-family:"Comic Sans MS";src:url(fonts/ComicRelief-Regular.ttf);font-weight:400}
@font-face{font-family:"Comic Sans MS";src:url(fonts/ComicRelief-Bold.ttf);font-weight:700}
@font-face{font-family:"Menlo";src:local("DejaVu Sans Mono");font-weight:400}
@font-face{font-family:"Menlo";src:local("DejaVu Sans Mono Bold");font-weight:700}
</style>
</head><body>
<div id="g"></div>
<script>
fetch('figure.drawio').then(r=>r.text()).then(xml=>{
  const doc=mxUtils.parseXml(xml);
  let node=doc.documentElement;
  if(node.nodeName=='mxfile'){node=node.getElementsByTagName('diagram')[0].getElementsByTagName('mxGraphModel')[0];}
  const c=document.getElementById('g');
  const graph=new Graph(c); graph.setEnabled(false);
  new mxCodec(node.ownerDocument).decode(node, graph.getModel());
  graph.refresh();
  const svgRoot=graph.getSvg('#ffffff',1,0,false,null,true,null,null,null,null,null,null,null,null);
  const b=graph.getGraphBounds();
  const W=Math.ceil(b.width), H=Math.ceil(b.height);
  c.remove();
  document.body.appendChild(svgRoot);
  const st=document.createElement('style');
  st.textContent='@page{size:'+W+'px '+H+'px;margin:0} svg{display:block}';
  document.head.appendChild(st);
  document.title='DONE '+W+'x'+H;
});
</script></body></html>
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def kind_counts() -> dict[str, int]:
    tasks = read_csv(RESULTS / "screen_tasks.csv")
    union = next(r for r in read_csv(RESULTS / "screen_signals.csv") if r["signal"] == "selected")
    candidates = [t for t in tasks if t["selected"] == "True" and t["preliminary_label"] == "yes"]
    counts = Counter(t["category"] for t in candidates)
    assert set(counts) == {c[0] for c in CASES} and len(candidates) == int(union["candidate"])
    by_task = {t["task_id"]: t for t in tasks}
    for kind, _, _, task, _, _ in CASES:
        # Each example must be a currently selected candidate of its kind.
        assert by_task[task]["selected"] == "True" and by_task[task]["category"] == kind, task
    return counts


def check_sources() -> None:
    """The examples must come from reference copies whose hash matches the task catalog."""
    sources = {r["task_id"]: r for r in read_csv(RESULTS / "reference_sources.csv")}
    for _, _, _, task, _, _ in CASES:
        row = sources[task]
        copy = ROOT / row["available_copy"]
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == row["sha256"], task


cells: list[str] = []
_next = [2]


def new_id(prefix: str) -> str:
    _next[0] += 1
    return f"{prefix}-{_next[0]}"


def vertex(value: str, style: str, x, y, w, h, parent="1") -> str:
    cid = new_id("v")
    cells.append(
        f'<mxCell id="{cid}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
        f'vertex="1" parent="{parent}"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
        f'height="{h:.2f}" as="geometry"/></mxCell>')
    return cid


def image(name: str, x, y, w, h, parent) -> str:
    uri = "data:image/svg+xml," + base64.b64encode((ASSETS / f"{name}.svg").read_bytes()).decode()
    style = (f"shape=image;html=1;verticalLabelPosition=bottom;verticalAlign=top;aspect=fixed;"
             f"imageAspect=0;image={uri};")
    return vertex("", style, x, y, w, h, parent)


def text(value, x, y, w, h, size=18, parent="1", bold=False, align="left", valign="middle",
         color=INK, font=SANS, extra="") -> str:
    style = (f"text;html=1;whiteSpace=wrap;{font}fontSize={size};fontColor={color};align={align};"
             f"verticalAlign={valign};spacing=0;fontStyle={1 if bold else 0};{extra}")
    return vertex(value, style, x, y, w, h, parent)


def mono(s: str) -> str:
    return f'<span style="{MONO_CSS}">{html.escape(s, quote=False)}</span>'


def colorize(line: str) -> str:
    if line.lstrip().startswith("//"):
        return f'<span style="color:#7b8794;">{html.escape(line, quote=False)}</span>'
    out, pos = [], 0
    toks = sorted([(m.start(), m.end(), KW_BLUE) for m in KW.finditer(line)] +
                  [(m.start(), m.end(), NUM_GREEN) for m in NUM.finditer(line)])
    for a, b, col in toks:
        out.append(html.escape(line[pos:a], quote=False))
        out.append(f'<span style="color:{col};">{html.escape(line[a:b], quote=False)}</span>')
        pos = b
    out.append(html.escape(line[pos:], quote=False))
    return "".join(out)


def rows_block(c, rows, y0) -> None:
    for i, (line, hl, tag) in enumerate(rows):
        y = y0 + i * LH
        if hl in BAR:
            vertex("", f"rounded=0;html=1;fillColor={BAR[hl]};strokeColor=none;",
                   CODE_X - 3, y, CW - CODE_X - PAD + 6, LH, c)
        elif hl == "slot":
            vertex("", f"rounded=1;arcSize=30;html=1;fillColor=#fff5f5;strokeColor={SLOT_RED};strokeWidth=1.5;"
                       f"dashed=1;dashPattern=4 3;", CODE_X - 3, y + 1, CW - CODE_X - PAD + 6, LH - 2, c)
        if tag and line:
            text(tag, CW - PAD - 150, y - 2, 146, LH + 4, size=15, bold=True, align="right", parent=c,
                 color=TAG_COLOR.get(hl, "#b3261e"), font=HAND)
        elif tag:
            text(f"<i>{tag}</i>", CODE_X, y, CW - CODE_X - PAD, LH, size=15, align="center", parent=c,
                 color="#b3261e")
    div = f'<div style="{MONO_CSS}font-size:{CF}px;line-height:{LH}px;white-space:pre;">'
    vertex(div + "<br>".join(colorize(t) or "&nbsp;" for t, _, _ in rows) + "</div>",
           f"text;html=1;whiteSpace=nowrap;spacing=0;verticalAlign=middle;overflow=visible;fontSize={CF};"
           f"align=left;fontColor={INK};", CODE_X, y0, CW - CODE_X - PAD, len(rows) * LH, c)


def build(counts: dict[str, int]) -> str:
    for k, (kind, title, icon, task, spec, impl) in enumerate(CASES):
        x = k * (CW + GAP)
        c = vertex("", f"swimlane;rounded=1;arcSize=3;html=1;startSize={HDR};fillColor={HDR_BLUE};"
                       f"swimlaneFillColor=#ffffff;strokeColor={CARD_STROKE};strokeWidth=2;shadow=1;"
                       f"collapsible=0;recursiveResize=0;swimlaneLine=0;", x, 0, CW, CH)
        vertex("", "ellipse;html=1;fillColor=#ffffff;strokeColor=none;", 8, 4, 32, 32, c)
        image(icon, 12, 8, 24, 24, c)
        text(title, 48, 0, CW - 110, HDR, size=18, bold=True, parent=c, color="#ffffff",
             extra="whiteSpace=nowrap;")
        vertex(str(counts[kind]), f"rounded=1;arcSize=50;html=1;fillColor={DEFECT};strokeColor=#ffffff;"
                                  f"strokeWidth=1.5;{SANS}fontSize=16;fontStyle=1;fontColor=#ffffff;spacing=0;",
               CW - 52, 9, 42, 22, c)
        text(mono(task), PAD, HDR + 6, CW - 2 * PAD, 18, size=13.5, parent=c, color="#6e7781",
             extra="whiteSpace=nowrap;")
        vertex("", "rounded=0;html=1;fillColor=#f5f7fa;strokeColor=none;", 1, DIV_Y, CW - 2, CH - DIV_Y - 4, c)
        vertex("", "line;html=1;strokeWidth=1;strokeColor=#c9d4e2;", 1, DIV_Y - 1, CW - 2, 2, c)
        text("implementation", PAD, DIV_Y + 2, 140, 18, size=13, parent=c, color="#6e7781", font=HAND)
        rows_block(c, spec, SPEC_Y)
        rows_block(c, impl, IMPL_Y)
    return ('<mxfile host="VerusEval"><diagram id="reference_screendefects" name="rq4_reference_defects">'
            '<mxGraphModel dx="1400" dy="800" grid="1" gridSize="2" guides="1" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="0" pageScale="1" math="0" shadow="0"><root>'
            '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
            '</root></mxGraphModel></diagram></mxfile>')


def viewer() -> Path:
    if not VIEWER_CACHE.exists():
        raise FileNotFoundError("Missing bundled draw.io viewer; unpack the complete code package.")
    return VIEWER_CACHE


def export(drawio: Path, pdf: Path) -> None:
    """Render with the draw.io viewer in headless Chromium; the page size equals the diagram bounds."""
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        shutil.copy(viewer(), work / "viewer-static.min.js")
        shutil.copytree(ASSETS / "fonts", work / "fonts")
        shutil.copy(drawio, work / "figure.drawio")
        (work / "render.html").write_text(RENDER_HTML, encoding="utf-8")
        from playwright.sync_api import sync_playwright
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(work)))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/render.html")
                page.wait_for_function("document.title.startsWith('DONE ')")
                page.evaluate("document.fonts.ready")
                page.pdf(path=str(pdf), prefer_css_page_size=True, print_background=True)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def main() -> None:
    check_sources()
    counts = kind_counts()
    drawio = PAPER_FIG / "rq4_reference_defects.drawio"
    drawio.write_text(build(counts), encoding="utf-8")
    export(drawio, PAPER_FIG / "rq4_reference_defects.pdf")
    print(drawio, {title: counts[kind] for kind, title, *_ in CASES}, f"W={4 * CW + 3 * GAP} H={CH}")


if __name__ == "__main__":
    main()
