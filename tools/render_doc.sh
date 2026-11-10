#!/bin/bash
# 把 PIPER_抓取全流程.md 渲染成 PDF (markdown -> 内嵌 CSS 的 HTML -> soffice PDF)
set -e
cd /home/po/vision
SRC="${1:-PIPER_抓取全流程.md}"
BASE="${SRC%.md}"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY uv run --no-project --with markdown python - "$SRC" "$BASE" <<'PY'
import sys, markdown
src, base = sys.argv[1], sys.argv[2]
md = open(src, encoding="utf-8").read()
html = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
css = """
body{font-family:'Noto Sans CJK SC',sans-serif;font-size:10.5pt;line-height:1.55;margin:2mm 4mm;color:#111}
h1{font-size:19pt;color:#124;border-bottom:2px solid #2aa;padding-bottom:4px}
h2{font-size:14pt;color:#135;margin-top:14px;border-left:5px solid #2aa;padding-left:8px}
h3{font-size:11.5pt;color:#246;margin-top:10px}
code{font-family:'Maple Mono NF CN',monospace;background:#f2f5f7;padding:1px 3px;font-size:9pt}
pre{background:#f5f8fa;border:1px solid #cfd8dc;border-radius:3px;padding:6px;font-size:8.4pt;white-space:pre-wrap}
pre code{background:none;font-size:8.4pt}
table{border-collapse:collapse;font-size:9pt}
td,th{border:1px solid #b0bec5;padding:3px 6px}
th{background:#e3ecf2}
ul{list-style-type:disc;margin-left:16px}
blockquote{border-left:3px solid #7cc;background:#f3fbfb;margin:6px 0;padding:4px 8px}
@page{size:A4;margin:12mm}
"""
open(base + ".html", "w", encoding="utf-8").write(
    f"<!doctype html><meta charset='utf-8'><style>{css}</style>{html}")
print("html ok")
PY
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY soffice --headless --norestore --convert-to pdf --outdir /home/po/vision "/home/po/vision/$BASE.html" >/dev/null 2>&1
pdfinfo "/home/po/vision/$BASE.pdf" | grep Pages
