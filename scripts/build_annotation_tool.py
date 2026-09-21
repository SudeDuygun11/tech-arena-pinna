"""Splice an exported ear payload into the annotation-tool template.

A separate step because the published page must be fully self-contained --
external fetches are blocked -- so the mesh has to be inlined into the HTML.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--payload", default="ear_payload.json")
p.add_argument("--template", default="tools/annotation_tool.template.html")
p.add_argument("--out", default="tools/annotation_tool.html")
a = p.parse_args()

payload = Path(a.payload).read_text()
html = Path(a.template).read_text(encoding="utf-8")
assert "__PAYLOAD__" in html, "template has no __PAYLOAD__ placeholder"
Path(a.out).write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
print(f"wrote {a.out}  ({Path(a.out).stat().st_size / 1e6:.2f} MB)")
