"""Figure 1, combined: stack the map 1 row (a, b; 03_plot.py) over the map 2 row (c, d; 09_plot_pair.py
--letters cd) on one 180 mm page. The PDF stays vector (pages merged with pypdf); the PNG is stacked from the
two 600 dpi PNGs.

Usage: python 10_compose.py <out_stem> <map1_stem> <map2_stem> [--gap 2.0]
  reads <map1_stem>.pdf/.png and <map2_stem>.pdf/.png, writes <out_stem>.pdf and <out_stem>.png
Requires pypdf (pip install pypdf).
"""
import argparse
import pypdf
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from PIL import Image

# the map rows are dense vector PDFs written by matplotlib here; pypdf >= 6 caps content-stream parsing at
# 2 M tokens and 75 MB of decoded output per stream as a guard against hostile files; these trusted local
# files exceed both at basin level
pypdf.overwrite_configuration(stream_parsing_work_maximum_count=200_000_000,
                              array_based_stream_maximum_output_length=2_000_000_000,
                              zlib_maximum_output_length=2_000_000_000,
                              stream_decoding_work_maximum_length=4_000_000_000)

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("top"); ap.add_argument("bottom")
ap.add_argument("--gap", type=float, default=2.0, help="space between the rows in mm")
a = ap.parse_args()
PT = 72 / 25.4

# ---- PDF: one page, map 2 at the bottom, map 1 above it
top, bot = PdfReader(a.top + ".pdf").pages[0], PdfReader(a.bottom + ".pdf").pages[0]
wt, ht = float(top.mediabox.width), float(top.mediabox.height)
wb, hb = float(bot.mediabox.width), float(bot.mediabox.height)
W, gap = max(wt, wb), a.gap * PT
page = PageObject.create_blank_page(width=W, height=ht + gap + hb)
page.merge_transformed_page(bot, Transformation().translate((W - wb) / 2, 0))
page.merge_transformed_page(top, Transformation().translate((W - wt) / 2, hb + gap))
w = PdfWriter(); w.add_page(page)
with open(a.out + ".pdf", "wb") as f:
    w.write(f)

# ---- PNG: the same stack at 600 dpi
it, ib = Image.open(a.top + ".png").convert("RGB"), Image.open(a.bottom + ".png").convert("RGB")
gp = round(a.gap / 25.4 * 600)
im = Image.new("RGB", (max(it.width, ib.width), it.height + gp + ib.height), "white")
im.paste(it, ((im.width - it.width) // 2, 0)); im.paste(ib, ((im.width - ib.width) // 2, it.height + gp))
im.save(a.out + ".png", dpi=(600, 600), optimize=True)
print("wrote", a.out, f"({W / PT:.0f} x {(ht + gap + hb) / PT:.1f} mm)")
