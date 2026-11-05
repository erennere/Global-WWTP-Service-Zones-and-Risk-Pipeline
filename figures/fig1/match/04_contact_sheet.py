"""Stack all rendered variants of one or more aggregation levels into a labelled overview PNG."""
import sys, glob, os
from PIL import Image, ImageDraw, ImageFont

out, levels = sys.argv[1], sys.argv[2:]
ORDER = ["base", "europe", "europe_latprofile", "spikes_europe", "main_with_spikes", "europe_with_spikes", "main_with_circles", "europe_with_circles"]
font = ImageFont.truetype("C:/Windows/Fonts/LiberationSans-Bold.ttf", 34)
rows = []
for lev in levels:
    for v in ORDER:
        f = f"{os.environ.get('FIGS', 'figs')}/{lev}/fig1_{lev}_{v}.png"
        if os.path.exists(f):
            rows.append((f"{lev}  ·  {v}", Image.open(f).convert("RGB")))
W = 1800
tiles = []
for lab, im in rows:
    im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
    t = Image.new("RGB", (W, im.height + 60), "white")
    ImageDraw.Draw(t).text((10, 10), lab, fill="#333333", font=font)
    t.paste(im, (0, 60)); tiles.append(t)
sheet = Image.new("RGB", (W, sum(t.height + 20 for t in tiles)), "#e9e9e9")
y = 0
for t in tiles:
    sheet.paste(t, (0, y)); y += t.height + 20
sheet.save(out, optimize=True)
print(out, sheet.size)
