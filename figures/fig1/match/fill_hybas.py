#!/usr/bin/env python3
"""
fill_hybas.py: fill HYBAS_ID (HydroBASINS level 9) in national_wwtp_AU_JP_NZ_ZA.gpkg.

Use the same HydroBASINS version as the HeiGIT file (standard version, without lakes):
  Africa       hybas_af_lev09_v1c.shp   (South Africa)
  Asia         hybas_as_lev09_v1c.shp   (Japan)
  Australasia  hybas_au_lev09_v1c.shp   (Australia, New Zealand)
Shapefiles, or the zips they come in, both work.

Each plant gets the basin it lies in; plants on the coast just outside every basin
polygon get the nearest basin within 5 km. The file is updated in place; its 'columns'
layer is updated too.

Usage (one line in PowerShell):
  python fill_hybas.py --gpkg <path>\\national_wwtp_AU_JP_NZ_ZA.gpkg
      --hybas <path>\\hybas_af_lev09_v1c.shp <path>\\hybas_as_lev09_v1c.shp <path>\\hybas_au_lev09_v1c.shp
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


def read_basins(path: Path, bbox):
    src = f"/vsizip/{path.resolve()}" if path.suffix.lower() == ".zip" else str(path)
    if path.suffix.lower() == ".zip":
        import zipfile

        shp = [n for n in zipfile.ZipFile(path).namelist() if n.lower().endswith(".shp")]
        src = f"{src}/{shp[0]}"
    g = gpd.read_file(src, bbox=bbox, columns=["HYBAS_ID"])
    return g.to_crs(4326) if g.crs is not None else g.set_crs(4326)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpkg", required=True, type=Path)
    ap.add_argument("--hybas", required=True, type=Path, nargs="+")
    ap.add_argument("--layer", default="national_wwtp")
    a = ap.parse_args(argv)

    pts = gpd.read_file(a.gpkg, layer=a.layer)
    hyb = pd.Series(np.nan, index=pts.index)
    for f in a.hybas:
        b = read_basins(f, tuple(pts.total_bounds))
        if b.empty:
            continue
        todo = pts[hyb.isna()]
        j = gpd.sjoin(todo[["geometry"]], b, how="inner", predicate="within")
        j = j[~j.index.duplicated()]
        hyb.loc[j.index] = j["HYBAS_ID"].to_numpy()
        # coastal plants just outside the polygons: nearest basin within 5 km
        todo = pts[hyb.isna()]
        if len(todo):
            near = gpd.sjoin_nearest(todo[["geometry"]].to_crs(6933), b.to_crs(6933), how="inner", max_distance=5_000)
            near = near[~near.index.duplicated()]
            hyb.loc[near.index] = near["HYBAS_ID"].to_numpy()
        print(f"{f.name}: {int(hyb.notna().sum())} of {len(pts)} plants have a basin so far")
    pts["HYBAS_ID"] = hyb.astype(float)

    cols = pyogrio.read_dataframe(a.gpkg, layer="columns")
    m = cols["column"] == "HYBAS_ID"
    cols.loc[m, "filled"] = int(hyb.notna().sum())
    cols.loc[m, "description"] = "HydroBASINS level-9 basin of the plant (standard version, as in the HeiGIT file)."
    tmp = a.gpkg.with_name(a.gpkg.stem + ".tmp.gpkg")
    if tmp.exists():
        tmp.unlink()
    pts.to_file(tmp, layer=a.layer, driver="GPKG")
    pyogrio.write_dataframe(cols, tmp, layer="columns")
    os.replace(tmp, a.gpkg)
    print(pts.groupby("ISO_2")["HYBAS_ID"].count().to_dict(), "plants with HYBAS_ID; written to", a.gpkg)


if __name__ == "__main__":
    main()
