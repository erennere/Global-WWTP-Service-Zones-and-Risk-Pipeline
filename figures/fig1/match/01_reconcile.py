"""Reconcile HeiGIT WWTP locations against the reference dataset by proximity.

Reference: the EU UWWTD in its 30 countries, HydroWASTE v1.0 everywhere else. Within each dataset, records
less than 500 m apart form one plant (single linkage, shortest links first, located at the members' centroid;
two records that HeiGIT treats as separate plants are never joined).

HeiGIT plants and reference plants are paired one-to-one by proximity: all pairs within RADIUS, accepted
shortest first. Every plant ends up in exactly one class:
  corrected  - HeiGIT plant paired with a reference plant; shift = great-circle distance between the two
  new        - HeiGIT plant without a reference plant within RADIUS
  missing    - reference plant without a HeiGIT plant within RADIUS
The ID columns (HW_WASTE_ID, uwwCode) are not used for pairing; the summary reports how often the proximity
pair is the plant named by the ID.

Usage: python 01_reconcile.py <data_dir> <out_dir> [radius_m ...]
"""
import sys, json
import numpy as np, pandas as pd, pycountry
from sklearn.neighbors import BallTree
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig1_config import MATCH_RADIUS_M, MIN_PLANTS, PATHS, TAG  # noqa: E402  (config.yaml, section figure1)

R_EARTH = 6371008.8
DATA, OUT = sys.argv[1], sys.argv[2]
RADII = [float(r) for r in sys.argv[3:]] or [MATCH_RADIUS_M]


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def greedy_match(a_lonlat, b_lonlat, radius, a_flag=None, b_flag=None):
    """One-to-one matching: all pairs within radius, accepted shortest-first. Returns (ia, ib, d).
    Pairs where both a_flag[ia] and b_flag[ib] are True are not allowed."""
    if len(a_lonlat) == 0 or len(b_lonlat) == 0:
        return np.array([], int), np.array([], int), np.array([])
    tree = BallTree(np.radians(b_lonlat[:, ::-1]), metric="haversine")
    ind, dist = tree.query_radius(np.radians(a_lonlat[:, ::-1]), r=radius / R_EARTH, return_distance=True)
    ia = np.repeat(np.arange(len(a_lonlat)), [len(x) for x in ind])
    if len(ia) == 0:
        return np.array([], int), np.array([], int), np.array([])
    ib = np.concatenate(ind); d = np.concatenate(dist) * R_EARTH
    if a_flag is not None:
        ok = ~(a_flag[ia] & b_flag[ib]); ia, ib, d = ia[ok], ib[ok], d[ok]
    o = np.argsort(d, kind="stable"); ia, ib, d = ia[o], ib[o], d[o]
    ua, ub = np.zeros(len(a_lonlat), bool), np.zeros(len(b_lonlat), bool)
    keep = []
    for k in range(len(d)):
        if not ua[ia[k]] and not ub[ib[k]]:
            ua[ia[k]] = ub[ib[k]] = True; keep.append(k)
    keep = np.array(keep, int)
    return ia[keep], ib[keep], d[keep]


def group_records(lon, lat, link, r=500.0):
    """Records closer than r form one plant (single linkage, shortest links first). Two groups that are
    ID-linked to different HeiGIT points are never joined. Returns consecutive group labels."""
    X = np.radians(np.c_[lat, lon])
    ind, dist = BallTree(X, metric="haversine").query_radius(X, r / R_EARTH, return_distance=True)
    i = np.repeat(np.arange(len(X)), [len(x) for x in ind]); j = np.concatenate(ind); d = np.concatenate(dist)
    keep = i < j; i, j, d = i[keep], j[keep], d[keep]
    o = np.argsort(d, kind="stable")
    par = np.arange(len(X)); lk = np.asarray(link, dtype=float).copy()      # NaN = not linked

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in zip(i[o], j[o]):
        ra, rb = root(a), root(b)
        if ra == rb or (lk[ra] == lk[ra] and lk[rb] == lk[rb] and lk[ra] != lk[rb]):
            continue
        par[rb] = ra
        if lk[ra] != lk[ra]:
            lk[ra] = lk[rb]
    return np.unique([root(x) for x in range(len(X))], return_inverse=True)[1]


def plants_of(rec, grp, linked):
    """One row per plant: centroid of its records, most common country, representative record (linked if any)."""
    g = pd.DataFrame({"grp": grp, "lon": rec["lon"].values, "lat": rec["lat"].values, "ISO_2": rec["ISO_2"].values,
                      "rec": np.arange(len(rec)), "linked": linked})
    lo, la = np.radians(g.lon), np.radians(g.lat)
    g["x"], g["y"], g["z"] = np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)
    p = g.groupby("grp").agg(n=("rec", "size"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean"),
                             lon1=("lon", "first"), lat1=("lat", "first"),
                             ISO_2=("ISO_2", lambda v: v.value_counts().index[0]), linked=("linked", "any"))
    p["lon"] = np.where(p.n > 1, np.degrees(np.arctan2(p.y, p.x)), p.lon1)
    p["lat"] = np.where(p.n > 1, np.degrees(np.arctan2(p.z, np.hypot(p.x, p.y))), p.lat1)
    rep = g.sort_values(["grp", "linked", "rec"], ascending=[True, False, True]).groupby("grp").rec.first()
    p["rep"] = rep
    return p[["lon", "lat", "ISO_2", "linked", "rep", "n"]].reset_index(drop=True)


def a3_to_a2(a3):
    c = pycountry.countries.get(alpha_3=a3)
    return c.alpha_2 if c else {"XKX": "XK"}.get(a3)


# ---------------------------------------------------------------- load
H = pd.read_csv(f"{DATA}/heigit_points.csv", dtype={"uwwCode": str, "ISO_2": str}, keep_default_na=False, na_values=[""])
H["HW_WASTE_ID"] = H["HW_WASTE_ID"].astype("Int64")
U = pd.read_csv(f"{DATA}/uwwtd_points.csv", keep_default_na=False, na_values=[""])
W = pd.read_csv(f"{DATA}/HydroWASTE_v10.csv", encoding="latin-1", keep_default_na=False, na_values=[""])
W = W.rename(columns={"LON_WWTP": "lon", "LAT_WWTP": "lat"})
W["ISO_2"] = W["CNTRY_ISO"].map(a3_to_a2)
U["ISO_2"] = U["ISO_2_UWW"].replace({"EL": "GR"})
H.loc[H["ISO_2"] == "EL", "ISO_2"] = "GR"
# Namibia's ISO-2 "NA" must survive CSV parsing
assert W["ISO_2"].isna().sum() == 0, W.loc[W.ISO_2.isna(), "CNTRY_ISO"].unique()

# ---------------------------------------------------------------- reference plants
UC = set(U["ISO_2"])                                   # the 30 UWWTD countries
W = W[~W["ISO_2"].isin(UC)].reset_index(drop=True)     # there the UWWTD is the reference
fid_of_hw = dict(zip(H.HW_WASTE_ID.dropna().astype(int), H.fid[H.HW_WASTE_ID.notna()]))
fid_of_uw = dict(zip(H.uwwCode.dropna(), H.fid[H.uwwCode.notna()]))
W["grp"] = group_records(W.lon.values, W.lat.values, W.WASTE_ID.map(fid_of_hw).values)
U["grp"] = group_records(U.lon.values, U.lat.values, U.uwwCode.map(fid_of_uw).values)
Wg = plants_of(W, W.grp.values, W.WASTE_ID.isin(fid_of_hw).values)
Ug = plants_of(U, U.grp.values, U.uwwCode.isin(fid_of_uw).values)
Wg["WASTE_ID"] = W.WASTE_ID.values[Wg.rep]; Wg["uwwCode"] = ""
Ug["uwwCode"] = U.uwwCode.values[Ug.rep]; Ug["WASTE_ID"] = -1
ref0 = pd.concat([Ug.assign(ref_src="UWWTD"), Wg.assign(ref_src="HW")], ignore_index=True)
ref0 = ref0[["lon", "lat", "ISO_2", "ref_src", "WASTE_ID", "uwwCode", "n"]].rename(
    columns={"WASTE_ID": "HW_WASTE_ID", "n": "n_records"})
# record -> reference plant
rec_ref = pd.concat([pd.DataFrame({"src": "UWWTD", "row": U.index, "WASTE_ID": -1, "uwwCode": U.uwwCode,
                                   "lon": U.lon, "lat": U.lat, "ref_id": U.grp}),
                     pd.DataFrame({"src": "HW", "row": W.index, "WASTE_ID": W.WASTE_ID, "uwwCode": "",
                                   "lon": W.lon, "lat": W.lat, "ref_id": W.grp + len(Ug)})], ignore_index=True)
# the plant each HeiGIT point names by ID (only to report agreement with the proximity pairs)
uwg = dict(zip(U.uwwCode, U.grp)); hwg = dict(zip(W.WASTE_ID, W.grp + len(Ug)))
id_ref = np.array([uwg[c] if isinstance(c, str) and c in uwg else (hwg.get(int(w), -1) if pd.notna(w) else -1)
                   for c, w in zip(H.uwwCode, H.HW_WASTE_ID)])
print(f"HydroWASTE plants (outside the UWWTD countries): {len(Wg):,}  UWWTD plants: {len(Ug):,}")

summary = {}
for RADIUS in RADII:
    tag = f"r{int(RADIUS)}"
    ref = ref0.copy()
    ia, ib, d = greedy_match(H[["lon", "lat"]].values, ref[["lon", "lat"]].values, RADIUS)
    H["cls"] = "new"; H["ref_id"] = -1; H["shift_m"] = np.nan; H["link"] = ""
    H.loc[ia, "cls"] = "corrected"; H.loc[ia, "ref_id"] = ib; H.loc[ia, "shift_m"] = d
    H.loc[ia, "link"] = ref.ref_src.values[ib]
    ref["cls"] = "missing"; ref.loc[ib, "cls"] = "corrected"
    has_id = id_ref >= 0
    agree = float((H.ref_id.values[has_id] == id_ref[has_id]).mean())

    # ------------------------------------------------------------ scope & coverage
    ref_countries = set(ref.ISO_2)
    hei_countries = set(H.ISO_2.dropna())
    uncovered = sorted(ref_countries - hei_countries)
    H["in_scope"] = H.ISO_2.isin(ref_countries)
    ref["covered"] = ~ref.ISO_2.isin(uncovered)

    # ------------------------------------------------------------ outputs
    hp = H[["fid", "lon", "lat", "ISO_2", "HYBAS_ID", "SOURCE_DS", "link", "cls", "shift_m", "ref_id", "in_scope"]].copy()
    hp["kind"] = "heigit"
    mp = ref.loc[ref.cls == "missing", ["lon", "lat", "ISO_2", "ref_src", "covered", "HW_WASTE_ID", "uwwCode"]].copy()
    mp["kind"] = "reference"; mp["cls"] = "missing"
    hp.to_csv(f"{OUT}/plants_heigit_{tag}.csv", index=False)
    mp.to_csv(f"{OUT}/plants_missing_{tag}.csv", index=False)
    ref.to_csv(f"{OUT}/reference_plants_{tag}.csv", index_label="ref_id")
    rec_ref.to_csv(f"{OUT}/reference_records_{tag}.csv", index=False)

    sc = H[H.in_scope]
    s = dict(
        radius_m=RADIUS,
        n_heigit=len(H), n_heigit_in_scope=int(len(sc)),
        n_hydrowaste=len(Wg), n_uwwtd=len(Ug), n_reference_plants=len(ref),
        ref_by_src=ref.ref_src.value_counts().to_dict(),
        heigit_cls=H.cls.value_counts().to_dict(),
        heigit_cls_in_scope=sc.cls.value_counts().to_dict(),
        ref_cls=ref.cls.value_counts().to_dict(),
        missing_in_covered=int(((ref.cls == "missing") & ref.covered).sum()),
        missing_in_uncovered=int(((ref.cls == "missing") & ~ref.covered).sum()),
        uncovered_countries=uncovered,
        heigit_with_id=int(has_id.sum()), proximity_pair_is_id_plant=round(agree, 4),
        corrected_shift_quantiles_m={str(q): round(float(np.nanquantile(H.loc[H.cls == "corrected", "shift_m"], q)), 1)
                                     for q in (.1, .25, .5, .75, .9, .99)},
    )
    summary[tag] = s
    print(json.dumps(s, indent=1, default=str))

json.dump(summary, open(f"{OUT}/reconcile_summary.json", "w"), indent=1, default=str)
