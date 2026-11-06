"""Contiguous, density-equalising cartogram of HydroBASINS basins
(fast flow-based method, Gastner, Seguy & More 2018, PNAS 115:E2156).

Each basin keeps its neighbours, but is deformed so that its area is proportional to the urban population
NOT served by a wastewater treatment plant (P_non). Colour (in the plotting step) = non-served ratio
R = 1 - P_served / (P_served + P_non).

Method: rasterise the target density rho0 = P_non / area on a grid (ocean and the mean land density set equal),
solve the Poisson equation lap(phi) = rho_mean - rho0 once by FFT, and move every polygon vertex with the
velocity v = -grad(phi) / rho(t), rho(t) = (1 - t) rho0 + t rho_mean, from t = 0 to 1. The whole procedure is
repeated on the deformed map a few times to sharpen the area match.
Basins without non-served population (all served / no urban population / not covered) get a small floor
density so the flow stays regular; they shrink to slivers.

Usage: python 07_cartogram.py <level 3|4|country> [--rounds 4] [--grid 1024] [--floor 0.02] [--simplify 4000]
Reads svc/service_hybas<lev>.gpkg (05_service.py) or svc/service_country.gpkg (05b_service_country.py);
writes svc/cartogram_hybas<lev>.gpkg / svc/cartogram_country.gpkg (EPSG:8857 metres, deformed) and a JSON
accuracy report.
"""

import argparse, json, time
import numpy as np, geopandas as gpd, shapely
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("level", choices=["3", "4", "country"])
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--grid", type=int, default=1024, help="cells along the long axis")
ap.add_argument("--floor", type=float, default=0.02, help="floor density as a fraction of the mean land density")
ap.add_argument("--simplify", type=float, default=4000, help="pre-simplification tolerance in metres")
ap.add_argument("--blur", type=float, default=1.0, help="Gaussian blur of the density grid, in cells")
a = ap.parse_args()

NAME = "country" if a.level == "country" else f"hybas{a.level}"
u = gpd.read_file(f"svc/service_{NAME}.gpkg").to_crs(8857)
u = u[u.geometry.notna() & ~u.geometry.is_empty].reset_index(drop=True)
u["geometry"] = u.geometry.simplify(a.simplify).make_valid()
u = u[u.geometry.area > 0].reset_index(drop=True)
u["R"] = np.where((u.P_served + u.P_non) > 0, u.P_non / (u.P_served + u.P_non), np.nan)
val = u.P_non.where(u.status == "ok", 0.0).astype(float).values
geoms = u.geometry.values
area0 = shapely.area(geoms)
target_share = None


def one_round(geoms, val):
    area = shapely.area(geoms)
    land_mean = val.sum() / area.sum()
    dens = np.maximum(val / area, a.floor * land_mean)
    # grid with generous padding (ocean) around the current map
    x0, y0, x1, y1 = shapely.total_bounds(geoms)
    w, h = x1 - x0, y1 - y0
    pad = 0.5 * max(w, h)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    cell = max(x1 - x0, y1 - y0) / a.grid
    nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    # rasterise: polygon index of every cell centre
    cx = x0 + (np.arange(nx) + 0.5) * cell
    cy = y0 + (np.arange(ny) + 0.5) * cell
    XX, YY = np.meshgrid(cx, cy)
    pts = shapely.points(XX.ravel(), YY.ravel())
    # deformed rounds can self-intersect; newer GEOS raises on 'within' against invalid polygons,
    # so rasterise against a repaired copy (index order is preserved, vertices are not altered)
    tree = shapely.STRtree(shapely.make_valid(geoms))
    ip, ig = tree.query(pts, predicate="within")
    # mean density of everything (land at its density, ocean at the land mean)
    rho_mean = (dens * area).sum() / area.sum()
    rho0 = np.full(nx * ny, rho_mean)
    rho0[ip] = dens[ig]
    rho0 = rho0.reshape(ny, nx)
    if a.blur > 0:
        rho0 = ndimage.gaussian_filter(rho0, a.blur, mode="nearest")
    rho_bar = rho0.mean()
    # Poisson: lap(phi) = rho_bar - rho0  ->  phi_hat = (rho0 - rho_bar)_hat / k^2
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy)
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0
    phi_hat = np.fft.fft2(rho0 - rho_bar) / k2
    phi_hat[0, 0] = 0
    gx = np.real(np.fft.ifft2(1j * KX * phi_hat))
    gy = np.real(np.fft.ifft2(1j * KY * phi_hat))
    # integrate vertex positions (grid coordinates) from t = 0 to 1
    xy = shapely.get_coordinates(geoms)
    P = np.c_[(xy[:, 0] - x0) / cell - 0.5, (xy[:, 1] - y0) / cell - 0.5]  # (col, row)

    def vel(P, t):
        c = [P[:, 1], P[:, 0]]
        gxi = ndimage.map_coordinates(gx, c, order=1, mode="nearest")
        gyi = ndimage.map_coordinates(gy, c, order=1, mode="nearest")
        r0 = ndimage.map_coordinates(rho0, c, order=1, mode="nearest")
        rho = (1 - t) * r0 + t * rho_bar
        return -np.c_[gxi, gyi] / rho[:, None]

    t, steps = 0.0, 0
    while t < 1.0 - 1e-12:
        v1 = vel(P, t)
        dt = min(0.05, 0.25 / max(np.abs(v1).max(), 1e-9), 1.0 - t)
        Pm = P + 0.5 * dt * v1
        v2 = vel(Pm, t + 0.5 * dt)
        P = P + dt * v2
        t += dt
        steps += 1
    new = np.c_[x0 + (P[:, 0] + 0.5) * cell, y0 + (P[:, 1] + 0.5) * cell]
    return shapely.set_coordinates(geoms.copy(), new), steps, (nx, ny)


t0 = time.time()
target = val / val.sum()
hist = []
for r in range(a.rounds):
    geoms, steps, shape = one_round(geoms, val)
    ar = shapely.area(geoms)
    share = ar / ar.sum()
    ok = val > 0
    rel = np.abs(share[ok] - target[ok] * (ar[ok].sum() / ar.sum())) / (target[ok] * (ar[ok].sum() / ar.sum()))
    hist.append(
        dict(
            round=r + 1,
            steps=steps,
            grid=shape,
            median_rel_err=round(float(np.median(rel)), 4),
            p90_rel_err=round(float(np.quantile(rel, 0.9)), 4),
        )
    )
    print(hist[-1], f"{time.time() - t0:.0f} s", flush=True)

out = gpd.GeoDataFrame(u.drop(columns="geometry"), geometry=gpd.GeoSeries(geoms, crs=8857), crs=8857)
out["geometry"] = out.geometry.make_valid()
ar = out.geometry.area
okm = val > 0
out["area_share"] = ar / ar[okm].sum()
out["target_share"] = np.where(okm, val / val.sum(), np.nan)
inv = int((~shapely.is_valid(geoms)).sum())
rep = dict(
    level=a.level,
    rounds=hist,
    invalid_before_fix=inv,
    floor_fraction=a.floor,
    grid=a.grid,
    land_area_share_of_non_served=round(float(ar[okm].sum() / ar.sum()), 4),
)
out.to_file(f"svc/cartogram_{NAME}.gpkg", driver="GPKG")
json.dump(rep, open(f"svc/cartogram_{NAME}_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
