"""Figure 1: HeiGIT plants against reference plants (map 1) and the not-served cartogram (map 2).

Run with `snakemake fig1`. Settings and folders: wwtp/config.yaml, section `figure1`.
Inputs, prepared once (see figures/fig1/README.md):
    <match_dir>/data:     heigit_points.csv, uwwtd_points.csv, HydroWASTE_v10.csv, ne_50m_admin_0_countries.geojson
                          (00_fix_utm.py and 00b_national_reference.py are run by hand before, they edit these files)
    <cartogram_dir>/data: non_served_points.csv, heigit_points.csv, heigit_pop.csv, ne_50m_admin_0_countries.geojson
    hydrobasins_dir, heigit_attrs
The plotting steps run the folder's render_*.sh scripts (they fan out over variants); their figures are listed
by stamp files, the data steps by their real outputs.
"""
FIG1 = wf.config("figure1")
FIG1_DIR = os.path.join(os.path.dirname(workflow.basedir), "figures", "fig1")
M, C, K = (FIG1["paths"][k] for k in ("match_dir", "cartogram_dir", "combined_dir"))
HB, ATTRS = FIG1["paths"]["hydrobasins_dir"], FIG1["paths"]["heigit_attrs"]
TAG = f"r{int(FIG1['match_radius_m'])}"
FIG1_RES = {"mem_mb": 32768, "runtime": 720}


def fig1_script(path):
    return os.path.join(FIG1_DIR, path)


rule fig1_reconcile:
    """Pair HeiGIT and reference plants one-to-one within figure1.match_radius_m."""
    input: [f"{M}/data/{f}" for f in ("heigit_points.csv", "uwwtd_points.csv", "HydroWASTE_v10.csv")]
    output: [f"{M}/agg/{f}_{TAG}.csv" for f in ("plants_heigit", "plants_missing", "reference_plants", "reference_records")]
    params: script=fig1_script("match/01_reconcile.py"), radius=int(FIG1["match_radius_m"])
    log: wf.log("fig1_reconcile")
    resources: **FIG1_RES
    shell: "cd {M} && {PY} {params.script} data agg {params.radius} > {log} 2>&1"


rule fig1_aggregate:
    """Plants and reference plants per country (map 1 units)."""
    input: rules.fig1_reconcile.output, f"{M}/data/ne_50m_admin_0_countries.geojson"
    output: f"{M}/agg/units_country_{TAG}.gpkg", f"{M}/agg/plants_with_units_{TAG}.csv", f"{M}/agg/reference_with_units_{TAG}.csv"
    params: script=fig1_script("match/02_aggregate.py")
    log: wf.log("fig1_aggregate")
    resources: **FIG1_RES
    shell: "cd {M} && {PY} {params.script} data agg agg {TAG} > {log} 2>&1"


rule fig1_aggregate_basins:
    """The same per HydroBASINS level 3 and 4 basin."""
    input: rules.fig1_aggregate.output, HB
    output: f"{M}/agg/units_hybas3_{TAG}.gpkg", f"{M}/agg/units_hybas4_{TAG}.gpkg"
    params: script=fig1_script("match/02b_aggregate_basins.py")
    log: wf.log("fig1_aggregate_basins")
    resources: **FIG1_RES
    shell: "cd {M} && {PY} {params.script} {HB} agg {TAG} > {log} 2>&1"


rule fig1_maps:
    """Map 1 in all its variants, per level, and the contact sheets (render_all.sh)."""
    input: rules.fig1_aggregate.output, rules.fig1_aggregate_basins.output
    output: touch(wf.stamp("fig1_maps"))
    params: script=fig1_script("match/render_all.sh")
    log: wf.log("fig1_maps")
    resources: **FIG1_RES
    shell: "bash {params.script} {TAG} > {log} 2>&1"


rule fig1_tanks:
    """Panel 1d: round-tank share and plant surroundings (render_tanks.sh)."""
    input: rules.fig1_aggregate.output, rules.fig1_aggregate_basins.output, ATTRS
    output: touch(wf.stamp("fig1_tanks"))
    params: script=fig1_script("match/render_tanks.sh")
    log: wf.log("fig1_tanks")
    resources: **FIG1_RES
    shell: "ATTRS={ATTRS} HB={HB} bash {params.script} {TAG} > {log} 2>&1"


rule fig1_service:
    """Served share of the urban population per basin (05) and per country (05b; adds to service_report.json)."""
    input:
        rules.fig1_aggregate.output, rules.fig1_aggregate_basins.output,
        [f"{C}/data/{f}" for f in ("non_served_points.csv", "heigit_points.csv", "heigit_pop.csv",
                                   "ne_50m_admin_0_countries.geojson")],
    output: [f"{C}/svc/service_{n}.gpkg" for n in ("hybas3", "hybas4", "country")], f"{C}/svc/service_report.json"
    params: basins=fig1_script("cartogram/05_service.py"), countries=fig1_script("cartogram/05b_service_country.py")
    log: wf.log("fig1_service")
    resources: **FIG1_RES
    shell:
        "cd {C} && {PY} {params.basins} data {HB} {M}/agg svc {TAG} > {log} 2>&1"
        " && {PY} {params.countries} data {M}/agg svc {TAG} >> {log} 2>&1"


rule fig1_cartogram:
    """Area-by-population cartogram of one unit level (07_cartogram.py)."""
    input: f"{C}/svc/service_{{name}}.gpkg"
    output: f"{C}/svc/cartogram_{{name}}.gpkg"
    wildcard_constraints: name="hybas3|hybas4|country"
    params:
        script=fig1_script("cartogram/07_cartogram.py"),
        args=lambda w: {"hybas3": "3 --rounds 8", "hybas4": "4 --rounds 10", "country": "country --rounds 8"}[w.name],
    log: wf.log("fig1_cartogram_{name}")
    resources: **FIG1_RES
    shell: "cd {C} && {PY} {params.script} {params.args} --grid 1536 > {log} 2>&1"


rule fig1_cartogram_figures:
    """Map 2: cartograms, pairs and the above-50 % maps (render_cartograms/pairs/above50.sh)."""
    input: expand(f"{C}/svc/cartogram_{{name}}.gpkg", name=["hybas3", "hybas4", "country"])
    output: touch(wf.stamp("fig1_cartogram_figures"))
    params: scripts=[fig1_script(f"cartogram/render_{s}.sh") for s in ("cartograms", "pairs", "above50")]
    log: wf.log("fig1_cartogram_figures")
    resources: **FIG1_RES
    shell: "for s in {params.scripts}; do bash $s >> {log} 2>&1 || exit 1; done"


rule fig1_combined:
    """Map 1 over map 2, one figure per level and version (render_combined.sh)."""
    input: wf.stamp("fig1_maps"), wf.stamp("fig1_cartogram_figures"), ATTRS
    output: touch(wf.stamp("fig1_combined"))
    params: script=fig1_script("compose/render_combined.sh")
    log: wf.log("fig1_combined")
    resources: **FIG1_RES
    shell: "ATTRS={ATTRS} HB={HB} TAG={TAG} bash {params.script} > {log} 2>&1"


rule fig1:
    input: wf.stamp("fig1_maps"), wf.stamp("fig1_tanks"), wf.stamp("fig1_cartogram_figures"), wf.stamp("fig1_combined")
