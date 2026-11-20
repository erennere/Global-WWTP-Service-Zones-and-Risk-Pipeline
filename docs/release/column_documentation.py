"""The column documentation (PDF) of a release file made by wwtp.release_export.

Usage: python docs/release/column_documentation.py <release.gpkg> <size_measure> <output.pdf>

The text follows the documentation of the first release (HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg, 17 Sep 2026),
updated for plant list v3; the numbers (plants, sources, the size and radius settings) are read from the file
and from config.yaml (create_voronoi of the size measure). Rendered with headless Chrome (playwright; Chrome at
$CHROME, default /usr/bin/google-chrome); the HTML is written next to the PDF.
"""

from __future__ import annotations

import datetime
import html
import os
import re
import sys

import numpy as np
import pyogrio

from wwtp.starter import load_config

SOURCES = {  # SOURCE_DS: dataset, what SOURCE_ID holds
    "UWW": ("EU Urban Waste Water Treatment Directive (UWWTD) reporting data, plant table "
            "(and the corrected EU file of plant list v3)", "uwwCode"),
    "HW": ("HydroWASTE v1.0 (Ehalt Macedo et al., 2022), locations corrected by a detection model",
           "HydroWASTE WASTE_ID"),
    "US_CWNS": ("US EPA Clean Watersheds Needs Survey (CWNS 2022, else 2016), matched to HydroWASTE",
                "CWNS facility ID (11 digits); empty for plants without one"),
    "CA_WSER": ("Canada, Wastewater Systems Effluent Regulations (WSER) reporting", "WSER facility ID"),
    "AU_GA": ("Australia, Geoscience Australia national wastewater treatment plants", "objectid"),
    "JP_P22": ("Japan, MLIT National Land Numerical Information P22 (sewage treatment plants)", "TreeCode"),
    "OSM": ("OpenStreetMap (New Zealand plants)", "element, type-id: way-997826589 is "
            "openstreetmap.org/way/997826589"),
    "TH_GEOAI": ("geoAI water dataset of treatment ponds, Thailand and neighbours (Thailand_500m_merged)",
                 "none: the file's FID was not kept by the merge and is not unique"),
    "DETECTED": ("New plants found by the location model (not in HydroWASTE)", "the detection's image"),
}  # fmt: skip

MEASURES = {
    "reported": "the reported size where the plant has one, otherwise the tank estimate",
    "footprint": "the tank estimate for every plant (no reported number enters)",
    "combined_25": "for a plant with both a reported size and a tank estimate: reported<sup>0.75</sup> &times; "
    "estimate<sup>0.25</sup> (a weighted mean in log terms, the weight chosen on real sewer catchments in the US, "
    "GB, the Netherlands and Ireland); a plant with only one of the two uses that one",
    "combined_25_spread": "the sizes of combined_25 (reported<sup>0.75</sup> &times; estimate<sup>0.25</sup> for a "
    "plant with both, otherwise the one there is), with the radius rule that also follows the settled area",
} | {
    f"combined_{w}": f"for a plant with both: reported<sup>{1 - int(w) / 100:g}</sup> &times; "
    f"estimate<sup>{int(w) / 100:g}</sup>; otherwise the one there is"
    for w in ("10", "20", "30", "40", "50", "60", "70", "75", "80", "90")
}
SCALE_LIMITS = {"footprint": "0.2 and 5"}  # the others: 0.1 and 10 (config.yaml, size_scale)


def code(name):
    return f"<code>{html.escape(name)}</code>"


def table(rows, header=("Column", "Description")):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(path, measure):
    name = os.path.basename(path)
    df = pyogrio.read_dataframe(path, layer="WWTP_SERVICE_AREAS", read_geometry=False,
                                columns=["ISO_2", "HYBAS_ID", "SOURCE_DS", "SOURCE_ID", "2024_zonal_sum"])  # fmt: skip
    fields = pyogrio.read_info(path, layer="WWTP_SERVICE_AREAS")["fields"]
    years = sorted(int(f[:4]) for f in fields if re.fullmatch(r"\d{4}_zonal_sum", f) and f != "2014_zonal_sum")
    span = f"{years[0]}&ndash;{years[-1]}"
    projected = (" The years after 2024 are WorldPop's projections, for the plants and service areas of this file."
                 if years[-1] > 2024 else "")  # fmt: skip
    counts = df["SOURCE_DS"].value_counts()
    with_id = df.groupby("SOURCE_DS")["SOURCE_ID"].count()
    cfg = load_config("create_voronoi", size_measure=measure)
    rc = cfg["radius_calibration"][measure]
    rule = rc["by_connection"]
    spread_words, spread_term = "", ""
    if rule["spread_slope"] != 0:  # the radius also grows with how spread out the country's people live
        spread_words = (" and with its settled area S<sub>c</sub>, the km&sup2; of 1 km cells with people per "
                        "1,000 people (WorldPop)")  # fmt: skip
        spread_term = f" + {rule['spread_slope']:.3f} &middot; log S<sub>c</sub>"
    f = cfg["footprint_to_people"]
    n, people = len(df), np.nansum(df["2024_zonal_sum"].to_numpy(dtype=float))
    today = datetime.date.today().isoformat()

    source_rows = [
        (code(k), html.escape(SOURCES[k][0]), html.escape(SOURCES[k][1]), f"{int(counts[k]):,}")
        for k in SOURCES if k in counts.index
    ]  # fmt: skip
    ident = table([
        (code("WASTE_ID"), f"New sequential ID (0, 1, 2, &hellip;) of each row of this file, in its sort order "
         f"(country, then {code('HYBAS_ID')}, then source). Not an ID of any source dataset."),
        (code("SOURCE_DS"), "The dataset the plant's record comes from (table below). Where a plant is in "
         "several, the UWWTD comes first, then the national registers, the US file, HydroWASTE, Canada's WSER, "
         "the Thailand file and the detections. Every plant has one."),
        (code("SOURCE_ID"), "<b>New.</b> The plant's ID in that dataset (table below), so that every plant can be "
         "traced to its source record."),
        (code("HW_WASTE_ID"), f"The plant's HydroWASTE ID ({code('WASTE_ID')} in the raw HydroWASTE dataset) "
         f"wherever one is known: for HydroWASTE records and for US plants the US file matched to HydroWASTE. "
         f"A cross-reference, also for plants whose {code('SOURCE_DS')} is another dataset."),
    ])  # fmt: skip
    sources_table = table(source_rows, header=("SOURCE_DS", "Dataset", "SOURCE_ID", "Plants"))

    uww = table([
        (code("ISO_2"), f"Two-letter (ISO 3166-1 alpha-2) country code; companion to {code('CNTRY_ISO')}'s "
         "three-letter code."),
        (code("epsg"), "EPSG code of the coordinate reference system associated with the row before it was "
         "reprojected into this file."),
        (code("uwwState"), "Status of the UWWTP (its reported state under the Directive)."),
        (code("repCode"), "Reporting ID for the record, as assigned by the reporting country."),
        (code("aggCode"), "ID of the agglomeration (served urban area) that this plant / collecting system serves."),
        (code("uwwCode"), "Unique ID of the wastewater treatment plant / collecting system (primary key in the "
         "UWWTD dataset)."),
        (code("uwwName"), "Name of the wastewater treatment plant / collecting system."),
        (code("uwwCapacity"), "Organic design capacity of the plant, in population equivalent (p.e.)."),
        (code("uwwPrimaryTreatment"), "Whether the plant provides primary treatment (boolean)."),
        (code("uwwSecondaryTreatment"), "Whether the plant provides secondary treatment (boolean)."),
        (code("uwwOtherTreatment"), "Whether the plant provides more stringent (\"other\") treatment beyond "
         "secondary (boolean)."),
    ])  # fmt: skip
    hw = table([
        (code("HYBAS_ID"), "ID of the HydroBASINS level-6 basin the plant's service area is confined to, linking it "
         "into the HydroSHEDS basin hierarchy."),
        (code("HW_WASTE_ID"), "The plant's HydroWASTE ID (see Identifiers section)."),
        (code("SOURCE"), "Name of the national/regional source dataset the HydroWASTE record was compiled from."),
        (code("ORG_ID"), "The plant's original ID as assigned by that source dataset/agency."),
        (code("WWTP_NAME"), "Name of the treatment plant, as given by HydroWASTE."),
        (code("COUNTRY"), "Country name."),
        (code("CNTRY_ISO"), "Three-letter (ISO 3166-1 alpha-3) country code."),
        (code("QUAL_LOC"), "Quality/confidence flag for the plant's geolocation accuracy."),
        (code("STATUS"), "Reported operational status of the plant (e.g. active/inactive)."),
        (code("POP_SERVED"), "Population served by the plant, as HydroWASTE records it; for Japan's register "
         "plants the treated population of MLIT P22."),
        (code("QUAL_POP"), "Quality/confidence flag for the population-served estimate (reported vs. "
         "modeled/estimated); value 4 (people within 11 km of the plant) is not used as a plant size."),
        (code("LEVEL"), "Level of wastewater treatment applied (e.g. advanced, secondary, primary, none, or unknown)."),
        (code("HYRIV_ID"), "ID of the HydroRIVERS river reach the plant's effluent discharges into."),
        (code("RIVER_DIS"), "Mean annual discharge (m&sup3;/s) of that river at the discharge point."),
    ])  # fmt: skip
    seg = table([
        (code("num_detection_circle"), "Number of circular tank/basin detections found for this plant."),
        (code("num_detection_rect"), "Number of rectangular tank/basin detections found for this plant."),
        (code("num_detections"), "Total number of detections (circular + rectangular) for this plant."),
        (code("diameters"), "Diameter(s) of the detected circular tanks (m)."),
        (code("round_area"), "Total area of the detected circular tanks, &Sigma; &pi;(d/2)&sup2; (m&sup2;)."),
        (code("rect_area"), "Total area of the detected rectangular tanks (m&sup2;)."),
        (code("total_area"), "Combined footprint of all detected tanks, round + rectangular (m&sup2;): the tank "
         "area F the plant size is estimated from."),
    ])  # fmt: skip
    ann = table([
        (code("category_number"), "Site category the image-to-text model assigned from the plant's satellite image "
         "with OpenStreetMap labels: 1 agricultural, 2 industrial (general, incl. food processing), 4 industrial "
         "(heavy industry, mining, power), 5 residential (dense urban), 6 mixed-use (urban/suburban), 7 "
         "rural/small community, 8 unclear. Categories 2, 4 and 6 count as industrial in the analyses."),
        (code("category_name"), "The model's label for that category, as it wrote it (in a few cases it does not "
         "match the number; the number is what the analyses use)."),
        (code("justification"), "The model's free-text rationale for the category."),
    ])  # fmt: skip
    limits = SCALE_LIMITS.get(measure, "0.1 and 10")
    geo = table([
        (code("basin_area"), "Area of the HydroBASINS level-6 basin the plant's service area is confined to "
         "(m&sup2;). Not used for the service radius in this file."),
        (code("weights"), f"The plant's weight in the weighted Voronoi diagram, &radic;P (P: plant size in people, "
         f"below), normalised across the plants of its basin ({code('HYBAS_ID')}). A plant takes the land where "
         f"distance / &radic;P is smallest, so at equal density its area grows in proportion to its size."),
        (code("buffer_length"), f"The plant's maximum service radius (m): r<sub>c</sub> &times; (P / 10,000)"
         f"<sup>{rc['radius_exponent']:g}</sup>, at most {cfg['max_buffer'] / 1000:g} km. r<sub>c</sub>, the radius "
         f"of a plant of 10,000 people in country c, grows with the share C<sub>c</sub> of its people connected "
         f"to treatment (JMP){spread_words}: r<sub>c</sub> = min({rule['max_km']:g} km, exp({rule['intercept']:.3f} "
         f"+ {rule['slope']:.3f} &middot; logit C<sub>c</sub>{spread_term}))."),
        (code("<year>_zonal_sum") + f" ({span})", "People (WorldPop 100 m constrained, R2024B) within "
         f"the plant's service area in that year: the population estimate of the plant.{projected}"),
        (code("<year>_zonal_std") + f" ({span})", "Standard deviation of the 100 m population cells "
         "within the service area."),
        (code("2014_zonal_sum") + ", " + code("2014_zonal_std"), "Kept for the format; empty (the 2014 WorldPop "
         "product is a 1 km one, not comparable)."),
    ])  # fmt: skip
    method = (
        f"<p>Each plant's service area is the part of its HydroBASINS level-6 basin that it wins in a weighted "
        f"Voronoi diagram ({code('weights')}), cut to a circle of radius {code('buffer_length')} around the plant and "
        f"to the plant's country. Land in a basin outside every circle is served by no plant.</p>"
        f"<p><b>Plant size P in this file (size measure <i>{measure}</i>):</b> {MEASURES[measure]}. The reported "
        f"size is HydroWASTE's {code('POP_SERVED')} (not {code('QUAL_POP')} 4), else the US CWNS 2022 population "
        f"served, else the UWWTD load entering the plant (p.e.). The tank estimate is log<sub>10</sub> P = "
        f"{f['intercept']:g} + &delta;<sub>c</sub> + {f['slope']:g} &middot; log<sub>10</sub> F, F = "
        f"{code('total_area')} and &delta;<sub>c</sub> a country offset; a plant without detected tanks (or a lagoon) "
        + ("gets a substitute tank area, its country's 16.2&nbsp;% quantile of tank areas (the median for plants "
           "never imaged)" if measure == "footprint" else
           "gets its country's typical size of such plants (for plants never imaged, Japan's median register "
           "plant)")
        + f". Every size of a country is then multiplied by its national scale k<sub>c</sub> = connected people "
        f"(JMP) / sum of the country's plant sizes, kept between {limits}.</p>"
    )  # fmt: skip
    new = (
        "<ul>"
        f"<li>Plant list version 3: HydroWASTE (corrected locations), UWWTD, US CWNS, Canada WSER, the Thailand geoAI "
        f"data and the location model's detections as before, with the corrected EU file (plants moved, added and "
        f"removed) and the national registers of Australia, Japan and New Zealand (and Canada's WSER plants) added; "
        f"plants closer than 500 m merged.</li>"
        f"<li>{code('SOURCE_ID')} (new column) and {code('SOURCE_DS')} for every plant: the source dataset and the "
        f"plant's ID there. {code('HW_WASTE_ID')} also for US plants matched to HydroWASTE.</li>"
        f"<li>Service areas confined to HydroBASINS level 6 (was 9); radius and weights from the plant size in people "
        f"({code('buffer_length')}, {code('weights')}), no longer from the tanks and the basin area.</li>"
        f"<li>Text such as &lsquo;&lt;NA&gt;&rsquo; that stood for a missing value in the first release (e.g. in "
        f"{code('uwwCode')}, {code('WWTP_NAME')}) is now empty.</li>"
        "</ul>"
    )  # fmt: skip
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{html.escape(name)} — Column Documentation</title>
<style>
@page {{ size: Letter; margin: 0.75in 0.8in 0.8in 0.8in; }}
body {{ font-family: 'DM Sans', 'Fira Sans', 'DejaVu Sans', sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1f1e1d; }}
h1 {{ font-family: 'Crimson Text', 'Gelasio', 'DejaVu Serif', serif; font-weight: 400; font-size: 25pt; line-height: 1.15; margin: 0 0 10pt; overflow-wrap: anywhere; }}
h2 {{ font-family: 'Crimson Text', 'Gelasio', 'DejaVu Serif', serif; font-weight: 400; font-size: 17pt; margin: 20pt 0 8pt; break-after: avoid; }}
.date {{ display: inline-block; background: #f0eee6; border-radius: 4px; padding: 1pt 6pt; font-size: 9pt; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; background: #f0eee6; border-radius: 3px; padding: 0.5pt 3pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 4pt; font-size: 9.5pt; }}
th {{ background: #f5f4ef; font-weight: 400; color: #5e5d59; text-align: left; padding: 5pt 7pt; border: 0.6pt solid #dedcd1; }}
td {{ vertical-align: top; padding: 5pt 7pt; border: 0.6pt solid #dedcd1; }}
tr {{ break-inside: avoid; }}
td:first-child {{ white-space: nowrap; }}
.note {{ background: #f5f4ef; border-radius: 6px; padding: 6pt 12pt; margin: 10pt 0; }}
.src {{ font-size: 9.5pt; color: #5e5d59; }}
ul {{ margin: 4pt 0; padding-left: 16pt; }} li {{ margin: 2pt 0; }}
</style></head><body>
<h1>{html.escape(name)}<br>— Column Documentation</h1>
<span class="date">{today}</span>
<p>This file holds {n:,} wastewater treatment plants (WWTPs) in {df["ISO_2"].nunique()} countries, merged from public
datasets &mdash; the EU Urban Waste Water Treatment Directive (UWWTD) reporting data, the HydroWASTE global database
after its WWTP locations have been corrected by a detection model, national plant data of the US, Canada, Australia,
Japan, New Zealand and Thailand &mdash; and new plants found by the detection model. It adds features detected by a
segmentation model on satellite imagery, site categories from an image-to-text model, and derived geospatial-analysis
fields: each plant's service area (a weighted Voronoi diagram within river basins, cut to a radius that grows with the
plant's size in people) and the people living in it, {people / 1e9:.2f} billion in 2024.</p>
<p>The GeoPackage holds two layers that share the same attribute columns:</p>
<ol><li>{code("WWTP_LOCATIONS")} &mdash; one point per WWTP; carries {code("WKT_SERVICE_AREA")}, the matching
service-area polygon written out as WKT text.</li>
<li>{code("WWTP_SERVICE_AREAS")} &mdash; one polygon per WWTP (its service area); carries {code("WKT_WWTP")}, the
matching plant point written out as WKT text.</li></ol>
<p>Every attribute column below (aside from {code("geometry")} itself) appears identically in both layers.</p>
<div class="note"><b>New since HeiGIT_WWTP_LVL9_BFK0_8_LOG_MULT.gpkg (17 Sep 2026)</b>{new}</div>
<h2>Identifiers &amp; provenance</h2>{ident}
<p class="src">Source datasets ({code("SOURCE_DS")}) and what {code("SOURCE_ID")} holds; {int(with_id.sum()):,} of
{n:,} plants have a {code("SOURCE_ID")}:</p>{sources_table}
<h2>Urban Waste Water Treatment Directive (UWWTD) columns</h2>{uww}
<p class="src">Source: EEA/EIONET UWWTD Art. 15 reporting &mdash; UWWTPs table definition.</p>
<h2>HydroWASTE columns</h2>{hw}
<p class="src">Source: HydroWASTE v1.0 (Ehalt Macedo et al., 2022).</p>
<h2>Segmentation model columns</h2>
<p>Detections of individual treatment tanks/basins in satellite imagery, split by shape.</p>{seg}
<h2>Annotation stage columns</h2>{ann}
<h2>Geospatial analysis columns</h2>{method}{geo}
</body></html>"""


def render(html_text, pdf_path):
    from playwright.sync_api import sync_playwright

    title = html.escape(os.path.basename(pdf_path).removesuffix(".pdf"))
    header = f'<div style="font-size:7pt;color:#8a8984;width:100%;padding:0 0.8in;font-family:sans-serif">{title}</div>'
    footer = ('<div style="font-size:7pt;color:#8a8984;width:100%;padding:0 0.8in;text-align:right;'
              'font-family:sans-serif">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>')  # fmt: skip
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME", "/usr/bin/google-chrome"))
        page = browser.new_page()
        page.set_content(html_text, wait_until="load")
        page.pdf(path=pdf_path, format="Letter", display_header_footer=True, header_template=header,
                 footer_template=footer, margin={"top": "0.75in", "bottom": "0.8in", "left": "0.8in", "right": "0.8in"},
                 print_background=True)  # fmt: skip
        browser.close()


if __name__ == "__main__":
    gpkg, size_measure, out = sys.argv[1:4]
    text = build_html(gpkg, size_measure)
    with open(out.removesuffix(".pdf") + ".html", "w", encoding="utf-8") as fh:
        fh.write(text)
    render(text, out)
    print(out)
