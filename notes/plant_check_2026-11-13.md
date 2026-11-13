# Calibration on every dataset of real catchments (2026-11-13)

Until now the national side of the calibration used 126 countries (JMP), but the plant-level side,
which picks the cap of the radius rule and the limits of the national size scale, ran only on US
sewersheds, GB catchments and NL CBS counts, in a scratch script with reference sets built by hand.
This moves it into the repository and the cluster run, and widens it to every dataset of real
catchments and census districts the downloads provide.

Code: `wwtp/calibration/plant_check.py` (rule `connection_radius_plant_check`, part of the target
`calibration`); SDG 6.3.1 and the non-fitted checks in `connection_radius.calibrate`; the file of
uncovered countries in `pop_validation_scripts/national_data.py`. Settings: `connection_radius`
(`plant_reference_kinds`, `plant_reference_polygons`, `district_references`, `check_references`,
`sdg631`, `choice_national`), `national_data.uncovered`.

## What each dataset does now
| level | dataset | country | role |
|---|---|---|---|
| plants | EPA sourced sewersheds (people from the raster in the polygon) | US | plant check |
| plants | Hoffmann et al. catchments | GB | plant check |
| plants | CBS inhabitants per RWZI | NL | plant check |
| plants | purification units (zuiveringseenheid, polygons) | NL | plant check |
| plants | EPA agglomeration boundaries (1,076 polygons) | IE | plant check |
| plants | MITECO agglomerations Q2023 | ES | plant check once downloaded (connection failed on 12 Nov) |
| districts | Census 2020 localities (ITER): drainage by municipality | MX | upper-bound check, reported |
| national | JMP sewered and treated | 126 countries | fit (leaving one SDG region out) |
| national | SDG 6.3.1 2025: sewer volume share x share delivered to treatment | ~165 | check, and half of the national error in the choice |
| national | Eurostat, OECD | Europe, OECD | checks |

The plant check uses the population raster of the calibration year inside each real catchment as the
reference (except the CBS counts), so it tests the shape of the service areas, not the population
data. Every configuration of `calibrate` (4 scale limits x 13 caps) and the fixed radius is run with
the stand-in of the curves on each reference country.

**Choice** per size measure: the smallest national error (mean of the JMP error leaving one region out
and the SDG 6.3.1 error) among the configurations whose plant-level error (mean over the reference
countries) is not above the fixed radius's. `plant_choice.csv` also gives the choice with each
reference country left out; if no rule keeps the plant-level error, the fixed radius stays.

## Not used, and why
- France: no national file; the territorial layers on data.gouv.fr (Grand Est catchment sectors,
  Puy-de-Dôme agglomerations) could not be used (the files did not work) and are left out. The file
  kind `manual` (saved by hand into a source folder) stays for such cases.
- Germany NRW (Kommunale Kläranlagen NRW): the open.nrw entry lists no download; the data are only
  in the ELWAS-WEB viewer, and the organisation's policy blocks that site for Claude's browser, so
  its requests could not be inspected and no scraper was written. Options: an export from ELWAS-WEB
  in Eren's own browser (saved into a folder like the French files), or a request to LANUK.
- Eurostat NUTS 2 (`env_wwcon_r2`): discontinued (last update 2016, years to 2013); removed from the
  downloads.
- Brazil ANA Atlas Esgotos: the files exist at the configured addresses but the server refused the
  download (403); retried with a browser-style User-Agent (`national_data.request_headers`).
- Japan MLIT P22: the address returned 404 and no working download link was found; Japan's plants
  come from HydroWASTE for now.
- Switzerland, Denmark: catchment polygons not confirmed as national files.

## Countries our list does not cover
Australia, Japan and New Zealand have no plant of ours (the detections never covered them; Australia's
1,363 register plants are all more than 1 km from any plant of ours). `uncovered_plants.gpkg / .csv`
(national_data) holds their register plants (Geoscience Australia; Japan P22 when downloadable) and
HydroWASTE records (cleaned file: AU 818, JP 77, NZ 123), one row per plant, without size, not merged
into the plant list: their size will come from a model.

## Smoke run here (GB only, WorldPop 2020 1 km, plants of the 13 Nov run)
1,727 GB catchments matched. Reported sizes: fixed 3.75 km x1.60 (total 1.16); every rule raises the
GB error (cap 5.5: x1.70, total 1.27), because our list misses most small GB plants and their people
go to ours. GB alone would keep the fixed radius; the choice weighs all reference countries.

## First run on the cluster (13 Nov, evening)
References used: US 2,355 sewersheds (95 M people), GB 1,727 catchments (50 M), NL 308 CBS plants and
290 purification units, IE 176 agglomerations. Spain: download failed again (connection); Brazil
still refused (403) with the header.

The plant-level error hardly separates the configurations: reported sizes, every rule between x1.37
and x1.47 (fixed radius x1.38: GB 1.33, IE 1.33, NL 1.48, US 1.39); the national error ranges from
13.7 to 5.8 pp (JMP, region left out). The plain "not worse than the fixed radius" chose cap 5.25
without scale (plants x1.383 = fixed), but with one reference country left out the choice jumped
between cap 3.5 and cap 7.0 with the 0.1-10 scale: chosen by noise. So the comparison is now a
paired bootstrap over the units (`plant_bootstrap_samples`, `plant_confidence`): a rule is excluded
only when it is worse than the fixed radius in at least 95 % of the samples. The run needs only
`connection_radius_plant_check` again (with the per-unit predictions, `plant_units.csv.gz`).

Footprint sizes: chosen scale 1/3-3, no cap (a 1.321, b 0.197): plants x1.480 against x1.493 fixed,
national 7.1 pp (JMP) / 11.5 pp (SDG 6.3.1), better nationally than any reported-size rule.
Mexico: served share 42-52 % against drainage (sewer or septic) 96 %: the upper bound does not bind.
New plants (`new_plants.csv`, one file, no size, not merged): AU 1,490 (691 in both the register and HydroWASTE), CA 91 (Canada's register, 1-86 km from our plants, median 9 km), JP 77, NZ 123; 1,781 in all.

## Rerun with the bootstrap (13 Nov, afternoon): final values
Spain's agglomerations are still missing (no polygon layer in the download folder); the other five
references as before.

reported (the paper's measure): scale 0.1-10, cap 5.75 km, a 1.387, b 0.187. 24 of 52 settings pass
the bootstrap; the chosen one is worse per plant than the fixed radius in 94.8 % of the samples (just
inside the 95 % limit): the national error falls with every larger cap while the per-plant error
rises, so the choice is always the largest cap the catchments allow. Plants x1.41 against x1.38
(US 1.40/1.39, GB 1.42/1.33, IE 1.49/1.33, NL CBS 1.30/1.42, NL units 1.33/1.54; totals US 0.84 ->
1.07, NL 0.86 -> 0.97, IE 1.47 -> 1.80). National, region left out: JMP 7.2 pp (fixed 14.8;
people-weighted 5.3 against 10.3), SDG 6.3.1 12.5 (20.1), Eurostat 12.3 (19.9), OECD 13.3 (24.7);
service areas below SDG/Eurostat/OECD by 10-12 pp on average. Leaving one reference country out:
always 0.1-10, caps 4 (NL out) to 40 (IE out). Mexico: 51 % served against 96 % of dwellings with
drainage.

footprint: scale 1/3-3, no cap (40 km), a 1.321, b 0.197; plants x1.48 against x1.49; JMP 7.1 pp,
SDG 6.3.1 11.5 pp.

Written to config.yaml (create_voronoi.radius_calibration.*.by_connection, size_scale.footprint;
size_scale.reported unchanged). Figures and Methods numbers: `snakemake calibration_figures`.
