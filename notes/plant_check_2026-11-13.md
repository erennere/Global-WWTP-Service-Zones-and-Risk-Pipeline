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
| plants | agglomerations d'assainissement: Grand Est layer (catchment sectors of the plants, SIG-GR) and Puy-de-Dôme (DDT 63) | FR | plant check once saved by hand into their folders (data.gouv.fr refuses automated downloads) |
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
- France: no national file; the two territorial layers above are used (people from the raster in
  the polygon). Files of kind `manual` in `national_data.sources`: saved by hand, any file name,
  every polygon layer in the folder is read.
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
