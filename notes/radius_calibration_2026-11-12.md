# Radius scale calibrated on 126 countries (2026-11-12)

The service radius is `r10k * (people / 10,000) ** e`. Until now e and r10k came from one source: EPA
sewersheds in seven US states. This note replaces the single r10k by a rule calibrated on national
statistics from every region, and records what was tested and rejected.

Code: `python -m wwtp.calibration.connection_radius calibrate` (inputs: the `curves` and `references`
of the same module, run on the cluster 12 Nov); rule in `create_voronoi.radius_calibration.<measure>.
by_connection` and `connected_share`; applied in `service_areas/size.py`.

## The rule
`r10k_km = min(max_km, exp(intercept + slope * logit(C)))`, C = share of the country's people whose
sewage reaches treatment (JMP household file, "safely managed via sewer", value nearest 2022; 166
countries in config). Countries without a value keep the fixed r10k (3.75 km reported, 4.25 km
footprint; the rule gives 3.6 km at the median country, C = 0.25).

| size measure | e | intercept | slope | max_km | r10k at C = 5 %, 25 %, 50 %, 80 % |
|---|---|---|---|---|---|
| reported | 0.3 | 1.456 | 0.166 | 5.5 | 2.6, 3.6, 4.3, 5.4 km |
| footprint | 0.5 | 1.297 | 0.151 | 4.5 | 2.3, 3.1, 3.7, 4.5 km |

## How it was calibrated
- Data: for 126 countries with plants, a 2022 WorldPop raster and a JMP value, the share of people
  inside service areas (stand-in of `connection_radius.py`: plant with the smallest distance /
  sqrt(people), inside its circle; no basins) for 40 values of r10k.
- Fit: per country the r10k that reproduces the JMP share ("needed"); a line through
  log(min(needed, cap)) against logit(C).
- Test: every SDG region predicted from a fit on the other six (leave one region out). Error = served
  share minus JMP share, percentage points.
- Cap: the largest value that does not raise the mean plant-level error on the three countries with
  real catchments (US sewersheds, GB catchments, NL CBS; class-balanced error, stand-in at 1 km).

### Reported sizes, e 0.3
| rule | countries: mean abs | people-weighted | bias | plants US, GB, NL (mean) | plant totals US, GB, NL |
|---|---|---|---|---|---|
| 3.75 km everywhere (before) | 15.0 | 11.0 | -10.8 | x1.38 1.44 1.51 (x1.445) | 0.99 1.11 0.86 |
| one r10k for all, region left out | 12.3 | 10.9 | -1.8 | – | – |
| rule, cap 5.0 | 11.0 | 7.0 | -8.4 | x1.43 1.51 1.36 (x1.431) | 1.19 1.20 0.95 |
| **rule, cap 5.5** | **10.1** | **6.2** | -7.0 | x1.46 1.53 1.35 (**x1.440**) | 1.25 1.22 0.97 |
| rule, cap 6.0 | 9.5 | 5.7 | -5.8 | x1.49 1.54 1.33 (x1.452) | 1.30 1.24 0.99 |
| rule, no cap | 8.6 | 7.5 | -0.4 | x1.62 1.60 1.35 (x1.518) | 1.40 1.30 1.02 |

By held-out region (rule without cap against 3.75 km): Northern Africa and Western Asia 18.0 against
32.4 pp; Central and Southern Asia 3.6 / 5.4; Europe and Northern America 10.0 / 16.0; Sub-Saharan
Africa 2.9 / 4.4; Latin America 7.6 / 11.4; Eastern and South-Eastern Asia 6.0 / 18.2; Oceania 3.0 /
4.7. Intercept and slope hardly move with the region left out (1.62-1.97, 0.18-0.29).

### Footprint sizes, e 0.5
4.25 km everywhere: 15.8 pp, plants x1.526. Rule with cap 4.5: 14.0 pp (people-weighted 9.8),
plants x1.529. Larger caps lower the national error (cap 6: 10.5 pp) but raise the plant error
(x1.61), so the cap stays at 4.5.

### Real pipeline (basins, level 6) on GB and NL, reported sizes
| | GB class-bal., total, capture | NL class-bal., total | JMP error (GB, NL) |
|---|---|---|---|
| 3.75 km | x1.49, 1.15, 0.69 | x1.57, 0.82 | 14.0 pp |
| rule (5.5 km in both) | x1.59, 1.27, 0.72 | x1.36, 0.92 | 4.5 pp |

## What it costs
In the US and GB our plants now hold about 20-25 % more people in total than their real catchments
(plant-level error per plant about the same on average, worse in GB: x1.49 -> x1.59). Both lists miss
plants (GB: 1,693 of 8,185 catchments matched); the larger circle gives their people to the plants we
have. In NL, whose list is complete, the rule is better on every measure. Adding the register plants
(`national_data`, new_plants.gpkg) is the way to remove this cost.

## Tested and not adopted
- One r10k for all, refitted on US + GB + NL: flat between 3.5 and 6 km (x1.43-1.46); nothing to gain.
- r10k = median radius holding 10,000 people around the plants of a country: wrong direction (NL is
  densest and needs the largest radius).
- Circle holding as many people as the plant's size: better with reported sizes (x1.36 against x1.445)
  but fails with footprint sizes (x2.2 against x1.53); 115 of 165 countries have no reported sizes.
- The rule without cap: best for countries, worst for plants.

## Plant size from the footprint: not calibrated globally yet
`footprint_to_people` (log10 people = 1.523 + 0.70 log10 m2) rests on 24,505 plants, 83 % of them in
Europe and Northern America. Median reported / predicted people: China x5.7 (1,638 plants), India x7.8
(380), Mexico x3.1 (179), Brazil x0.67, South Africa x0.56, Europe and Northern America x0.88. Plants
without a reported size: 77 % in Eastern and South-Eastern Asia, 72 % in Central and Southern Asia,
99 % in Northern Africa and Western Asia. A regional fit cannot be tested yet (one or two countries
per region outside Europe); the registers of the `national_data` run (Brazil, Japan, Canada, then
Mexico, India, Korea, South Africa) are the data for it.

## Independence
JMP is now calibration data for the radius scale of size-based variants; the leave-one-region-out
numbers above are the out-of-sample ones. Eurostat and OECD are partly the same national sources.
Independent: GB catchments, NL CBS (plants); census sewer connection by district (to come).
