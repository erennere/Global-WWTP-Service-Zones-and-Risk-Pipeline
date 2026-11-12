# Service radius from the national connection rate: first fit (2026-11-12)

Question: can one global formula `r10k = exp(a + b * logit(C))`, with C the national share of people
connected to wastewater treatment, replace the single US-calibrated r10k (3.75 km reported, 4.25 km
footprint) and get each country's served share right? Nothing in config.yaml was changed.

## Inputs
- Curves: `snakemake connection_radius` on the cluster (branch `connection-radius`, c1480e3), WorldPop
  2022 1 km, 165 countries with plants and a raster. Per country, size measure (reported, footprint),
  exponent (0.3, 0.5) and 40 r10k values 0.5-40 km: people inside service areas of the stand-in
  (`wwtp/calibration/connection_radius.py` docstring; no basins).
- C (`read_references`, value of the year nearest 2022, at most 5 years off):
  JMP "safely managed via sewer" `san_sew_sm_t` (sewered and treated; 167 countries) is fitted;
  Eurostat `env_ww_con` URB_OTH_CS (33 of our countries) and OECD PUBTOTTR (39) are checks.
  On these countries JMP differs from Eurostat by 5.3 pp on average, from OECD by 6.8 pp (the floor
  for any check).
- 126 countries have curves and a JMP value. Missing: Japan, Australia, New Zealand (no plants in
  `all_annotated_v2.gpkg`, worth a look), Indonesia (38 plants, no JMP treated value).
- Fit: 5-fold cross-validation over countries (seed 0); a country's served share comes from a fit
  on the other folds. Errors = served share - C, in percentage points.

## Results, reported sizes, e 0.3 (JMP, 126 countries)

| model | mean abs | median abs | bias | within 10 pp | served / C people | Eurostat (33) mean abs, bias | OECD (39) mean abs, bias |
|---|---|---|---|---|---|---|---|
| current r10k 3.75 km | 15.0 | 10.1 | -10.8 | 50 % | 0.89 | 19.9, -19.1 | 24.7, -23.1 |
| one r10k for all, 4.89 km (in sample) | 12.4 | 7.1 | -4.7 | 64 % | 1.06 | 13.4, -11.1 | 18.1, -14.8 |
| **formula, logit(JMP), cross-validated** | **7.5** | **4.2** | **-1.9** | **77 %** | 1.10 | **10.9, -3.8** | **12.4, -6.2** |
| formula, logit(our reported coverage) | 12.9 | 5.9 | -7.1 | 62 % | 1.01 | 11.5, -7.5 | 18.2, -13.6 |

Fitted on all countries: a = 1.70, b = 0.206, so r10k = 3.0 km at C 5 %, 4.1 at 20 %, 5.5 at 50 %,
7.3 at 80 %, 10.0 at 95 %. Footprint sizes, e 0.5: current 4.25 km 15.8 pp -> formula 7.9 pp
(a 1.71, b 0.24). Exponent 0.5 with reported sizes is no better (8.3 pp). Link variants (clip 0.05,
linear C, sqrt C, capping C at 0.85-0.95) all land at 7.3-7.9 pp.

By connection class (reported, e 0.3; mean abs and bias, pp):

| C | countries | current | formula | needed r10k (median) |
|---|---|---|---|---|
| < 20 % | 54 | 5.1, +3.3 | 3.6, +1.5 | 2.3 km |
| 20-60 % | 30 | 13.6, -11.1 | 11.0, -3.7 | 5.5 km |
| > 60 % | 42 | 28.7, -28.7 | 10.1, -4.9 | 7.7 km |

Fitted on Eurostat or OECD instead (only high-connection countries, 33 / 40): current bias -19 / -23 pp;
one r10k for all (6.9 / 7.3 km) is about as good as the formula (10.1 vs 9.9 pp; 13.9 vs 11.9 pp). The
formula's gain is the split between low- and high-connection countries, not detail within Europe.

## Countries (reported, e 0.3)

| | C (JMP) | Eurostat / OECD | served, current | served, formula (CV) | needed r10k | formula r10k |
|---|---|---|---|---|---|---|
| USA | 0.84 | - | 0.57 | 0.83 | 7.5 | 7.6 |
| GBR | 0.97 | - | 0.85 | 0.99 | 7.4 | 10.8 |
| NLD | 0.97 | 1.00 / 1.00 | 0.85 | 1.00 | 6.2 | 11.4 |
| DEU | 0.95 | 0.96 / 0.98 | 0.77 | 0.98 | 7.5 | 9.8 |
| FRA | 0.79 | 0.82 / 0.82 | 0.74 | 0.88 | 4.5 | 7.1 |
| ESP | 0.86 | 0.89 / 0.89 | 0.83 | 0.95 | 4.4 | 8.0 |
| POL | 0.76 | 0.76 / 0.76 | 0.67 | 0.83 | 5.1 | 6.9 |
| TUR | 0.66 | 0.74 / 0.74 | 0.33 | 0.54 | 8.4 | 6.3 |
| CHN | 0.60 | - | 0.53 | 0.65 | 4.8 | 5.9 |
| MEX | 0.55 | - | 0.37 | 0.52 | 6.8 | 5.7 |
| RUS | 0.51 | - | 0.38 | 0.54 | 5.1 | 5.5 |
| ZAF | 0.48 | - / 0.65 | 0.43 | 0.56 | 4.3 | 5.4 |
| EGY | 0.47 | - | 0.68 | 0.83 | 2.6 | 5.3 |
| BRA | 0.42 | - / 0.55 | 0.46 | 0.56 | 3.3 | 5.1 |
| THA | 0.09 | - / 0.15 | 0.37 | 0.34 | 1.3 | 3.4 |
| IND | 0.05 | - | 0.16 | 0.13 | 1.3 | 3.0 |

No radius on the grid reaches C (plant list too thin): SVN (2 plants), KWT (6), ARE, CHL, KOR, LVA,
QAT, SGP, YEM; Georgia (5 plants) and Nigeria (3) need 30+ km. C below what the smallest radius gives
(plants exist, almost no one counted as treated): ARM, BGD, LSO, NPL.

## Reading
- The connection rate explains most of the country differences: one global formula with two numbers
  halves the error of the current setting (15.0 -> 7.5 pp; within 10 pp 50 % -> 77 %) and removes
  the bias in high-connection countries (-28.7 -> -4.9 pp). The checks against Eurostat and OECD,
  which were not fitted, agree (19.9 -> 10.9 pp, 24.7 -> 12.4 pp).
- The current setting under-serves every well-connected country by 10-30 pp (US 57 % vs 84 %).
  This is the national non-served share in the risk outputs, so the bias matters there.
- It overshoots at the top: the logit keeps growing, NL gets 11.4 km where 6.2 km already reaches
  97 %. In served share that costs little (NL 1.00 vs 0.97), but per-plant people grow with the radius.
- It over-serves where plants exist but much sewage is not treated: India 13 % vs 5 % (close to JMP
  "sewer connection" 14 %), Thailand 34 % vs 9 %, Egypt 83 % vs 47 %. These dominate the
  people-weighted total (1.10).
- Our own reported coverage is a much weaker predictor (12.9 pp): most countries report no sizes.
- Tension with the plant-level calibration: on matched plants the best r10k is GB 3.0, US 3.75,
  NL 6.0 km (2026-11-11 note); the national shares need 7.4, 7.5, 6.2 km. NL agrees (complete plant
  list). In GB and the US the larger radius reaches the national share by giving our plants the people
  of plants missing from our list (GB: 1,693 of 8,185 catchments matched). So national non-served
  share and per-plant served people cannot both be right with one radius where the list is incomplete.
- Usable in production: C comes from JMP for 167 countries. The formula needs C as input, so it fits
  the national share by construction; what the cross-validation tests is whether one curve transfers
  across countries. Eurostat and OECD are partly the same national sources JMP uses, so they are a
  consistency check, not independent truth.

## Open decisions
1. Adopt the formula for r10k per country (fallback for countries without a JMP value: the current
   3.75 km or the one-for-all 4.9 km)?
2. If yes, what to do about the plant-level tension: accept larger per-plant loads in GB/US-like
   countries, or use the formula only for the non-served share and keep 3.75 km for per-plant loads.
3. Countries with a thin plant list (above grid) and countries where plants exist but treatment is
   near zero (India, Thailand, Egypt): leave as is or flag.

Files (scratch, not in the repo): `fit_summary_jmp.csv`, `fit_countries_jmp.csv`, `reference_table.csv`
from `python -m wwtp.calibration.connection_radius fit` on the cluster outputs.
