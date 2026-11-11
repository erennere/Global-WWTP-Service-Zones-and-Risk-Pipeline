# Service areas against CBS inhabitants per RWZI (NL), 2026-11-06

Independent test of the service-area rule in `service_radius_size_rule_2026-11-05.md`. Nothing was fitted
on these numbers.

## Setup
- Truth: CBS "Aantal inwoners per RWZI-verzorgingsgebied, 1 januari 2024" (Tabel 1, inhabitants per
  plant catchment; 17,942,942 people, 771 of them not assigned to a plant). Our plant = the one with
  `uwwCode` = `NL` + the 5-digit `rwzi_code`: 307 of 316 RWZIs matched (98.0 % of the people).
- Plants: the 313 rows of `data/all_merged_v2.gpkg` inside the NL population raster (all carry a UWWTD
  code; 5 have `POP_SERVED`). With `size_measure: reported` the size is the UWWTD load (p.e.).
- Run: `python -m wwtp.create_voronoi --approach 1` (code at 74863be) on that data, HydroBASINS lake
  `eu` levels 6 (12 basins in NL) and 8 (98), country boundary from the raster mask.
- Population: WorldPop 2020 1 km density x cell area (17.50 M), scaled by 1.025 to the CBS 2024 total so
  that growth does not count as error. A plant's people = cells whose centre lies in its service area.
- No PC4 polygons (PDOK not reachable from the sandbox), so per plant totals only: no capture or IoU.

## Results (307 plants: 61 <10k, 198 10k-100k, 46 100k-500k, 2 >500k people)
Ratio = service-area people / CBS inhabitants.

| variant | typical error | within x2 | median bias | total | <10k | 10k-100k | 100k-500k | >500k | NL people in any area |
|---|---|---|---|---|---|---|---|---|---|
| level 6, reported (default) | x1.47 | 74 % | x0.70 | 0.84 | 0.63 | 0.75 | 0.86 | 1.22 | 82 % |
| level 8, reported | x1.63 | 67 % | x0.65 | 0.72 | 0.75 | 0.73 | 0.69 | 0.85 | 71 % |
| level 6, footprint | x1.52 | 68 % | x0.86 | 0.83 | 1.13 | 0.88 | 0.81 | 0.66 | 82 % |
| level 8, footprint | x1.62 | 60 % | x0.73 | 0.69 | 1.05 | 0.75 | 0.63 | 0.65 | 68 % |
| diagnostic: level 6, no circle (rigid 50 km) | x1.34 | 81 % | x0.84 | 1.01 | 0.90 | 0.92 | 1.03 | 1.37 | 99 % |
| diagnostic: level 8, no circle | x1.40 | 79 % | x0.89 | 0.99 | 1.64 | 1.08 | 0.88 | 0.93 | 97 % |
| diagnostic: level 6, reported, r10k 5 km | x1.36 | 82 % | x0.79 | 0.93 | 0.77 | 0.83 | 0.95 | 1.34 | 91 % |

- The size is not the problem: UWWTD load / CBS inhabitants has median 1.02 (IQR 0.87-1.18), the same
  in every size class.
- The shortfall is the circle: 3.1 M people (18 %) lie outside every service area at level 6. They are
  mostly rural and suburban: uncovered share 60 % of people in cells <100 /km2, 39 % at 100-300,
  25 % at 300-1,000, 12 % at 1,000-3,000, 3.5 % above 3,000.
- Level 8 cuts through cities: 15 % of people in cells above 3,000 /km2 are uncovered (a basin without a
  plant gets no service area). Level 6 is better on every measure, as on GB.

## GB with the same code and variants (Hoffmann catchments, 1,693 plants; see the 2026-11-05 note)

| variant | typical error | within x2 | total | <10k | 10k-100k | 100k-500k | >500k | capture | GB people in any area |
|---|---|---|---|---|---|---|---|---|---|
| level 6, reported | x1.56 | 68 % | 1.11 | 1.34 | 1.10 | 1.08 | 1.06 | 0.69 | 84 % |
| level 8, reported | x1.59 | 66 % | 1.07 | 1.41 | 1.10 | 1.00 | 1.01 | 0.65 | 77 % |
| level 6, no circle | x1.70 | 62 % | 1.30 | 1.57 | 1.27 | 1.33 | 1.19 | 0.73 | 99 % |
| level 8, no circle | x1.89 | 54 % | 1.35 | 2.27 | 1.37 | 1.24 | 1.14 | 0.69 | 98 % |

## Reading
- GB and NL do not share a bias: GB slightly over (1.07-1.11), NL under (0.72-0.84). So no refit of the
  radius (the agreed rule: refit with leave-one-country-out only if both show the same bias).
- The circle is a stand-in for plants missing from our list. GB lacks most small plants (only 1,693 of the 8,185
  catchments could be matched to a plant of ours): without the circle our plants take their people and GB gets worse
  (x1.70, total 1.30). NL's list is complete (307 of 316 RWZIs) and everyone is connected: without the
  circle NL gets better (x1.34, 81 %, total 1.01).
- Consequence for the risk outputs: in a fully sewered country with a complete plant list the pipeline
  counts about 16-18 % of the people as non-served (CBS: 0.004 %).
- Open decision: keep the circle everywhere and report this as a limit, or loosen it per country where an
  independent source says nearly everyone is connected to treatment and our plant list is complete
  (Eurostat `env_ww_con`, UWWTD plant counts). Best decided after the country table (validation plan,
  step 3), which shows how many countries are like NL.
- Level 6 stays the default; both tests prefer it to level 8.
