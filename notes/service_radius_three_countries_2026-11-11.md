# Service radius: does the US calibration transfer? Leave-one-country-out (2026-11-11)

Question: the radius rule `radius_km = r10k * (people / 10,000) ** e` (at most 50 km) was fitted on the
US only (`service_radius_size_rule_2026-11-05.md`). Fitted on two of US, GB and NL, how well does it do on
the third? Nothing in config.yaml was changed.

## Method
- The same stand-in as `python -m wwtp.calibration.service_radius` (approach 1 without basins): each
  population point goes to the plant with the smallest distance / sqrt(people) among its 64 nearest
  plants and stays with it only inside that plant's circle. Plant sizes as in the pipeline
  (`plant_size_people`, `reported` and `footprint`).
- US: 2020 census block-group centroids, 444 plants matched to EPA sourced sewersheds in CT, KY, MA,
  NH, RI, UT, VT (11.4 M people). GB: 1 km WorldPop 2020 cells, 1,693 plants matched to Hoffmann et al.
  catchments (44.7 M). NL: 1 km WorldPop 2020 cells scaled to the 2024 total, 307 plants matched to
  CBS inhabitants per RWZI (17.6 M).
- Grid e 0.20-0.65, r10k 1.5-12 km. Criterion as in the calibration: mean over the size classes <10k,
  10k-100k, >100k people of the class median |log10 predicted / reference| ("class-balanced error"),
  with the total within 10 % of the reference. Two training countries: mean of their criteria, both
  totals within 10 %.

## Results, reported sizes (the default)
Class-balanced error (typical error), share within a factor 2, total predicted / reference.

| setting | US | GB | NL |
|---|---|---|---|
| current: e 0.30, r10k 3.75 (US fit) | x1.38 (x1.46), 74 %, 0.99 | x1.44 (x1.58), 67 %, 1.11 | x1.51 (x1.42), 74 %, 0.86 |
| each country's own best | e 0.30, 3.75: x1.38, 0.99 | e 0.45, 2.25: x1.40, 1.06 | e 0.30, 6.0: x1.33, 0.99 |
| own best at e 0.30 | r10k 3.75 | r10k 3.0 (x1.42, 1.00) | r10k 6.0 |
| fit GB+NL (e 0.20, 5.5) -> US | **x1.46**, 69 %, 1.07 | x1.49, 1.13 | x1.36, 0.90 |
| fit US+NL (e 0.20, 5.75) -> GB | x1.45, 1.10 | **x1.50**, 64 %, 1.15 | x1.35, 0.92 |
| fit US+GB (e 0.30, 3.5) -> NL | x1.40, 0.93 | x1.43, 1.08 | **x1.56**, 73 %, 0.83 |
| fit all three (e 0.30, 5.0) | x1.43, 1.19 | x1.51, 1.20 | x1.36, 0.95 |

Bold: the held-out country. GB+NL and all three: no pair keeps every training total within 10 %, so
the best criterion is shown regardless.

Footprint sizes: current e 0.50, r10k 4.25 gives US x1.54, GB x1.60, NL x1.44; held out x1.56-1.62;
own bests x1.37-1.54. Same picture, everything about x0.1 worse than reported sizes.

## Reading
- The US setting transfers about as well as any two-country fit: on GB x1.44 against GB's own best
  x1.40; on NL x1.51 against x1.33. Fitted on two countries, the held-out errors are x1.46-1.56, not
  better than the current setting on GB and NL. Calibrating on the US alone cost little.
- The exponent is stable: e 0.30 is (near) best in all three countries.
- The scale of the radius is not: at e 0.30 the best r10k is 3.0 km (GB), 3.75 km (US), 6.0 km (NL).
  No single value keeps all three totals within 10 %: larger circles over-count GB, smaller ones
  under-count NL. GB's plant list misses most small plants (their people go to ours), NL's list is
  complete and nearly everyone is connected.
- So the country difference sits in one number, the radius at 10,000 people, and it follows how
  complete the plant list is and how many people are connected. That is what a density- or
  connection-based radius (option 2) would have to capture; a single global r10k leaves a country
  bias of about -14 % (NL) to +11 % (GB) in served population (current setting).
