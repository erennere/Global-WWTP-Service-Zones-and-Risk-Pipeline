# Plant size by country and the national size scale (2026-11-12, final values 13 Nov)

Follows `radius_calibration_2026-11-12.md`. The radius rule was calibrated on 126 countries, but the
plant sizes it scales were still one global model: the footprint fit rests on 24,505 plants, 83 % in
Europe and Northern America, and plants without a usable footprint got one value everywhere. This
note adds the country parts of the size model and a national scale, and fixes the order of work:
calibration first (one cluster run, target `calibration`), sweep afterwards.

Code: `service_areas/size.py` (`plant_size_people`), `calibration/service_radius.py`
(`fit_country_sizes`), `calibration/connection_radius.py` (`curves`, `calibrate`, `pack`); config
`create_voronoi.footprint_to_people`, `size_scale`, `radius_calibration.*.by_connection`.

## The size model
`people = scale(country) * (reported size | footprint estimate | size-free value)`

- **Footprint estimate**: `log10 people = 1.523 + offset(country) + 0.70 log10 footprint_m2`. The
  offset is the median of `log10(reported / fit)` over the country's plants with both, where it has
  at least 30 (32 countries). In these countries the typical error of the estimate drops from x2.19
  to x1.92 (on held-out halves of each country: x2.20 -> x1.92). Largest: India +0.89 (x7.8),
  China +0.76 (x5.7), Mexico +0.48 (x3.0); smallest: Latvia, Denmark -0.39 (x0.41).
- **Plants without a usable footprint** (lagoons, no detections): the median reported size of the
  country's such plants where it has at least 30 (28 countries), otherwise 0.409 x the median
  footprint estimate of its other plants (29 more countries), otherwise 2,580 as before. 0.409 is the
  median over the 28 countries of (median size-free / median sized); it predicts a left-out
  country's median within x1.44, one value for all within x1.72. Before, 2,580 everywhere: US 1,180,
  Netherlands 36,600, China 127,500.
- **National scale** k = connected people (JMP share x WorldPop 2022 people) / sum of the country's
  plant sizes, kept within the limits below. Reported sizes are partly population equivalents
  (industry), and our list is incomplete in some countries and over-complete in others; with k the
  sizes of a country add up to the people connected there. Reported sizes, raw k: median 1.74,
  US 1.16, GB 0.95, NL 0.93, Germany 0.74, China 1.19, Brazil 1.56, Mexico 2.22, India 0.34,
  Thailand 0.04 (2,917 plants, 9 % connected), Türkiye 5.2, Nigeria 855 (3 plants). Within 0.1-10:
  101 of 126 countries; 18 at 10, 7 at 0.1.

The country values come from `calibrate` on the pipeline's own country join (Overture), run of
13 Nov (`plant_size_countries.csv`).

A factor k on every size of a country is a factor k ** e on all its radii and leaves the Voronoi
weights within the country unchanged, so the served-share curves made without k serve every k.

## How limits and cap were chosen
`calibrate` tabulates, for every pair of limits (none, 1/3-3, 0.2-5, 0.1-10) and cap of the radius
rule `r10k = min(cap, exp(a + b logit C))`, the error of the national served share when each SDG
region is predicted from a fit on the other six. The choice: the smallest national error among the
pairs whose mean plant-level error on the US sewersheds, GB catchments and NL CBS plants (class-
balanced, stand-in without basins) is not above that of the fixed radius with the same sizes.

Reported sizes, e 0.3 (curves of 13 Nov, made with the new sizes before the scale):

| limits | cap | national mean abs (people-weighted, bias), pp | plants US, GB, NL (mean) | totals US, GB, NL |
|---|---|---|---|---|
| 3.75 km everywhere | – | 14.8 (10.3, -10.5) | x1.38 1.44 1.51 (x1.445) | 0.99 1.11 0.86 |
| none | 5.5 | 10.1 (7.5, -6.8) | x1.45 1.53 1.35 (x1.441) | 1.25 1.22 0.97 |
| 1/3-3 | 5.5 | 8.3 (6.0, -5.2) | x1.46 1.53 1.34 (x1.443) | 1.27 1.22 0.97 |
| **0.1-10** | **5.5** | **7.5 (5.3, -4.3)** | x1.46 1.53 1.34 (**x1.440**) | 1.25 1.22 0.97 |
| 0.1-10 | 6.0 | 6.9 (5.3, -3.4) | x1.48 1.54 1.33 (x1.450) | 1.28 1.23 0.98 |
| 0.1-10 | none | 5.8 (6.0, -0.9) | x1.58 1.60 1.35 (x1.508) | 1.36 1.29 1.02 |

Chosen: limits 0.1-10, cap 5.5, a 1.367, b 0.182 (r10k 2.3 km at C 5 %, 3.2 at 25 %, 3.9 at 50 %,
5.5 from 86 %; on top of it the scale moves a country's radii by k ** 0.3). The national error falls
by a quarter against the rule without scale, by half against 3.75 km everywhere; the per-plant
error stays. The same choice came out of the provisional curves of 12 Nov (old sizes): the country
parts of the size hardly move the reported-size results, since most plants there report a size.

Largest national misses of the chosen rule (region left out): countries with a handful of plants,
where even k = 10 cannot reach the connected share (Slovenia 2 plants -67 pp, Georgia, Kuwait,
Azerbaijan, UAE), Egypt +38 pp (460 plants, all where people live; the scale enlarges them), Chile
-26 pp, Canada -26 pp (2,352 plants, 7 with a reported size), Finland -23 pp. Weighted by people:
China +4 pp, India +4 pp (were +5 and +9 pp without the scale), US -9 pp. Thin plant lists are the
remaining error, which the register plants address.

Footprint sizes, e 0.5: the country parts alone improve the plants at the fixed 4.25 km (x1.526 ->
x1.457; NL x1.44 -> x1.30, GB x1.60 -> x1.54). No pair with a scale keeps that (x1.50-1.62), so the
footprint measure gets no scale, and the rule with cap 4.0 (a 1.197, b 0.137; plants x1.440,
national 16.0 pp, people-weighted 14.0; 4.25 km everywhere 16.0 pp, people-weighted 17.8). With
footprint sizes the national level stays poor: China and India now get their larger sizes and are
over-served (+13 and +14 pp), the US under-served (-48 pp; its footprints are small and its many
lagoons get 1,180 people). The sweep shows what that costs against the reported sizes.

The plant-level check is a scratch stand-in (US/GB/NL reference sets and `plant_check.py`, kept
outside the repository on the laptop under `data/calibration/plant_check/`).

## Checks on the code
- `fit_country_sizes` (run on the cluster) and the scratch fit on the Natural Earth join agree:
  offsets within 0.02, size-free values within a few % except Croatia (+26 %, 46 plants) and
  Zimbabwe (new); the config takes the cluster values.
- `calibrate` on the 12 Nov tables reproduces the earlier rule without scale (a 1.456, b 0.166 at
  cap 5.5, 10.1 pp) and the scratch tables with scale.
- Unit tests: country offsets, size-free values, scale on reported and estimated sizes, radius x k ** e;
  a synthetic world whose sizes are 4x off by country is fitted again by the scale; plan test: the
  target `calibration` makes no service areas.

## National data in the same run
`national_data/` of the zip: Australia's register (Geoscience Australia, 1,363 plants) is the only one
read; none of its plants is within 1 km of ours (our list has no Australian plant; median distance
2,300 km), so all are in `new_plants.csv`. Canada's register failed on its text encoding (cp1252;
fixed: `encoding` per register), Brazil's (ANA, 403) and Japan's (MLIT P22, 404) downloads failed,
as did Spain's agglomerations (connection) and Eurostat NUTS 2 (404). Downloaded and not yet used:
Ireland's agglomerations, NL purification units, Mexico's census by locality, SDG 6.3.1, UNSD.

## Independence
JMP now sets both the national scale and the radius rule; the leave-one-region-out errors are the
out-of-sample ones. GB and NL catchments enter only as a constraint on the choice of limits and cap
(not fitted), the US sewersheds also set the exponent. Census sewer connection by district
(Brazil, Mexico, India) stays independent.
