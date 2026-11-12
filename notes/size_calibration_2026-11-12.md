# Plant size by country and the national size scale (2026-11-12)

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
  at least 30 (33 countries). In these countries the typical error of the estimate drops from x2.20
  to x1.93 (on held-out halves of each country: x2.18 -> x1.94). Largest: India +0.89 (x7.8),
  China +0.76 (x5.7), Mexico +0.48 (x3.0); smallest: Latvia, Denmark -0.39 (x0.41).
- **Plants without a usable footprint** (lagoons, no detections): the median reported size of the
  country's such plants where it has at least 30 (28 countries), otherwise 0.41 x the median
  footprint estimate of its other plants (29 more countries), otherwise 2,580 as before. 0.41 is the
  median over the 28 countries of (median size-free / median sized); it predicts a left-out
  country's median within x1.44, one value for all within x1.70. Before, 2,580 everywhere: US 1,180,
  Netherlands 39,000, China 127,500.
- **National scale** k = connected people (JMP share x WorldPop 2022 people) / sum of the country's
  plant sizes, kept within the limits below. Reported sizes are partly population equivalents
  (industry), and our list is incomplete in some countries and over-complete in others; with k the
  sizes of a country add up to the people connected there. Reported sizes, raw k: median 1.73,
  US 1.17, GB 0.95, NL 0.93, Germany 0.74, China 1.19, Brazil 1.56, Mexico 2.17, India 0.34,
  Thailand 0.04 (2,917 plants, 9 % connected), Türkiye 5.2, Nigeria 856 (3 plants).

A factor k on every size of a country is a factor k ** e on all its radii and leaves the Voronoi
weights within the country unchanged, so the served-share curves made without k serve every k.

## How limits and cap were chosen
`calibrate` tabulates, for every pair of limits (none, 1/3-3, 0.2-5, 0.1-10) and cap of the radius
rule `r10k = min(cap, exp(a + b logit C))`, the error of the national served share when each SDG
region is predicted from a fit on the other six. The choice: the smallest national error among the
pairs whose mean plant-level error on the US sewersheds, GB catchments and NL CBS plants (class-
balanced, stand-in without basins) is not above that of the fixed radius with the same sizes.

Reported sizes, e 0.3 (curves of 12 Nov):

| limits | cap | national mean abs (people-weighted, bias), pp | plants US, GB, NL (mean) | totals US, GB, NL |
|---|---|---|---|---|
| 3.75 km everywhere | – | 15.0 (11.0, -10.8) | x1.38 1.44 1.51 (x1.445) | 0.99 1.11 0.86 |
| none | 5.5 | 10.1 (6.2, -7.0) | x1.45 1.53 1.35 (x1.441) | 1.25 1.22 0.97 |
| 1/3-3 | 5.5 | 8.3 (5.4, -5.2) | x1.47 1.53 1.34 (x1.443) | 1.27 1.22 0.97 |
| **0.1-10** | **5.5** | **7.5 (4.8, -4.3)** | x1.46 1.53 1.34 (**x1.441**) | 1.26 1.22 0.97 |
| 0.1-10 | 6.0 | 6.9 (4.8, -3.5) | x1.49 1.54 1.33 (x1.452) | 1.29 1.23 0.98 |
| 0.1-10 | none | 5.8 (5.4, -0.9) | x1.59 1.60 1.35 (x1.509) | 1.36 1.29 1.02 |

Chosen: limits 0.1-10, cap 5.5, a 1.371, b 0.183 (r10k 2.3 km at C 5 %, 3.2 at 25 %, 3.9 at 50 %,
5.5 from 86 %; on top of it the scale moves the radii by k ** 0.3). The national error falls by a quarter against the rule without scale; the
per-plant error stays.

Footprint sizes, e 0.5: the country parts alone improve the plants at the fixed 4.25 km (x1.526 ->
x1.459; NL x1.44 -> x1.31, GB x1.60 -> x1.54). No pair with a scale keeps that (x1.50-1.62), so the
footprint measure gets no scale, and the rule with cap 4.0 (a 1.214, b 0.135; plants x1.441,
national 16.0 pp). Its national numbers come from curves made with the old footprint sizes (China
and India were x5.7 and x7.8 too small), so they will move most in the calibration run.

## Status: provisional until the calibration run
The values in config.yaml come from the curves of 12 Nov, made with the old size model; the scale
was computed from the new one. The run `snakemake --profile workflow/profiles/slurm calibration`
remakes the curves with the new sizes (before the scale) and writes `calibration_results.zip`:
`plant_size_countries.csv` (offsets and size-free values from the pipeline's own country join,
next to the config values), `calibration_by_cap.csv` (a, b and national errors per limits and cap),
`calibration_countries.csv` (k and r10k per country), `plants.csv`, the curves, and the
`national_data` results (register plants, new plants, peek of every downloaded file). From it: the
plant-level check again, then the final values, then the sweep.

## Checks on the code
- `fit_country_sizes` on the plant table reproduces the config values (offsets within 0.0005,
  size-free values within 0.2 %; the config was computed with Natural Earth borders, the run uses
  the pipeline's Overture join, so small differences are expected).
- `calibrate` on the 12 Nov tables reproduces the earlier rule without scale (a 1.456, b 0.166 at
  cap 5.5, 10.1 pp) and the scratch tables with scale.
- Unit tests: country offsets, size-free values, scale on reported and estimated sizes, radius x k ** e;
  a synthetic world whose sizes are 4x off by country is fitted again by the scale; plan test: the
  target `calibration` makes no service areas.

## Independence
JMP now sets both the national scale and the radius rule; the leave-one-region-out errors are the
out-of-sample ones. GB and NL catchments enter only as a constraint on the choice of limits and cap
(not fitted), the US sewersheds also set the exponent. Census sewer connection by district
(Brazil, Mexico, India) stays independent.
