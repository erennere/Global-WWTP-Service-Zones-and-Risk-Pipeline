# Service areas from plant size in people (2026-11-05)

Replaces the footprint-only radius of commit 1939fc6 (`radius_quantile`).

## Rule
- Plant size S in people (`create_voronoi.size_measure`):
  - `reported` (default): first positive value of `POP_SERVED` (HydroWASTE; QUAL_POP 4 skipped, it is the
    people within 11 km of the plant), `EPA POP 2022` (CWNS), `uwwLoadEnteringUWWTP` (UWWTD, p.e.);
    otherwise the footprint estimate.
  - `footprint`: `log10 S = 1.523 + 0.70 log10 F` (F = detected tanks + rectangles; 24,505 plants, R2 0.31);
    lagoons and plants without detections 2,580 people.
- Voronoi weight (`weight_method: size`, multiplicative distance): sqrt(S). A weighted site's area grows
  with the square of its weight, so at equal density its area grows in proportion to its people.
- Circle: `radius_km = r10k * (S / 10,000) ** e`, at most 50 km. reported: e 0.3, r10k 3.75 km;
  footprint: e 0.5, r10k 4.25 km.

## Calibration (US only): `python -m src.calibration.service_radius`
2020 census block-group centroids in 444 EPA sourced sewersheds (CT, KY, MA, NH, RI, UT, VT) matched to
our plants. Each centroid goes to the plant with the smallest distance / sqrt(S) and stays inside its
circle. Grid e in 0.3..0.5, r10k in 2..6 km. Chosen: smallest class-balanced error (mean over <10k,
10k-100k, >100k of the median |log10 predicted/census|; a plant holding nobody counts as a failure) among
pairs whose total is within 10 % of the census total.

- reported: class-balanced error x1.38, 74 % within a factor 2, total 0.99, class ratios 1.25 / 1.02 / 0.94.
  Leave-one-state-out picks e 0.3-0.35, r10k 3.25-3.75.
- footprint: x1.54, total 0.92, class ratios 1.56 / 1.20 / 0.67.

The 10 % total constraint was added after a first GB run of the unconstrained pick (e 0.3, r10k 3.25,
US total 0.87) showed 3 M fewer GB catchment residents inside any service area. On the US the two picks
are equally good (x1.378 vs x1.381); the constraint decides between them. No number was fitted on GB.

## GB test: the real pipeline (`create_voronoi --approach 1`, level-8 basins)
1,909 GB plants, 1 km WorldPop 2020 (user upload), Hoffmann et al. catchments. Catchment-plant match:
UWWTD link (Waterbase lookup, our plant within 1 km of the UWWTD plant) for the 1,770 linked catchments,
otherwise the only plant of ours inside an unlinked catchment. 1,693 plants scored.

Service-area population / catchment population, by catchment size (capture = share of the catchment's
people inside the plant's area):

| rule | typical error | within x2 | <10k | 10k-100k | 100k-500k | >500k | capture 100k-500k | capture >500k |
|---|---|---|---|---|---|---|---|---|
| before the audit (438428d) | x2.10 | 48 % | 2.99 | 1.14 | 0.66 | 0.40 | 0.42 | 0.28 |
| footprint radius, q 0.6 (1939fc6) | x2.06 | 48 % | 2.96 | 1.38 | 0.87 | 0.69 | 0.50 | 0.40 |
| this rule, `footprint` | x1.82 | 56 % | 2.21 | 1.35 | 0.97 | 0.67 | 0.55 | 0.39 |
| this rule, `reported` (default) | x1.59 | 66 % | 1.40 | 1.09 | 1.00 | 1.01 | 0.61 | 0.60 |

People inside any service area: 49.2 M before, 53.0 M now (42.6 M of them inside a real catchment; the
catchments hold 52.1 M).

## What limits large plants
- Footprint alone cannot size the largest plants: GB catchments >500k people have a median footprint of
  52k m2 against 43k m2 for 100k-500k (people per footprint varies 10x). Reported sizes fix this.
- Still missed for >500k (reported): 30 % of their people lie in another level-8 basin, 9 % go to another
  plant, 2 % outside every circle. Coarser basins or none did not raise capture (tested level 5-7 and no
  basins): the cells then go to neighbouring plants. Shares between large neighbours (London's
  Beckton/Crossness/Mogden/Long Reach) are set by sewer history, not by distance.
- "Beyond" (share of catchments whose area misses more than 5 % of their people) stays high for large
  catchments (0.79 and 0.89); no point-based split of a multi-plant city reaches 95 %.

## Consequence for validation (pop_served_validation_plan, rule 2)
With `size_measure: reported` HydroWASTE POP_SERVED, CWNS and UWWTD loads are inputs. Plant-level
comparisons against them test the geometry only weakly; GB catchments (Hoffmann) and CBS NL stay
independent. `size_measure: footprint` keeps them independent.
