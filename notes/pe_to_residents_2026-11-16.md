# UWWTD loads (p.e.) to residents for the per-plant validation (2026-11-16)

Why: in Europe the reported plant number is the UWWTD load entering the plant in population
equivalents (1 p.e. = 60 g BOD5 a day). Our model counts residents. Before the conversion the
median plant had ours / load = 0.67 and a typical error of x1.63 across Europe; most of that was the
p.e. surplus, not our error.

## What the literature says

- A resident alone makes less than 1 p.e.: 44.6 +- 5.9 g BOD5 per person a day in 168 Italian
  agglomerations (Sci. Total Environ. 2022, S0048969722003266), i.e. about 0.75 p.e.; 60 g is a
  70th-90th percentile design value (safety factor about 1.3).
- Loads also count industry, commerce, commuters and tourists (HydroWASTE, Ehalt Macedo et al. 2022,
  ESSD 14, 559: European p.e. "not only residents but also workers, tourists, and service providers";
  HydroWASTE used the p.e. as population served without conversion).
- National ratios: EU generated load 533.8 M p.e. (UWWTD 2022 reporting, water.europa.eu) for about
  447 M inhabitants, i.e. about 1.2 p.e. per inhabitant; Germany (DWA Leistungsvergleich 2023):
  101.4 M p.e. mean load in the 5,121 participating plants (not all) for about 81 M connected
  residents, i.e. at least about 1.25.
- Per plant the ratio varies a lot: BOD-based p.e. against census population in four English
  catchments -11 % to +116 % (ES&T 2025, PMC12676738; agricultural/industrial inputs).

## Method (validation_figures.pe_conversion)

f_c = (sum of the UWWTD loads of a country's plants) / (residents connected to treatment: people x
share of Eurostat, else OECD, else JMP). Each load is divided by f_c. A country below 0.75 (what
residents alone make) misses loads (Ireland: Dublin's Ringsend has no load in the layer; Cyprus) and,
like a country without a national number (Iceland, Malta), takes the ratio of all others (1.27).

f_c: DE 1.30, FR 1.29, IT 1.59, ES 1.40, AT 1.60, PT 1.38 (JMP), NL 1.04 (the NL note found UWWTD
load / CBS inhabitants median 1.02), PL 1.08, CH 1.16, BE 0.82, CZ 0.87 (pe_per_resident.csv).

## Result (recommended candidate, UWWTD plants)

| | before | after |
|---|---|---|
| Europe, typical error | x1.63 | x1.50 |
| Europe, median ours / reference | 0.67 | 0.86 |
| Germany | x1.64 | x1.40 |
| Austria | x2.26 | x1.58 |
| Spain | x2.04 | x1.67 |
| France | x1.50 | x1.48 |
| Netherlands | x1.36 | x1.33 (real catchments x1.30) |

Size classes: under 10,000 x1.54, 10,000-100,000 x1.43, over 100,000 x1.39. Still high: Greece,
Norway, Finland (x1.6-1.9, ours 0.55-0.64 of the reference): our national totals there are also
below Eurostat (0.64-0.70), so this is our under-coverage; plus tourist plants in Greece. The rest
(about x1.4-1.5) is the plant-to-plant spread of p.e. per resident and our own error.
