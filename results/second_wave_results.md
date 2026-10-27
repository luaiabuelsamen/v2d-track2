# Second wave, as it lands

Written by `brev/second_wave_puller.sh` as each clip finishes, from
`v2d adopt-check` (`results/second_wave_adoption.json`). The **adopt** column is the
registered rule and reads only the noisy input reference; the kit columns are Tier 1
truth and are reported, never used to decide.

11 of 28 clips landed, 11 kit-scored, 8 adopted.

| clip | arm | record AUC | new AUC | Δ AUC | MP-SR rec → new | input dist cm, old → new | adopt |
|---|---|---|---|---|---|---|---|
| e00 | A | 0.4983 | 0.5178 | 0.0194 | pass → pass | 3.81 → 2.35 | **adopt** |
| e01 | B5 | 0.2001 | 0.0576 | -0.1425 | fail → fail | 44.07 → 45.90 | **keep** |
| e02 | A | 0.4263 | 0.3237 | -0.1026 | fail → fail | 4.36 → 5.75 | **keep** |
| e03 | A | 0.6589 | 0.7806 | 0.1218 | fail → pass | 3.58 → 1.63 | **adopt** |
| e04 | B1 | 0.1472 | 0.1466 | -0.0006 | fail → fail | 116.55 → 102.25 | **adopt** |
| e05 | B3 | 0.2660 | 0.3418 | 0.0758 | fail → fail | 14.95 → 20.63 | **keep** |
| e06 | A | 0.4730 | 0.4758 | 0.0029 | fail → pass | 2.50 → 2.08 | **adopt** |
| e08 | A | 0.4431 | 0.6654 | 0.2223 | fail → pass | 3.91 → 2.08 | **adopt** |
| e10 | A | 0.3147 | 0.4794 | 0.1647 | fail → pass | 4.22 → 2.72 | **adopt** |
| e14 | B2 | 0.1091 | 0.2055 | 0.0964 | fail → fail | 47.49 → 5.91 | **adopt** |
| e17 | B4 | 0.2466 | 0.4470 | 0.2004 | fail → fail | 20.25 → 4.73 | **adopt** |

## The pre-registered bars

- **Part A** (6/18 scored): MP-SR passes on **5**, bar is 9 (was 5); clips more than 0.05 AUC below their record: e02
- **Part B**, per clip against its own record:
  - e01: bar 0.200, new 0.0576 (below)
  - e04: bar 0.147, new 0.1466 (below)
  - e05: bar 0.266, new 0.3418 (beats)
  - e07: bar 0.251, new — (waiting)
  - e09: bar 0.245, new — (waiting)
  - e14: bar 0.109, new 0.2055 (beats)
  - e17: bar 0.247, new 0.4470 (beats)
  - e19: bar 0.116, new — (waiting)
  - e25: bar 0.134, new — (waiting)
  - e28: bar 0.089, new — (waiting)

Still training or not yet selected — A: e11 e12 e15 e16 e18 e20; B1: e25; B2: e19; B3: e07.
