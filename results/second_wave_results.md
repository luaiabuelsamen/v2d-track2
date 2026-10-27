# Second wave, as it lands

Written by `brev/second_wave_puller.sh` as each clip finishes, from
`v2d adopt-check` (`results/second_wave_adoption.json`). The **adopt** column is the
registered rule and reads only the noisy input reference; the kit columns are Tier 1
truth and are reported, never used to decide.

6 of 28 clips landed, 6 kit-scored, 4 adopted.

| clip | arm | record AUC | new AUC | Δ AUC | MP-SR rec → new | input dist cm, old → new | adopt |
|---|---|---|---|---|---|---|---|
| e01 | B5 | 0.2001 | 0.0576 | -0.1425 | fail → fail | 44.07 → 45.90 | **keep** |
| e02 | A | 0.4263 | 0.3237 | -0.1026 | fail → fail | 4.36 → 5.75 | **keep** |
| e03 | A | 0.6589 | 0.7806 | 0.1218 | fail → pass | 3.58 → 1.63 | **adopt** |
| e04 | B1 | 0.1472 | 0.1466 | -0.0006 | fail → fail | 116.55 → 102.25 | **adopt** |
| e08 | A | 0.4431 | 0.6654 | 0.2223 | fail → pass | 3.91 → 2.08 | **adopt** |
| e17 | B4 | 0.2466 | 0.4470 | 0.2004 | fail → fail | 20.25 → 4.73 | **adopt** |

## The pre-registered bars

- **Part A** (3/18 scored): MP-SR passes on **2**, bar is 9 (was 5); clips more than 0.05 AUC below their record: e02
- **Part B**, per clip against its own record:
  - e01: bar 0.200, new 0.0576 (below)
  - e04: bar 0.147, new 0.1466 (below)
  - e05: bar 0.266, new — (waiting)
  - e07: bar 0.251, new — (waiting)
  - e09: bar 0.245, new — (waiting)
  - e14: bar 0.109, new — (waiting)
  - e17: bar 0.247, new 0.4470 (beats)
  - e19: bar 0.116, new — (waiting)
  - e25: bar 0.134, new — (waiting)
  - e28: bar 0.089, new — (waiting)

Still training or not yet selected — A: e00 e06 e10 e11 e12; B1: e25; B2: e14; B3: e05.
