# Second wave, as it lands

Written by `brev/second_wave_puller.sh` as each clip finishes, from
`v2d adopt-check` (`results/second_wave_adoption.json`). The **adopt** column is the
registered rule and reads only the noisy input reference; the kit columns are Tier 1
truth and are reported, never used to decide.

1 of 28 clips landed, 1 kit-scored, 0 adopted.

| clip | arm | record AUC | new AUC | Δ AUC | MP-SR rec → new | input dist cm, old → new | adopt |
|---|---|---|---|---|---|---|---|
| e01 | B5 | 0.2001 | 0.0576 | -0.1425 | fail → fail | 44.07 → 45.90 | **keep** |

## The pre-registered bars

- **Part A** (0/18 scored): MP-SR passes on **0**, bar is 9 (was 5); clips more than 0.05 AUC below their record: none
- **Part B**, per clip against its own record:
  - e01: bar 0.200, new 0.0576 (below)
  - e04: bar 0.147, new — (waiting)
  - e05: bar 0.266, new — (waiting)
  - e07: bar 0.251, new — (waiting)
  - e09: bar 0.245, new — (waiting)
  - e14: bar 0.109, new — (waiting)
  - e17: bar 0.247, new — (waiting)
  - e19: bar 0.116, new — (waiting)
  - e25: bar 0.134, new — (waiting)
  - e28: bar 0.089, new — (waiting)

Still training or not yet selected — A: e00 e02 e03 e06 e08 e10; B1: e04; B2: e14; B3: e05; B4: e17.
