# Second wave, as it lands

Written by `brev/second_wave_puller.sh` as each clip finishes, from
`v2d adopt-check` (`results/second_wave_adoption.json`). The **verdict** column is the
registered rule and reads only the noisy input reference; the kit columns are Tier 1
truth and are reported, never used to decide. A clip two arms re-ran has one row per arm,
each judged against the incumbent at that moment, and **ships** marks the winner.

20 clips landed of 36 expected, 24 candidate rollouts, 17 clips replaced (A 11, B1 2, B2 2, B4 1, B4x 1).

| clip | arm | record AUC | new AUC | Δ AUC | MP-SR | input dist cm, incumbent → new | verdict | ships |
|---|---|---|---|---|---|---|---|---|
| e00 | A | 0.4983 | 0.5178 | 0.0194 | pass | 3.81 → 2.35 | adopt | **ships** |
| e00 | B4x | 0.4983 | 0.5190 | 0.0206 | fail | 2.35 → 3.77 | keep |  |
| e01 | B5 | 0.2001 | 0.0576 | -0.1425 | fail | 44.07 → 45.90 | keep |  |
| e01 | — | 0.2001 | — | — | — | — | — | **ships (board)** |
| e02 | A | 0.4263 | 0.3237 | -0.1026 | fail | 4.36 → 5.75 | keep |  |
| e02 | — | 0.4263 | — | — | — | — | — | **ships (board)** |
| e03 | A | 0.6589 | 0.7806 | 0.1218 | pass | 3.58 → 1.63 | adopt | **ships** |
| e04 | B1 | 0.1472 | 0.1466 | -0.0006 | fail | 116.55 → 102.25 | adopt | **ships** |
| e05 | B3 | 0.2660 | 0.3418 | 0.0758 | fail | 14.95 → 20.63 | keep |  |
| e05 | — | 0.2660 | — | — | — | — | — | **ships (board)** |
| e06 | A | 0.4730 | 0.4758 | 0.0029 | pass | 2.50 → 2.08 | adopt | **ships** |
| e06 | B4x | 0.4730 | 0.6332 | 0.1603 | fail | 2.08 → 3.35 | keep |  |
| e08 | A | 0.4431 | 0.6654 | 0.2223 | pass | 3.91 → 2.08 | adopt | **ships** |
| e08 | B4x | 0.4431 | 0.6020 | 0.1588 | pass | 2.08 → 3.06 | keep |  |
| e09 | B4x | 0.2448 | 0.3003 | 0.0555 | fail | 5.08 → 4.47 | adopt | **ships** |
| e10 | A | 0.3147 | 0.4794 | 0.1647 | pass | 4.22 → 2.72 | adopt | **ships** |
| e10 | B4x | 0.3147 | 0.3835 | 0.0688 | fail | 2.72 → 4.47 | keep |  |
| e11 | A | 0.3315 | 0.4676 | 0.1360 | fail | 5.59 → 3.90 | adopt | **ships** |
| e12 | A | 0.7097 | 0.7904 | 0.0806 | pass | 2.75 → 2.15 | adopt | **ships** |
| e14 | B2 | 0.1091 | 0.2055 | 0.0964 | fail | 47.49 → 5.91 | adopt | **ships** |
| e15 | A | 0.4538 | 0.6653 | 0.2115 | pass | 3.87 → 2.52 | adopt | **ships** |
| e16 | A | 0.3890 | 0.5639 | 0.1749 | fail | 4.63 → 2.71 | adopt | **ships** |
| e17 | B4 | 0.2466 | 0.4470 | 0.2004 | fail | 20.25 → 4.73 | adopt | **ships** |
| e18 | A | 0.7362 | 0.7385 | 0.0023 | pass | 3.24 → 2.54 | adopt | **ships** |
| e19 | B2 | 0.1163 | 0.1163 | 0.0000 | fail | 57.16 → 57.16 | adopt | **ships** |
| e20 | A | 0.4729 | 0.7149 | 0.2420 | pass | 3.22 → 2.52 | adopt | **ships** |
| e25 | B1 | 0.1335 | 0.1334 | -0.0001 | fail | 82.99 → 74.11 | adopt | **ships** |

## The pre-registered bars

- **Part A** (12/18 scored): MP-SR passes on **9**, bar is 9 (was 5); clips more than 0.05 AUC below their record: e02
- **Part B**, per clip against its own record:
  - e01 [B5]: bar 0.200, new 0.0576 (below)
  - e04 [B1]: bar 0.147, new 0.1466 (below)
  - e05 [B3]: bar 0.266, new 0.3418 (beats)
  - e07: bar 0.251, new — (waiting)
  - e09: bar 0.245, new — (waiting)
  - e14 [B2]: bar 0.109, new 0.2055 (beats)
  - e17 [B4]: bar 0.247, new 0.4470 (beats)
  - e19 [B2]: bar 0.116, new 0.1163 (beats)
  - e25 [B1]: bar 0.134, new 0.1334 (below)
  - e28: bar 0.089, new — (waiting)

- **B4x**, per clip against the best rollout held when it was judged — not the first-wave record, because six of its clips are Part A clips:
  - e00: bar 0.5178 (from A), new 0.5190 (beats)
  - e06: bar 0.4758 (from A), new 0.6332 (beats)
  - e08: bar 0.6654 (from A), new 0.6020 (below)
  - e09: bar 0.2448 (from first wave), new 0.3003 (beats)
  - e10: bar 0.4794 (from A), new 0.3835 (below)

Mean kit AUC of the shipping rollout over the clips landed so far: **0.4551** (not comparable to the 30-clip 0.3908 until every clip is in).

Still training or not yet selected — A: e22 e23 e24 e26 e27 e29; B2: e28; B3: e07; B4x: e13.
