# STATUS — v2d (NVIDIA V2D Challenge scoping, $0)
Goal: Decide go/no-go on V2D Track 2 (Robotic Grounding, G1 + Dex3) with a costed 4-week plan.
Finish line: docs/SCOPING.md committed covering requirements, dataset sizes, compute path + $ estimate, Jetson/Mac feasibility, 4-week plan, go/no-go.
Deadline: ~2026-10-07 (leaderboard freeze 2026-11-04 17:00 EST)
Milestone: 6/6 sections done; docs/SCOPING.md committed. Scoping complete.
Last result: Tier 2 input tracked perfectly scores SP-SR 0.97 / MP-SR 1.00 / MPPE 2.36 cm vs clean (leader's Tier 2 policy: 0.53 / 0.27 / 15.3); corruption is slow offset + rotation random walk clipped at 30°, smoothing does not help (results/tier2_input_ceiling.json, results/tier2_rest_snap.json)
Blocked: needs Luai decision: Conditional GO requires approving a $300 cap on rented 48 GB GPUs (A6000/L40S) for Tiers 1-2; no registration, submission or organizer questions sent (5 open questions in SCOPING.md §7)
Next: Stop. If Luai approves: Week 1 $0 Jetson tasks (local scorer, mesh support-plane test, soma_to_g1 on CPU), then Day-4 2-clip FlashCHORD gate
Updated: 2026-10-06 23:40

## Rules
No paid compute, no registration, no submission, no posting. No large downloads.

## Paid compute used
None. Downloads: kit (0.7 MB), FAQ, HF metadata, Track 2 tier 1/2 trajectory parquets (60 MB, scratchpad only, not committed).

- 2026-10-07: Luai emailed v2d_challenge@nvidia.com about GPU credits and the scoring/multi-motion questions. Awaiting reply; paid compute stays off.
