# Final checklist, mapped to what we actually have

*The organisers' final-stretch email, 2026-11-01. Freeze: **2026-11-04 17:00 EST = 22:00 UTC**,
about three days.* Every line below is either quoted from that email or checked in this repo.

## What changed in our favour

**"Submissions are now unlimited through the leaderboard freeze."** The written 5-per-week
limit is gone, and with it the one compliance worry that had no clean answer (Kaggle enforced
5/day while the rules said 5/week, so any burst violated the written rule). There is now no
reason to hold an upload back, and on Tier 1 — where the permitted input *is* the scoring
reference, so choosing by closeness to it reads nothing forbidden — repeated uploads are free.

## What changed against us: eligibility is now a form, not just a score

> "Each team must submit one technical report submission form per top submission, including a
> two-page technical report before the leaderboard closes. The google form asks for the report,
> Github links and links to your container. **A completed form is required for your entry to be
> valid** and eligible for first-place, second-place, or Industry Innovation Awards."

Three fields, and we can currently fill one of them.

| field | state | what it needs |
|---|---|---|
| two-page report PDF | **partial** — `docs/TECHNICAL_REPORT.md` is 2,202 words over ten sections, roughly double two pages, and is not in their required shape (Introduction, Approach, Experiments & Results, Conclusion) | free: restructure and compress to their template, export PDF |
| GitHub links | **exists but unopenable** — `https://github.com/luaiabuelsamen/v2d-track2` is private, and the `Nvidia-V2D-Challenge` invitation (340814066, re-sent 2026-11-01, expires Nov 8) is **unaccepted** | Luai: either confirm acceptance or make the repo public. Prizes require an Apache-2.0 release anyway (`docs/SCOPING.md:44`), and our `LICENSE` is already Apache-2.0 |
| container link | **does not exist** — the FlashCHORD image was lost in the 2026-10-28 reboot; `docker images` on this box holds only `jekyll-site` and `ursim_polyscopex` | a GPU box to rebuild and somewhere to host it. **Not free, and nothing else on the list matters without it** |

Nineteen uploads are currently unbacked by a container. This outranks every rank point on any
queue we hold.

## The three content rules, checked against our entries

**1. "Object trajectories must result from a policy rollout, not reconstructions or reference
trajectories."** — Clean. Every submitted row is an evaluator rollout with a real
`checkpoint_sha256`, which is exactly why `results/t3_reconstruction.json` refused to upload the
raw Tier 3 reconstruction ("a reconstruction is not an evaluator run, so there is no honest value
for `eval_code_sha256`").

**2. "Submissions for Tiers 2 and 3 must use the appropriate reference trajectory for each
tier."** — Our Tier 2 entry is defensible, and now has nothing to argue about:

- the cleaner reads `object_pose(dataset_root, "tier2", ...)` and no other trajectory
  (`src/v2d/clean/support_contact.py:294`);
- the per-clip decision to clean or pass through is our **own synthetic recovery gate**, built
  from noise we generated from a seed, never from truth (`src/v2d/clean/gates.py` docstring: the
  clean trajectory "is never an input or a target for anything that runs on Tier 2");
- selection across checkpoints uses the input-only z-drop rule, not truth;
- it *used* to take the static support-surface geometry from the `tier_1` directory. Both trees
  ship byte-identical support USDAs for all 30 clips (verified), so it now reads the Tier 2 copy
  — **no number changes**, and the Tier 2 path reads nothing from `tier_1`. Pinned by
  `tests/test_tier2_reads_no_tier1.py`.

A cleaned reference is still a *function of the Tier 2 release*. If the organisers mean "the
released file, unmodified", that reading would invalidate our Tier 2 entry — see the questions
below.

**3. "Track 3 — no ground-truth meshes."** Scoped to **Track 3**; we are Track 2, and our Track 2
Tier 3 pipeline does use the released mesh for FoundationPose and for AUC vertices. Flagged as a
question, not a finding, because the adjacent wording ("bring your own reconstruction") could be
read to carry the same intent.

## What this kills, and it is ours

> "Submissions that do not adhere to these rules will be invalidated, and continued attempts to
> circumvent challenge rules may result in disqualification. **Participants are responsible for
> the contents of their submissions, even if produced with agentic pipelines.**"

**The e01/e29 cross-tier substitution is shelved.** Putting Tier-3-driven rollouts into the Tier 1
entry was worth **two rank points and no change of place** (MPPE 11 → 9, Tier 1 sum 37 → 35), and
its legality was an *inference* — Tier 1 is not named in rule 2, and a video-driven rollout uses
strictly less information than Tier 1 permits, but nothing states it is permitted. Against a
sentence that names invalidation and disqualification, and that addresses agentic pipelines
directly, two rank points that do not move our place is a bad trade. It stays unshipped unless an
organiser says yes.

The same inference sits under **16 of 30 clips in our live Tier 1 entry**, which are wave-2
rollouts driven by the bridge-cleaned Tier 2 reference rather than the clean Tier 1 trajectory.
That is already uploaded. Rule 2 names only Tiers 2 and 3, and driving a Tier 1 entry from a
*noisier* reference is a handicap rather than an advantage, so the literal reading is fine — but
it is the same question, it is live, and it is worth asking rather than hoping.

## Questions for v2d_challenge@nvidia.com — one email, worth sending today

1. For Tiers 2 and 3, does "the appropriate reference trajectory" mean the released file
   unmodified, or may a submission drive its policy from a reference *derived only from* that
   tier's released file (smoothing, support-plane snapping), with no read of any other tier?
2. For Tier 1, is a policy rollout driven by a reference derived from the Tier 2 release — or
   from our own video reconstruction — a valid Tier 1 submission, given both use strictly less
   information than Tier 1 permits?
3. Does the Track 3 ground-truth-mesh restriction apply to Track 2 Tier 3?
4. For the final metric ranking, do teams on an equal score **share** a rank, as the public
   aggregator does (`leaderboard.js:74`), or are ties ordered by submission time?

Questions 1 and 2 decide whether entries we have already uploaded stand. Question 4 decides
whether several purchases on our queue are worth anything at all (`docs/RANKING_RULE.md`).

## Order of work, revised

1. **Container.** Rebuild and host it. Needs a GPU box and money. Without it the form cannot be
   completed and no entry is valid.
2. **The report.** Restructure to their four sections at two pages, PDF. Free, and required.
3. **GitHub access.** Luai's call: confirm NVIDIA accepted, or make it public.
4. **The email above.** Free, and it gates both the shelved move and the live Tier 1 entry.
5. Only then, score work — and under unlimited submissions, on Tier 2 and Tier 3, where a place
   is actually reachable (`docs/RANKING_RULE.md`).

The CoRL invitation (~10 talks, 20 posters, Nov 12) follows from the above; it is not a separate
task.

---

# Addendum, 2026-11-02 21:42 UTC: the organisers answered

Shalin Jain (NVIDIA), replying to the four questions above within 21 minutes, cc
`v2d_challenge@nvidia.com` and `mverghese@nvidia.com`. Gmail thread `1a19e23934b631fa`. Quoted in
full because three of our entries' legality rests on it:

> 1. Using derived from the tier's own released file is fine, with no read of any other tier's data.
> 2. Yes, using strictly less information than Tier 1 permits is okay!
> 3. For Tier 3, we ask that you do not use the ground truth mesh for training as the intent is to
>    evaluate directly from video, the hardest setting for this track. You may use the ground truth
>    mesh for metric computations and frame alignment.
> 4. All tied teams receive the same rank, and the next rank skips appropriately, there will not be
>    consideration of submission time.

## What this settles in our favour

**(1) The Tier 2 entry stands.** Our cleaner reads only the Tier 2 object trajectory and the Tier 2
support primitives, which is exactly the permitted reading. The audit above —
`src/v2d/clean/support_contact.py:294`, the synthetic recovery gate, and
`tests/test_tier2_reads_no_tier1.py` — was not wasted work; it is the evidence that we meet the
condition as stated.

**(2) The live Tier 1 entry stands, and the shelved move is legal.** 16 of 30 clips in the uploaded
Tier 1 entry are wave-2 rollouts driven by the bridge-cleaned Tier 2 reference. That is strictly
less information than Tier 1 permits, and it is now confirmed as valid. The same sentence legalises
the **e01/e29 cross-tier substitution** that this document shelved, so it is un-shelved as a rules
matter — though the fresh boards have since re-priced it (see below).

**(4) Ties share a rank, in the final ranking and not just on the public page.** `docs/RANKING_RULE.md`
no longer contains an inference.

## What this breaks, and it is ours

**(3) rules out our Tier 3 entry as built.** The released object mesh enters our Track 2 Tier 3
pipeline in three places, and only the first is covered by "metric computations and frame
alignment":

| use | where | covered by the answer? |
|---|---|---|
| AUC vertices, and the single frame-0 rigid transform | `src/v2d/score/track2.py:170`, FAQ PR 207 | **yes**, explicitly |
| the CAD model input to FoundationPose, which produces our per-frame object track from video | `docs/T3_RECON.md:53` | **no** — it is neither metric computation nor frame alignment |
| the object's visual and collision geometry in the simulator **while the policy trains** | `src/v2d/retarget/sequences.py:270`, copied to `reconstructed_mesh/output_aligned.glb` | **no** — this is literally "the ground truth mesh for training" |

The second row is the one we disclosed in the original email and the one Shalin's answer was written
against. The third is worse and we had not drawn attention to it: the directory is *named*
`reconstructed_mesh`, and what it holds is the released ground-truth mesh. Under the stated intent —
"evaluate directly from video, the hardest setting" — an object whose exact shape and scale came
from the dataset is not a reconstruction.

**A precise follow-up was sent at 2026-11-02 21:5x UTC** (same thread), naming all three uses
separately, asking whether (b) and (c) are permitted, and stating that if they are not we will
either rebuild Tier 3 on our own reconstruction or withdraw the tier before the freeze. It also
offers the object *name* (from `meta/episodes_metadata.jsonl`, used only as the detection prompt)
for the same scrutiny.

### Consequences, pending that answer

- **No credit goes to Tier 3 until it lands.** The staged 13-clip Tier 3 warm-start batch
  (`$S/ws_t3n`, 449 MB, manifest verified) is **held**, not launched. Fine-tuning a policy on a
  reference that may not be admissible spends money on an entry that may be withdrawn.
- **If (b) and (c) are disallowed, the compliant route exists and is NVIDIA's own:** the upstream
  repo ships `v2d_bundlesdf` and `v2d_hoi_object_reconstruction`, which reconstruct object geometry
  from the video itself. That mesh would then feed both the tracker and the simulator, with the
  released mesh kept only for scoring and frame-0 alignment. It is a rebuild of the whole Tier 3
  chain, not a patch, and it is not obviously affordable in the time and credit left.
- **Tier 1 and Tier 2 do not depend on this question at all.** Both were confirmed by answers (1)
  and (2), and neither reads a reconstruction.

## Where the cheap rungs actually are now (fresh boards, 2026-11-02, `--page-size 200`)

Standings: **Tier 1 12/24** (sum 43, tied with Team OPS), **Tier 2 7/18** (sum 22), **Tier 3 6/12**
(sum 21). Priced per board as the distance to the next *distinct* score, in units we can produce:

| board | +1 rank needs | +2 ranks needs |
|---|---|---|
| **tier1-mp-sr** | **1 clip** under the 3 cm / 30 deg gate | 2 clips |
| **tier1-mppe** | **+1.9 cm summed** over 30 clips (0.063 cm of mean) | +63 cm summed |
| tier1-auc | +1.03 summed AUC | +2.46 summed |
| tier1-sp-sr | 3 clips | 4 clips |
| tier2-auc | +0.169 summed AUC | +1.46 summed |
| tier2-sp-sr / mp-sr | 2 clips each | 4 clips each |
| tier3-sp-sr / mp-sr | 1 clip each | 4 / 2 clips |
| tier3-auc | +0.364 summed AUC | +0.418 summed |

Three of the four cheapest rungs on the whole board are **Tier 1**, the tier whose legality answer
(2) just confirmed without reservation, and the tier where selection carries no adoption risk
because the permitted input *is* the scoring reference — so a better rollout can simply be kept.
Seven Tier 1 clips sit between 3.85 and 5.15 cm of the 3 cm MP-SR gate (e29 3.85, e11 3.97, e02
4.57, e21 4.79, e09 4.89, e13 4.98, e17 5.15), and `results/t1_rank_sensitivity.json` records that
the warm start cannot collect them: 12 of 12 rejected clips came back *worse* on MPPE, because that
arm trades position accuracy for reference tracking. The centimetre needs a policy trained on the
clean Tier 1 reference at full length, which we have never run.
