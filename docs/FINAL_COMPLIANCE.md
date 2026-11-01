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
