# The technical-report form, field by field

Form: <https://docs.google.com/forms/d/e/1FAIpQLSekBx0IFs7Q_TVMu67UXNrh0S7FID6yIFGRyN1n6u4Xh2QAkw/viewform>
(from the organisers' 2026-11-01 email). Deadline **2026-11-04 17:00 EST / 22:00 UTC**.

> "Each team must submit one technical report submission form **per top submission** … A completed
> form is required for your entry to be valid and eligible for first-place, second-place, or
> Industry Innovation Awards."

## It is one form, not three

This page previously planned three filings, one per tier, reading "per top submission" as per
tier. **The live form says otherwise and is the authority.** Field 4 is a single-choice dropdown
whose options are whole tracks --- "Track 1 — Reconstruction", "Track 2 — Robotic Grounding",
"Track 3 — Egocentric" --- and field 8 says, in the organisers' own words, *"For Track 2, state
'All tiers' or list a commit for each tier."* A form that asks for one track and all of its tiers
in one answer is one form. Filing three would have produced three conflicting records of the same
entry.

Two other things the form wants that the plan had wrong:

* Field 6 asks for a **URL to the report**, not a file upload. The PDF has to be hosted somewhere a
  judge can open, which is why the repository had to be pushed before this could be filled in.
* Field 8 wants the **exact commit**, per tier if they differ. Ours differ, because each tier's CSV
  embeds the commit it was packed at in all 22,989 of its rows.

## The nine fields

| # | field | answer |
|---|---|---|
| 1 | Registered team name *(must match leaderboard and registration)* | `Luai Abuelsamen` |
| 2 | Contact email | `luai_abuelsamen@berkeley.edu` |
| 3 | Leaderboard team ID *(optional)* | `17198357` (Tier 1 AUC). Each metric board is its own Kaggle competition and issues its own team ID; the full list is below |
| 4 | Challenge track | **Track 2 — Robotic Grounding** |
| 5 | Final leaderboard submission IDs or URLs | the eight below — Tier 1 and Tier 2 only |
| 6 | Technical report URL | <https://github.com/luaiabuelsamen/v2d-track2/blob/f4e4e9c44fa8e9eda7fc8096e84a6b6bfd430192/docs/TECHNICAL_REPORT.pdf> |
| 7 | GitHub repository URL | <https://github.com/luaiabuelsamen/v2d-track2> (public, Apache-2.0) |
| 8 | Code version used for this submission | `Tier 1: 070eab9170d19d57e0b79e50d80fd3d36b1887f2 · Tier 2: f4e4e9c44fa8e9eda7fc8096e84a6b6bfd430192` — each tier's CSV carries its own commit in every row |
| 9 | Container for verification | `ghcr.io/luaiabuelsamen/v2d-track2:submission`, digest `sha256:f3f3d24bbf333a0e2e0323dda8968810ba18e68cc92e9c7da05d32ba0b7c7af1` (public, anonymous pull verified below) |

### Field 5, in full

Eight submissions across two tiers. Tier 3 is **deliberately absent**; see below.

| board | Kaggle competition | submission | score |
|---|---|---|---|
| Tier 1 AUC | `v2d-challenge-track2-tier1-auc` | 57959749 | 0.60459 |
| Tier 1 SP-SR | `v2d-challenge-track2-tier1-sp-sr` | 57959751 | 0.73333 |
| Tier 1 MP-SR | `v2d-challenge-track2-tier1-mp-sr` | 57959755 | 0.63333 |
| Tier 1 MPPE | `v2d-challenge-track2-tier1-mppe` | 57959758 | 11.72123 |
| Tier 2 AUC | `v2d-challenge-track2-tier2-auc` | 57996327 | 0.47125 |
| Tier 2 SP-SR | `v2d-challenge-track2-tier2-sp-sr` | 57996328 | 0.70000 |
| Tier 2 MP-SR | `v2d-challenge-track2-tier2-mp-sr` | 57996329 | 0.46666 |
| Tier 2 MPPE | `v2d-challenge-track2-tier2-mppe` | 57996331 | 12.36556 |

Each is the board's own current best and its latest, checked against the leaderboard rather than
assumed. The four Tier 2 IDs were **re-filed on 2026-11-04**: the contact arm's one adopted clip
(e19) took Tier 2 to 0.47125 AUC at 12.36556 cm, which superseded 57742766/68/70/73. See
`results/t2_submission_cr.json` and `results/cr_adoption.json`.

### Field 3, in full

| tier | auc | sp-sr | mp-sr | mppe |
|---|---|---|---|---|
| Tier 1 | 17198357 | 17198361 | 17198362 | 17198363 |
| Tier 2 | 17197340 | 17198366 | 17198367 | 17198369 |

## Tier 3 is withdrawn, and that is the honest answer

**Do not list the Tier 3 submissions.** On 2026-11-03 the organisers answered our follow-up:

> "Your read is correct, we ask that you please reconstruct the object mesh for your Tier 3 object
> tracking and simulation efforts. For the object name, you're welcome to use AI-generated subject
> names as an alternative."

The read they confirmed was ours: that using the released mesh as FoundationPose's CAD input **and**
as the simulator's object geometry are both inadmissible. Our Tier 3 entry does both, in all 30
clips, and additionally takes the object's name from `meta/episodes_metadata.jsonl` --- which the
second sentence also rules out. Three independent reasons, none of them borderline.

Rebuilding it compliantly means 30 object reconstructions from video, FoundationPose re-run on each,
the simulator re-staged on reconstructed geometry and 30 policies re-trained: a chain we have never
run end to end, against ~$28 of credit and ~25 hours. So the tier is withdrawn rather than fixed.

The submissions cannot be deleted from Kaggle, so 0.26781 stays visible on the Tier 3 board. Not
naming it on the form is how the tier is declined --- and the organisers are told so directly in the
same thread, rather than left to infer it from an absence. We asked the question that disqualified
our own entry; the cost of that is one tier, and the alternative was claiming an award on an entry
we had been told does not adhere.

## The container is public, and that is checked rather than assumed

It was flipped in the UI on 2026-11-02 (GitHub has no REST endpoint for package visibility --- a
`PATCH` to `/user/packages/container/v2d-track2` returns 404 --- so it could only ever be a UI
click). An anonymous pull now resolves:

```
TOK=$(curl -s "https://ghcr.io/token?scope=repository:luaiabuelsamen/v2d-track2:pull&service=ghcr.io" | jq -r .token)
curl -sI -H "Authorization: Bearer $TOK" https://ghcr.io/v2/luaiabuelsamen/v2d-track2/manifests/submission
# HTTP 200, docker-content-digest: sha256:f3f3d24bbf333a0e2e0323dda8968810ba18e68cc92e9c7da05d32ba0b7c7af1
```

That digest is the one we pushed, so a judge with no credentials gets the same image we built, and
"private package" is ruled out as a failure mode rather than hoped against.

Field 7 adds a condition the earlier plan missed: *"Repository shared with the Nvidia-V2D-Challenge
account."* The repository is public, so no invitation is needed for a judge to read it; if they
want the account listed as a collaborator explicitly, that is one click in Settings → Collaborators.

## Standings at filing time

`results/standings_tiers_2026_11_03.json`, all twelve boards pulled at `--page-size 200`:

| tier | AUC | SP-SR | MP-SR | MPPE cm | rank sum | place |
|---|---|---|---|---|---|---|
| Tier 1 | 0.60459 | 0.73333 | 0.63333 | 11.721 | 39 | 12 / 24 |
| Tier 2 | 0.47125 | 0.70000 | 0.46666 | 12.366 | 30 | 10 / 25 |
| ~~Tier 3~~ | ~~0.26781~~ | ~~0.20000~~ | ~~0.03333~~ | ~~47.772~~ | ~~23~~ | withdrawn |

Tier 2's place is worse than yesterday's 8th of 18 while every one of its numbers is better,
because the tier grew from 18 ranked teams to 24 overnight and three of the new ones are above us.
Measured against the same 2026-11-04 board, the e19 upload is worth +2 ranks on MPPE and -2 on the
rank sum (32 -> 30); without it we would be 10th with a rank sum of 32. The field moved, we did
not slip.

**The template.** The email attaches `V2D_Challenge_Extended_Abstract_Template.zip`: a LaTeX class,
not a Word file, so the report is written in it rather than approximated. The limit is two pages of
*main text* --- `nvidiaextabstract.cls:9`, "The two-page limit excludes references, acknowledgments
and appendices" --- and the class ships its own checker, which is silent on our PDF. The body ends
on page 2; page 3 holds nothing but reference entries.
