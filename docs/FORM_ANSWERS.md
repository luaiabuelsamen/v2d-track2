# The technical-report form, field by field

Form: <https://docs.google.com/forms/d/e/1FAIpQLSekBx0IFs7Q_TVMu67UXNrh0S7FID6yIFGRyN1n6u4Xh2QAkw/viewform>
(from the organisers' 2026-11-01 email). Deadline **2026-11-04 17:00 EST / 22:00 UTC**.

> "Each team must submit one technical report submission form **per top submission** … A completed
> form is required for your entry to be valid and eligible for first-place, second-place, or
> Industry Innovation Awards."

"Per top submission" is why this is a table of three rather than one: we have a live entry on
Tier 1, Tier 2 and Tier 3, and an unfilled form is the difference between an entry being judged
and an entry being void. Submitting three costs nothing but minutes.

| field | value |
|---|---|
| team | Luai Abuelsamen, University of California, Berkeley |
| report PDF | `docs/TECHNICAL_REPORT.pdf` — built from `docs/report/main.tex` in NVIDIA's own extended-abstract class; main text ends on page 2, references follow |
| GitHub | <https://github.com/luaiabuelsamen/v2d-track2> (public, Apache-2.0) |
| container | `ghcr.io/luaiabuelsamen/v2d-track2:submission` — pushed 2026-11-02, digest `sha256:f3f3d24bbf333a0e2e0323dda8968810ba18e68cc92e9c7da05d32ba0b7c7af1`, 23 layers, 7.51 GB. **Public and verified** (see below) |

**The package is public, and that is checked rather than assumed.** It was flipped in the UI on
2026-11-02 (GitHub has no REST endpoint for package visibility --- a `PATCH` to
`/user/packages/container/v2d-track2` returns 404 --- so it could only ever be a UI click). An
anonymous pull now resolves:

```
TOK=$(curl -s "https://ghcr.io/token?scope=repository:luaiabuelsamen/v2d-track2:pull&service=ghcr.io" | jq -r .token)
curl -sI -H "Authorization: Bearer $TOK" https://ghcr.io/v2/luaiabuelsamen/v2d-track2/manifests/submission
# HTTP 200, docker-content-digest: sha256:f3f3d24bbf333a0e2e0323dda8968810ba18e68cc92e9c7da05d32ba0b7c7af1
```

That digest is the one we pushed, so a judge with no credentials gets the same image we built, and
"private package" is ruled out as a failure mode rather than hoped against.

## Which forms to file, and when

Three live entries, three forms. They are **not** all in the same state:

| form | when | why |
|---|---|---|
| **Tier 1** | **now** | Confirmed valid in writing by the organisers' answer 2 (`results/organiser_answers.json`): driving a Tier 1 rollout from a reference carrying strictly less information than Tier 1 permits is allowed, which is what 16 of its 30 clips do. Nothing in the form is score-dependent, so a later score improvement does not invalidate a filed form |
| **Tier 2** | **now** | Confirmed valid by answer 1: a reference derived only from that tier's own released file, with no read of another tier, is permitted. That is exactly what `src/v2d/clean/support_contact.py` does |
| **Tier 3** | **hold** | Answer 3 forbids the ground-truth mesh for training, and we use it both as FoundationPose's CAD model and as the simulator's object geometry. A follow-up naming both uses is in the same thread. **Not filing this form is how we decline the tier** if the answer is no, so it costs nothing to wait and would cost our credibility to file an entry we have been told does not adhere |

The three form fields are identical across all three; only the tier and its scores differ. File
Tier 1 and Tier 2 on the evidence we now hold, and let Tier 3 wait on one email.

Per-tier scores and standings as of 2026-11-02 (`results/standings_tiers_2026_11_02.json`):

| tier | AUC | SP-SR | MP-SR | MPPE cm | rank |
|---|---|---|---|---|---|
| Tier 1 | 0.54808 | 0.70000 | 0.50000 | 12.060 | 12 / 24 |
| Tier 2 | 0.46522 | 0.66666 | 0.46666 | 14.128 | 7 / 18 |
| Tier 3 | 0.26781 | 0.20000 | 0.03333 | 47.772 | 6 / 12 |

We have slipped a place on Tier 1 and on Tier 2 since 2026-11-01 without changing a number:
two new teams entered Tier 1 (one of them, Rice RobotPI Lab, straight to first on AUC at
0.90927) and the field is still moving a day and a half from the freeze.

**The template.** The email attaches `V2D_Challenge_Extended_Abstract_Template.zip`: a LaTeX
class, not a Word file, so the report is written in it rather than approximated. The limit is two
pages of *main text* — references and appendices do not count — and `docs/report/build.sh` checks
exactly that rather than the page count.
