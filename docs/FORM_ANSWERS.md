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
| container | `ghcr.io/luaiabuelsamen/v2d-track2:submission` — pushed 2026-11-02, digest `sha256:f3f3d24b…`. **Must be made public before the form is submitted** (see below) |

**The package is private.** GitHub has no REST endpoint for package visibility --- a `PATCH` to
`/user/packages/container/v2d-track2` returns 404 --- so this is a UI click and nobody else can
do it:
<https://github.com/users/luaiabuelsamen/packages/container/v2d-track2/settings> ->
*Danger Zone* -> *Change visibility* -> **Public**. A judge pulling a private package gets
`denied`, which is indistinguishable from a broken link.

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
