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
| report PDF | `docs/TECHNICAL_REPORT.pdf` — two pages, their four sections |
| GitHub | <https://github.com/luaiabuelsamen/v2d-track2> (public, Apache-2.0) |
| container | `ghcr.io/luaiabuelsamen/v2d-track2:submission` — *filled in once the push lands* |

Per-tier scores, in case the form asks which submission the report is for:

| tier | AUC | SP-SR | MP-SR | MPPE cm | rank |
|---|---|---|---|---|---|
| Tier 1 | 0.54808 | 0.70000 | 0.50000 | 12.060 | 11 / 23 |
| Tier 2 | 0.46522 | 0.66666 | 0.46666 | 14.128 | 6 / 18 |
| Tier 3 | 0.26781 | 0.20000 | 0.03333 | 47.772 | 6 / 12 |

**The template.** The email attaches `V2D_Challenge_Extended_Abstract_Template.zip`, which this
machine cannot open from Gmail. The report was written to the four sections the email names in
its body, which is the stated requirement; if the zip imposes a different layout, download it
into the repo and the report can be reflowed without rewriting it, because
`tools/report/to_pdf.py` renders from the Markdown.
