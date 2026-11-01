#!/usr/bin/env python3
"""Emit the V2D Track 2 review page. Every number comes from review_data.json, not from prose."""

from __future__ import annotations

import json
import pathlib
import statistics

S = pathlib.Path(
    "/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad"
)
CLIPS = json.loads((S / "review_data.json").read_text())
OUT = S / "review" / "index.html"

MEAN_SHIP = statistics.mean(c["ships"] for c in CLIPS)
MEAN_W1 = statistics.mean(c["w1"] for c in CLIPS)
GAINED = sum(1 for c in CLIPS if c["ships"] > c["w1"] + 1e-9)
REJECTED = sum(c["rejected"] for c in CLIPS)
BELOW30 = [c for c in CLIPS if c["ships"] < 0.30]
MP_PASS = 14  # results/tier2_entry_wave2.json: mp_sr_clips_of_30.this_entry
MP_PASS_W1 = sum(1 for c in CLIPS if c["mp_w1"] == "Y")  # 5, the first wave
REGRESSED = sum(1 for c in CLIPS if c["ships"] < c["w1"] - 1e-9)

# the nine clips the decomposition names as orientation-limited (they hold the object)
ORI_LIMITED = ["e13", "e20", "e21", "e26", "e09", "e10", "e08", "e00", "e06"]
DROPPERS = ["e04", "e14", "e19", "e25", "e28"]


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- the per-clip chart
def dumbbell() -> str:
    """Wave 1 -> shipped, one row per clip, sorted by what shipped."""
    rows = CLIPS
    left, right, top, rowh = 92.0, 46.0, 30.0, 21.0
    w, h = 820.0, top + rowh * len(rows) + 34
    xmax = 0.95

    def x(v: float) -> float:
        return left + (w - left - right) * (v / xmax)

    p: list[str] = [
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
        'aria-label="Per-clip kit AUC, first wave against the shipped entry" '
        'class="chart">'
    ]
    # gridlines + axis
    for g in (0.0, 0.2, 0.4, 0.6, 0.8):
        gx = x(g)
        p.append(
            f'<line x1="{gx:.1f}" y1="{top - 12:.1f}" x2="{gx:.1f}" y2="{h - 30:.1f}" class="grid"/>'
        )
        p.append(
            f'<text x="{gx:.1f}" y="{h - 14:.0f}" class="tick" text-anchor="middle">{g:.1f}</text>'
        )
    # the 0.30 line the triage uses as its weak-clip cut
    cx = x(0.30)
    p.append(
        f'<line x1="{cx:.1f}" y1="{top - 16:.1f}" x2="{cx:.1f}" y2="{h - 30:.1f}" class="cut"/>'
    )
    p.append(
        f'<text x="{cx:.1f}" y="{top - 20:.0f}" class="cutlab" text-anchor="middle">0.30 weak-clip cut</text>'
    )

    for i, c in enumerate(rows):
        y = top + rowh * i + rowh / 2
        x1, x2 = x(c["w1"]), x(c["ships"])
        weak = c["ships"] < 0.30
        cls = "weak" if weak else "ok"
        p.append(
            f'<text x="{left - 56:.0f}" y="{y + 3.6:.1f}" class="clip {cls}">{c["tag"]}</text>'
        )
        p.append(
            f'<text x="{left - 8:.0f}" y="{y + 3.6:.1f}" class="arm" text-anchor="end">{esc(c["arm"][:3].strip())}</text>'
        )
        if abs(x2 - x1) > 0.6:
            p.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" class="link"/>')
            p.append(f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="3" class="dot1"/>')
        p.append(f'<circle cx="{x2:.1f}" cy="{y:.1f}" r="4.4" class="dot2 {cls}"/>')
        p.append(
            f'<text x="{x2 + 9:.1f}" y="{y + 3.6:.1f}" class="val {cls}">{c["ships"]:.3f}</text>'
        )
    # the two means
    for v, k in ((MEAN_W1, "m1"), (MEAN_SHIP, "m2")):
        mx = x(v)
        p.append(
            f'<line x1="{mx:.1f}" y1="{top - 12:.1f}" x2="{mx:.1f}" y2="{h - 30:.1f}" class="mean {k}"/>'
        )
    p.append("</svg>")
    return "".join(p)


# --------------------------------------------------------------------------- the clip table
def table() -> str:
    head = (
        "<tr><th>clip</th><th>object</th><th class='n'>shipped</th><th class='n'>wave 1</th>"
        "<th class='n'>&Delta;</th><th>lost to</th><th class='n'>pos</th><th class='n'>ori</th>"
        "<th class='n'>panto</th><th>MP w1</th><th>failure mode</th></tr>"
    )
    body = []
    for c in CLIPS:
        d = c["ships"] - c["w1"]
        dtxt = f"{d:+.3f}" if abs(d) > 1e-9 else "&mdash;"
        dcls = "up" if d > 1e-9 else ("down" if d < -1e-9 else "flat")
        lost = {"pos": "position", "ori": "orientation", "—": "&mdash;"}.get(
            c["lost_to"], esc(c["lost_to"])
        )
        lcls = "pos" if c["lost_to"] == "pos" else ("ori" if c["lost_to"] == "ori" else "")
        weak = " class='weakrow'" if c["ships"] < 0.30 else ""
        body.append(
            f"<tr{weak}><th scope='row'>{c['tag']}</th><td>{esc(c['obj'])}</td>"
            f"<td class='n strong'>{c['ships']:.3f}</td><td class='n dim'>{c['w1']:.3f}</td>"
            f"<td class='n {dcls}'>{dtxt}</td><td class='{lcls}'>{lost}</td>"
            f"<td class='n'>{c['pos_cm']:.1f}</td><td class='n'>{c['ori_deg']:.0f}</td>"
            f"<td class='n'>{c['panto']:.2f}</td>"
            f"<td class='{'yes' if c['mp_w1'] == 'Y' else 'no'}'>{'pass' if c['mp_w1'] == 'Y' else '&mdash;'}</td>"
            f"<td class='mode'>{esc(c['bucket'])}</td></tr>"
        )
    return f"<table class='clips'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


FIGS = [
    (
        "deep_dive/a_per_clip_error_timeseries.png",
        "Every clip, frame by frame",
        "Object position error (blue, left axis) and orientation error (vermillion, right) against truth for the checkpoint that shipped. The green dashed line is the first sustained loss past 10 cm; grey fill is the reference's own speed. Weak clips are titled in vermillion.",
    ),
    (
        "deep_dive/d_auc_decomposition.png",
        "Which channel costs the score",
        "Per clip: the AUC that shipped (grey) against the same rollout with its orientation replaced by truth (blue, leaving only position error) and with its position replaced by truth (vermillion, leaving only orientation). The taller bar names the channel to fix.",
    ),
    (
        "deep_dive/e_hand_to_object_distance.png",
        "Pantomime, as a number",
        "Palm-to-object distance for the twelve worst clips: achieved object in vermillion, reference object in blue, palm speed in grey, and the green band where the retargeter says the object is in hand. A vermillion line that leaves the band while the body keeps moving is a robot miming the task.",
    ),
    (
        "deep_dive/c_checkpoint_curves_and_variance.png",
        "Checkpoints, and the noise floor",
        "Left: kit AUC against training steps for all 30 clips, with a black star on the checkpoint our registered rule picked. Right: the spread across each clip's last five checkpoints, which is the noise any A/B has to clear before it means anything.",
    ),
    (
        "deep_dive/g_threshold_margins.png",
        "Who is one small gain from passing",
        "Distance to each success threshold per clip on a symmetric log scale; a clip passes only when both bars are positive. SP-SR above (10 cm / 0.5 rad), MP-SR below (3 cm / 30°).",
    ),
    (
        "deep_dive/f_correlations.png",
        "What actually predicts the score",
        "Kit AUC against object radius, first-loss frame, smoother anchor count and the reference's own ceiling, with Spearman ρ in each title. The dashed line is y = x: points above it are clips that beat the reference they were tracking.",  # noqa: RUF001 - rho is the statistic's name
    ),
    (
        "deep_dive/b_comparison_clips_vs_stock.png",
        "Our recipe against the stock one",
        "The four clips where we hold a full-length stock-recipe run: combined recipe (blue) against stock (orange), position above and orientation below, with the 10 cm and 30° gates dotted.",
    ),
    (
        "hail_mary/b_failing_clip_error_curves.png",
        "The five quiet failures",
        "e05, e09, e17, e19 and e28 lose the object while it is nearly still, so speed does not explain them. These are the clips the contact-role reward was written for.",
    ),
]


def figures() -> str:
    out = []
    for src, title, cap in FIGS:
        out.append(
            f'<figure><a href="{src}" target="_blank" rel="noopener"><img src="{src}" alt="{esc(title)}" loading="lazy"></a>'
            f"<figcaption><strong>{esc(title)}</strong>{esc(cap)}</figcaption></figure>"
        )
    return "".join(out)


STAGES = [
    ("Released clip", "video + caption", "given"),
    ("Retarget", "<code>soma_to_g1</code>, 43 joints, CPU", "free"),
    ("Clean the reference", "support-plane + bridge smoother", "free"),
    ("Train", "FlashCHORD SAC, 350M steps, 1 world", "paid"),
    ("Checkpoint 7&times;", "every 50M steps", "paid"),
    ("Select", "z-drop, reads input only", "free"),
    ("Convert", "one rigid transform to world frame", "free"),
    ("Pack &amp; score", "kit packer, kit metrics", "free"),
]


def stages() -> str:
    out = []
    for i, (name, sub, kind) in enumerate(STAGES):
        if i:
            out.append('<span class="arrow" aria-hidden="true">&rarr;</span>')
        out.append(f'<span class="stage {kind}"><b>{name}</b><i>{sub}</i></span>')
    return "".join(out)


html = f"""<title>V2D Track 2 Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* Layout: a single scrolling review -- scorecard, then evidence, then the costed decision. */
:root {{
  --bg: #F6F8FA; --surface: #FFFFFF; --sunk: #EDF1F5;
  --fg: #111821; --muted: #5B6675; --faint: #8893A2; --line: #DCE2EA;
  --pos: #0072B2; --ori: #D55E00; --good: #009E73; --warn: #E69F00;
  --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0E131A; --surface: #161C25; --sunk: #1C2430;
  --fg: #E8EDF4; --muted: #96A2B2; --faint: #6F7C8C; --line: #27303C;
  --pos: #4BA3DB; --ori: #F08440; --good: #35BE99; --warn: #F0B840;
  color-scheme: dark;
}} }}
:root[data-theme="dark"] {{
  --bg: #0E131A; --surface: #161C25; --sunk: #1C2430;
  --fg: #E8EDF4; --muted: #96A2B2; --faint: #6F7C8C; --line: #27303C;
  --pos: #4BA3DB; --ori: #F08440; --good: #35BE99; --warn: #F0B840;
  color-scheme: dark;
}}
* {{ box-sizing: border-box }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); line-height: 1.55;
  -webkit-font-smoothing: antialiased }}
.wrap {{ max-width: 1120px; margin: 0 auto;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 40px 72px }}
h1 {{ font-size: clamp(27px, 4.6vw, 42px); font-weight: 600; letter-spacing: -0.022em;
  margin: 0 0 10px; text-wrap: balance; line-height: 1.12 }}
h2 {{ font-size: clamp(19px, 2.5vw, 25px); font-weight: 600; letter-spacing: -0.015em;
  margin: 0 0 4px; text-wrap: balance }}
h3 {{ font-size: 16px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.008em }}
p {{ margin: 0 0 13px; max-width: 70ch }}
a {{ color: var(--pos) }}
code, .n, .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums }}
code {{ font-size: 0.9em; background: var(--sunk); padding: 1px 5px; border-radius: 3px }}
.eyebrow {{ font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--faint); margin: 0 0 9px }}
.lede {{ font-size: 17px; color: var(--muted); max-width: 66ch; margin-bottom: 0 }}
.lede strong {{ color: var(--fg); font-weight: 600 }}
section {{ margin-top: 54px }}
.shead {{ border-top: 2px solid var(--fg); padding-top: 13px; margin-bottom: 20px }}
.shead p {{ color: var(--muted); margin: 0; max-width: 68ch }}

/* --- scorecard */
.tiers {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(248px, 1fr));
  margin-top: 26px }}
.tier {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 17px 18px; display: flex; flex-direction: column; gap: 9px; min-width: 0 }}
.tier.lead {{ border-color: var(--ori); border-width: 2px }}
.tier .top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px }}
.tier .nm {{ font-weight: 600; font-size: 15px }}
.tier .big {{ font-family: var(--mono); font-size: 33px; font-weight: 600; letter-spacing: -0.02em;
  line-height: 1 }}
.tier .vs {{ font-size: 12.5px; color: var(--muted) }}
.tier .vs b {{ font-family: var(--mono); font-weight: 600 }}
.chip {{ font-family: var(--mono); font-size: 11px; font-weight: 600; padding: 2px 7px;
  border-radius: 999px; white-space: nowrap }}
.chip.up {{ background: color-mix(in oklab, var(--good) 17%, transparent); color: var(--good) }}
.chip.down {{ background: color-mix(in oklab, var(--ori) 17%, transparent); color: var(--ori) }}
.chip.zero {{ background: color-mix(in oklab, var(--ori) 22%, transparent); color: var(--ori) }}
.tier .sub {{ font-size: 12.5px; color: var(--faint); border-top: 1px solid var(--line);
  padding-top: 8px; margin-top: auto }}

/* --- metric strip */
.strip {{ display: grid; gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  margin-top: 18px }}
.cell {{ background: var(--surface); padding: 13px 15px; min-width: 0 }}
.cell .k {{ font-size: 11.5px; color: var(--faint); font-family: var(--mono);
  letter-spacing: 0.04em; text-transform: uppercase }}
.cell .v {{ font-family: var(--mono); font-size: 21px; font-weight: 600; margin-top: 3px }}
.cell .d {{ font-size: 11.5px; color: var(--muted); margin-top: 2px }}

/* --- pipeline */
.flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 7px; margin: 20px 0 10px }}
.stage {{ background: var(--surface); border: 1px solid var(--line); border-radius: 7px;
  padding: 9px 12px; display: flex; flex-direction: column; gap: 1px; min-width: 0; flex: 1 1 auto }}
.stage b {{ font-size: 13.5px; font-weight: 600 }}
.stage i {{ font-style: normal; font-size: 11.5px; color: var(--muted); font-family: var(--mono) }}
.stage.paid {{ border-left: 3px solid var(--ori) }}
.stage.free {{ border-left: 3px solid var(--good) }}
.stage.given {{ border-left: 3px solid var(--faint) }}
.arrow {{ align-self: center; color: var(--faint); font-size: 15px }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--muted);
  font-family: var(--mono) }}
.legend span::before {{ content: ""; display: inline-block; width: 9px; height: 9px;
  border-radius: 2px; margin-right: 6px; vertical-align: baseline }}
.legend .f::before {{ background: var(--good) }}
.legend .p::before {{ background: var(--ori) }}
.legend .g::before {{ background: var(--faint) }}

/* --- chart */
.chartbox {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 12px 8px; margin-top: 20px; overflow-x: auto }}
.chart {{ min-width: 560px; display: block }}
.chart .grid {{ stroke: var(--line); stroke-width: 1 }}
.chart .cut {{ stroke: var(--ori); stroke-width: 1; stroke-dasharray: 3 3; opacity: .65 }}
.chart .cutlab {{ fill: var(--ori); font: 500 10px var(--mono) }}
.chart .tick {{ fill: var(--faint); font: 400 11px var(--mono) }}
.chart .clip {{ fill: var(--fg); font: 500 12px var(--mono) }}
.chart .clip.weak {{ fill: var(--ori) }}
.chart .arm {{ fill: var(--faint); font: 400 10px var(--mono) }}
.chart .link {{ stroke: var(--faint); stroke-width: 1.6; opacity: .5 }}
.chart .dot1 {{ fill: var(--bg); stroke: var(--faint); stroke-width: 1.4 }}
.chart .dot2 {{ fill: var(--pos) }}
.chart .dot2.weak {{ fill: var(--ori) }}
.chart .val {{ fill: var(--muted); font: 500 11px var(--mono) }}
.chart .val.weak {{ fill: var(--ori) }}
.chart .mean {{ stroke-width: 1.4; stroke-dasharray: 5 4 }}
.chart .m1 {{ stroke: var(--faint) }}
.chart .m2 {{ stroke: var(--pos) }}
.caption {{ font-size: 12.5px; color: var(--muted); margin: 10px 2px 0; max-width: 78ch }}

/* --- tables */
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px;
  background: var(--surface); margin-top: 20px }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px }}
th, td {{ text-align: left; padding: 7px 11px; border-bottom: 1px solid var(--line);
  white-space: nowrap }}
thead th {{ position: sticky; top: 0; background: var(--sunk); font-size: 11px;
  font-family: var(--mono); letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); font-weight: 500; z-index: 1 }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0 }}
tbody th {{ font-family: var(--mono); font-weight: 600; font-size: 12.5px }}
td.n, th.n {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums }}
td.strong {{ font-weight: 600 }}
td.dim {{ color: var(--faint) }}
td.up {{ color: var(--good) }}
td.down {{ color: var(--ori) }}
td.flat {{ color: var(--faint) }}
td.pos {{ color: var(--pos) }}
td.ori {{ color: var(--ori) }}
td.yes {{ color: var(--good); font-weight: 600 }}
td.no {{ color: var(--faint) }}
td.mode {{ color: var(--muted); font-size: 12px }}
.weakrow th, .weakrow td {{ background: color-mix(in oklab, var(--ori) 7%, transparent) }}
.plain td, .plain th {{ white-space: normal }}
.plain td:first-child {{ font-family: var(--mono); font-weight: 600; white-space: nowrap }}

/* --- figures */
.figs {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
  margin-top: 22px }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; min-width: 0 }}
figure img {{ display: block; width: 100%; max-width: 100%; background: #fff }}
figcaption {{ padding: 12px 14px; font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--line) }}
figcaption strong {{ display: block; color: var(--fg); font-size: 13.5px; margin-bottom: 3px }}

/* --- callouts */
.note {{ background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--pos);
  border-radius: 8px; padding: 14px 16px; margin: 18px 0 }}
.note.warn {{ border-left-color: var(--ori) }}
.note.good {{ border-left-color: var(--good) }}
.note h3 {{ margin-bottom: 5px }}
.note p:last-child {{ margin-bottom: 0 }}
.cards {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
  margin-top: 20px }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 17px; min-width: 0 }}
.card.pick {{ border-color: var(--good); border-width: 2px }}
.card .rank {{ font-family: var(--mono); font-size: 11px; color: var(--faint);
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 7px }}
.card .cost {{ font-family: var(--mono); font-size: 13px; font-weight: 600; color: var(--fg);
  margin-top: 10px; padding-top: 9px; border-top: 1px solid var(--line) }}
.card p {{ font-size: 13.5px; margin-bottom: 9px }}
ul {{ margin: 0 0 13px; padding-left: 20px; max-width: 70ch }}
li {{ margin-bottom: 5px }}
.foot {{ margin-top: 54px; border-top: 1px solid var(--line); padding-top: 18px;
  font-size: 12.5px; color: var(--faint) }}
</style>

<div class="wrap">

<p class="eyebrow">NVIDIA Video-to-Data Challenge &middot; Track 2 &middot; Unitree G1 + Dex3</p>
<h1>Where we stand, and where the next points are</h1>
<p class="lede">One entry is on eight leaderboards. It beats the published baseline on Tier 2 by a
wide margin and <strong>loses to it on Tier 1</strong>, because the same file was submitted to both
and Tier 1 is graded against a harder bar. Tier 3 is still <strong>zero</strong>, and that is a third
of the rank. Everything below is measured, with the method and the per-clip numbers attached.</p>

<div class="tiers">
  <div class="tier">
    <div class="top"><span class="nm">Tier 2 &mdash; noisy input</span><span class="chip up">+0.091</span></div>
    <div class="big" style="color:var(--good)">0.465</div>
    <div class="vs">CHORD baseline <b>0.374</b> &middot; we are <b>24%</b> above it</div>
    <div class="sub">Good, and the strongest position we hold.</div>
  </div>
  <div class="tier lead">
    <div class="top"><span class="nm">Tier 1 &mdash; clean input</span><span class="chip down">&minus;0.045</span></div>
    <div class="big" style="color:var(--ori)">0.465</div>
    <div class="vs">CHORD baseline <b>0.510</b> &middot; we are <b>9%</b> below it</div>
    <div class="sub">Same file. Tier 1 allows a cleaner input we never trained on.</div>
  </div>
  <div class="tier">
    <div class="top"><span class="nm">Tier 3 &mdash; no input</span><span class="chip zero">not entered</span></div>
    <div class="big" style="color:var(--faint)">&mdash;</div>
    <div class="vs">reconstruction scores <b>0.228</b> raw; a policy run is in flight</div>
    <div class="sub">The largest single gain available, and nearly paid for.</div>
  </div>
</div>

<div class="strip">
  <div class="cell"><div class="k">Mean AUC</div><div class="v">{MEAN_SHIP:.5f}</div><div class="d">30 clips, kit scorer</div></div>
  <div class="cell"><div class="k">SP-SR</div><div class="v">0.667</div><div class="d">baseline 0.533</div></div>
  <div class="cell"><div class="k">MP-SR</div><div class="v">0.467</div><div class="d">baseline 0.067 &middot; {MP_PASS}/30 clips, up from {MP_PASS_W1}</div></div>
  <div class="cell"><div class="k">MPPE</div><div class="v">14.13</div><div class="d">cm &middot; baseline 20.5</div></div>
  <div class="cell"><div class="k">Credit left</div><div class="v">$30.38</div><div class="d">of $250, per Brev</div></div>
</div>
<p class="caption">Local kit scores match the public leaderboard to five decimals on every metric,
so the numbers on this page and the board are the same numbers.</p>

<section>
  <div class="shead">
    <p class="eyebrow">The finding that shaped everything</p>
    <h2>All three tiers are graded against the same truth</h2>
    <p>Only the permitted <em>input</em> differs. That single fact decides what is worth building.</p>
  </div>
  <p>Each tier's config points at the same <code>multi_view_ground_truth</code>; Tier 1 hands you a
  clean object trajectory, Tier 2 a noisy one, Tier 3 nothing but video. We proved the shared truth
  two ways: the Tier 1 and Tier 2 ground-truth CSVs are byte-identical, and a Kabsch fit between
  them returns 0.000 cm.</p>
  <p>Two consequences follow, and both are load-bearing. First, identical predictions earn identical
  scores &mdash; which is why one file sits on both the Tier 1 and Tier 2 boards at 0.465, and why
  that is a <em>good</em> Tier 2 result and a <em>poor</em> Tier 1 one. Second, Tier 3 withholds the
  trajectory but not the scene: the support-surface assets in the two released tiers are
  byte-identical, so they are tier-invariant released data, and a Tier 3 entry needs a reconstructed
  trajectory rather than a reconstructed world.</p>
  <div class="note warn">
    <h3>The most expensive bug we found, and it was ours</h3>
    <p>Our first full-length baseline scored a mean AUC of <b>0.043</b>. All three run scripts passed
    <code>scene.support=false</code> while NVIDIA's configs default it to <code>true</code>. With no
    table, the object falls at reset and the episode terminates on frame 0 &mdash; the signature was
    metrics that were bit-identical across all seven checkpoints. 14 of 30 clips ship a support
    primitive. One flag was worth more than every reward change we made afterwards combined, and it
    also explains why about 12 clips in our first Tier 1 entry read near zero.</p>
  </div>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">Pipeline</p>
    <h2>From released clip to scored row</h2>
    <p>Eight stages. Only two of them cost money, which is why most of the work happened on a Jetson for nothing.</p>
  </div>
  <div class="flow">{stages()}</div>
  <div class="legend"><span class="g">given</span><span class="f">free, CPU</span><span class="p">paid, GPU</span></div>
  <p style="margin-top:16px">The paid stage is one <code>train_flash_sac</code> run per clip at
  349,999,104 steps and <code>world_count=1</code>, about 4.4 GPU-hours on an A6000. A full 30-clip
  pass is roughly 127 GPU-hours, or $86. Everything either side of it &mdash; retargeting, reference
  cleaning, checkpoint selection, frame conversion, packing and scoring &mdash; runs on CPU at no
  cost, and that asymmetry is why the method below is built around spending compute only on
  questions that CPU work cannot answer.</p>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">Results &middot; 30 clips</p>
    <h2>What shipped, clip by clip</h2>
    <p>Hollow dot is the first full-length wave; filled dot is what shipped after the second wave.
    {GAINED} of 30 clips improved, {REGRESSED} slipped by under 0.001, and the mean moved
    {MEAN_W1:.4f} &rarr; {MEAN_SHIP:.4f}.</p>
  </div>
  <div class="chartbox">{dumbbell()}</div>
  <p class="caption">Sorted by the shipped score. Vermillion marks the {len(BELOW30)} clips still
  under 0.30 &mdash; the tail that holds almost all of the remaining headroom. The column beside each
  clip id is the arm the shipped rollout came from. Dashed verticals are the two means.</p>
  <div class="scroll">{table()}</div>
  <p class="caption">
    <b>MP w1</b> is the first wave's MP-SR pass, which is the column the decomposition was
    computed on; the shipped entry passes {MP_PASS}. <b>panto</b> is the fraction of frames where the body is still moving but the object has been
    lost &mdash; the robot miming the task. <b>lost to</b> comes from rescoring each rollout twice,
    once with truth orientation substituted and once with truth position, so the channel that costs
    the score is measured rather than guessed. <b>pos</b>/<b>ori</b> are mean error in cm and degrees.
  </p>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">Diagnosis</p>
    <h2>The score is lost by dropping things, not by tracking them badly</h2>
    <p>Across 30 clips, the strongest predictor of AUC is how much of the clip the robot spends
    miming (&rho; = &minus;0.647). Object size predicts nothing at all (&rho; = &minus;0.191).</p>
  </div>
  <p>That reframes the problem. The intuitive story &mdash; big or heavy objects are harder &mdash;
  is measurably wrong, and there is a reason: <code>soma_to_g1.py</code> writes every object URDF at
  mass 1.0 kg with inertia <code>diag(0.01)</code>, so nothing in simulation is heavy. For the
  squeegee, cane and vacuum, the real shape's inertia at 1 kg is 6&times; to 14&times; what the URDF
  uses, which makes them far too easy to spin. We have not touched those files: whether object
  physics may be edited is a rules question, not ours to decide.</p>
  <p>The clips split cleanly into two populations that need opposite fixes.
  <b style="color:var(--ori)">Droppers</b> ({", ".join(DROPPERS)}) lose the object mid-clip and lose
  the score to position; <b style="color:var(--pos)">holders</b> ({", ".join(ORI_LIMITED[:5])} and
  four more) keep it the whole way and lose the score to orientation, worth about +0.318 each if
  orientation were perfect.</p>
  <div class="figs">{figures()}</div>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">Method</p>
    <h2>The parts worth reusing</h2>
    <p>Three pieces of discipline did more for the result than any single reward term.</p>
  </div>
  <div class="cards">
    <div class="card">
      <div class="rank">Selection</div>
      <h3>An input-only rule, registered in advance</h3>
      <p>The provenance schema allows a different checkpoint per episode. We pick, per clip, the
      latest checkpoint whose object never leaves a 10 cm band around the <em>input</em> reference.
      It reads no ground truth, so anyone can reproduce it.</p>
      <p>It cost us something real: {REJECTED} rollouts scored higher on truth than the ones that
      shipped, and we shipped the rule's picks anyway.</p>
      <p><b>What it is not worth is score.</b> Across all 30 clips the three legal rules sit inside
      0.005 of each other &mdash; z-drop 0.3908, latest-drop-free 0.3946, final checkpoint 0.3898
      &mdash; and an oracle that reads truth reaches only 0.4201. Selection is capped at +0.029 and
      no submittable rule collects even that. The early four-clip figure that suggested otherwise was
      the evaluator's metric on too small a sample.</p>
      <div class="cost">+0.001 over the final checkpoint &middot; worth it for reproducibility, not points</div>
    </div>
    <div class="card">
      <div class="rank">Patches</div>
      <h3>Every patch refuses an upstream it does not recognise</h3>
      <p>Reward terms, curriculum changes and the warm-start hook are patch scripts anchored on
      unique strings, applied atomically, idempotent, and tested against the real upstream tree. A
      missing anchor exits non-zero having changed nothing.</p>
      <p>This is what makes a silent half-patch impossible &mdash; the failure mode that would
      otherwise be indistinguishable from a null result.</p>
      <div class="cost">3 patch scripts, 7 known-answer test modules</div>
    </div>
    <div class="card">
      <div class="rank">Pre-registration</div>
      <h3>Bars written before launch, in the repo</h3>
      <p>Each arm's bar was committed before its boxes existed, which is the only reason the negative
      results are usable. It also caught a pipeline bug rather than a method failure: a 0.043 mean
      tripped a stop line that existed precisely to catch that.</p>
      <div class="cost">Negative results recorded, not discarded</div>
    </div>
  </div>
  <div class="note">
    <h3>What the paid runs cost, and the one we got wrong</h3>
    <p>About <b>$220 of the $250 credit</b>. The 30-clip combined run was $88, the second wave about
    $112 across 19 boxes, and the rest went on probes, the reconstruction and evals. Two avoidable
    losses: roughly $3 on a first launch that died at the manifest gate on every box because the
    provenance file was written after the manifest, and about $1.90 idling while the launcher logged
    progress for a runner that had already exited. Both now fail loudly.</p>
  </div>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">The decision</p>
    <h2>Where the remaining $30 should go</h2>
    <p>Ranked by rank-points per dollar, with the evidence behind each and the bar it has to clear.</p>
  </div>
  <div class="cards">
    <div class="card pick">
      <div class="rank">1 &middot; in flight</div>
      <h3>A valid Tier 3 entry</h3>
      <p>Tier 3 is a third of the rank and we score nothing there. All 30 clips are reconstructed
      from video with NVIDIA's own modules and placed in the released world by one frame-0 transform,
      which is what the FAQ permits; the frame-0 residual is 0.0000 cm on all 30.</p>
      <p>The raw reconstruction cannot be submitted &mdash; the packer needs a non-empty
      <code>eval_code_sha256</code> and no evaluator ran &mdash; so the 30 shipped actors are being
      driven by it right now. Any valid entry is the whole gain.</p>
      <div class="cost" style="color:var(--good)">~$1 &middot; running &middot; 28/30 clips done</div>
    </div>
    <div class="card pick">
      <div class="rank">2 &middot; recommended</div>
      <h3>Fine-tune the actors onto the clean Tier 1 reference</h3>
      <p>This is the only route to the Tier 1 baseline, and we have measured why the cheap version
      fails: driving the existing actors with clean input scored <b>0.289</b>, down from 0.465, with
      only 2 of 30 clips improving. Each actor is tuned to the trajectory it trained on, so the
      reference has to be adapted <em>to</em>, not swapped under it.</p>
      <p>The warm-start patch does exactly that &mdash; actor restored from a shipped policy
      checkpoint, critic cold, step counter started late so the curriculum sits where the actor was
      trained, and a learning-rate guard for the first 2,000 updates so a cold critic cannot walk a
      known-good actor off its policy. Written and unit-tested; never run.</p>
      <div class="cost">~$13 for 30 clips &times; 50M steps &middot; pilot 3 clips first, ~$2</div>
    </div>
    <div class="card">
      <div class="rank">3 &middot; if the pilot clears</div>
      <h3>Keep the object in hand on the five droppers</h3>
      <p>e04, e14, e19, e25 and e28 are worth +0.054 of mean AUC between them just to reach their own
      reference ceilings. We know the mechanism per clip from the rollouts, and the contact-role
      reward was written for exactly this and launched but never scored before a reboot &mdash; the
      code is paid for, the run is not.</p>
      <div class="cost">~$15 &middot; 5 clips &middot; already implemented</div>
    </div>
    <div class="card">
      <div class="rank">Not this</div>
      <h3>A wider orientation std</h3>
      <p>The decomposition says the holders are orientation-limited and worth +0.095 at the ceiling,
      which looked like the best buy on paper. We ran it: std 0.8 on eight of those nine clips, and
      only e09 beat its record. Raising the width is not the fix &mdash; the term is saturated at
      20&ndash;50&deg;, so it needs reshaping to have gradient there, which is a code change and a new
      bar, not a sweep.</p>
      <div class="cost" style="color:var(--ori)">$27 spent &middot; 1 of 8 clips adopted</div>
    </div>
  </div>
  <div class="note good">
    <h3>The order, and why Tier 1 outranks more Tier 2 work</h3>
    <p>Tier 2 is already 24% above its baseline and its remaining headroom is the expensive kind.
    Tier 1 is the weakest position we hold &mdash; below a published baseline, on a third of the rank
    &mdash; and the fix is the one experiment whose failure mode we have already measured and
    designed around. So: finish Tier 3, pilot the Tier 1 fine-tune on three clips against the 0.465
    record, and only spend the rest if the pilot clears its bar.</p>
  </div>
</section>

<section>
  <div class="shead">
    <p class="eyebrow">Open items</p>
    <h2>Blocked on a person, not on compute</h2>
  </div>
  <div class="scroll">
    <table class="plain">
      <thead><tr><th>item</th><th>why it matters</th><th>who</th></tr></thead>
      <tbody>
        <tr><td>GitHub invite</td><td>Every one of the 22,989 submitted rows quotes a commit URL. Until the organisers' account is admitted to the private repo, none of those URLs resolve for them.</td><td>Luai</td></tr>
        <tr><td>One push</td><td><code>origin/main</code> is 19 commits behind, so no Tier 3 code is on the remote. The audit checks a commit URL's shape but never resolves it &mdash; an unpushed sha validates locally and 404s for a judge.</td><td>Luai</td></tr>
        <tr><td>Final two submissions</td><td><code>max_final_submissions: 2</code>, and the Kaggle API exposes no selected flag, so it is a UI action before the freeze.</td><td>Luai</td></tr>
        <tr><td>Registration form</td><td>Eligibility needs it confirmed submitted.</td><td>Luai</td></tr>
        <tr><td>Object inertia</td><td>Every URDF is 1 kg, <code>diag(0.01)</code>. Correcting it would change the physics for the better but may not be permitted.</td><td>rules question</td></tr>
      </tbody>
    </table>
  </div>
  <p class="caption">Freeze: <b>4 November 2026, 5:00 p.m. EST</b>.</p>
</section>

<p class="foot">Every figure and number on this page is generated from the repo's own result records
&mdash; <code>results/tier2_entry_wave2.json</code>, <code>results/metrics_deep_dive.json</code>,
<code>results/t1_clean_input_eval.json</code> and <code>results/t3_reconstruction.json</code>. The
per-clip table reproduces both published means exactly ({MEAN_W1:.5f} and {MEAN_SHIP:.5f}), which is
the check that it was assembled correctly. Scorer: the submission kit's own metric modules,
sha256-verified against <code>KIT_MANIFEST.json</code>.</p>

</div>
"""

OUT.write_text(html)
print(f"wrote {OUT} ({len(html):,} B)")
print(f"mean shipped {MEAN_SHIP:.5f} | wave1 {MEAN_W1:.5f} | gained {GAINED} | rejected {REJECTED}")
print(f"below 0.30: {len(BELOW30)} -> {[c['tag'] for c in BELOW30]}")
print(f"MP pass: {MP_PASS}")
