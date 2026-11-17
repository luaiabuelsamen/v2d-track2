import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 7.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "stix",
})
BLACK="#111111"; ORANGE="#555555"; GREY="#9a9a9a"; PURPLE="#4a6fa5"; TEAL="#4a6fa5"; RED="#777777"; BAND="#ececec"

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw=dict(width_ratios=[0.95, 1.05, 1.2], wspace=0.40))

# ---------------- (a) ----------------
ax = axs[0]
t = np.linspace(0, 1, 300)
x = 0.06 + 0.88*t
y = 0.22 + 0.42/(1+np.exp(-(t-0.45)*11)) + 0.10*np.sin(2*np.pi*t)*(1-t)
ax.plot(x, y, color=BLACK, lw=1.4, zorder=3, label=r"$r^{*}$  ground truth (Tier 1 input)")
off = 0.022
x2 = x + off*np.cos(1.6*np.pi*t+0.6); y2 = y + off*np.sin(1.3*np.pi*t+0.3) + 0.012
ax.plot(x2, y2, color=ORANGE, lw=0.9, ls=(0,(4,1.5)), zorder=2, label=r"$r_{2}$  released retarget (Tier 2)")
rng = np.random.default_rng(5)
x3 = x + 0.004*rng.standard_normal(300).cumsum()*0.3
y3 = y - 0.006 - 0.0012*np.maximum(0, np.arange(300)-170)
ax.plot(x3, y3, color=GREY, lw=0.9, ls=(0,(2,1.6)), zorder=1, label=r"$r_{3}$  own reconstruction (Tier 3)")
def glyph(ax, cx, cy, ang, col, w=0.07, h=0.042, lw=0.8):
    tr = transforms.Affine2D().rotate_deg_around(cx, cy, ang) + ax.transData
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, fill=False, ec=col, lw=lw, transform=tr, zorder=4))
k = 215
glyph(ax, x[k], y[k], 0, BLACK); glyph(ax, x2[k], y2[k], 21.4, ORANGE)
ax.annotate("21.4°", xy=(x2[k]+0.03, y2[k]+0.03), xytext=(x2[k]+0.07, y2[k]+0.13), color=ORANGE, fontsize=6.2, ha="center",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
k2 = 70
ax.annotate("", xy=(x[k2], y[k2]), xytext=(x2[k2], y2[k2]), arrowprops=dict(arrowstyle="|-|", color=ORANGE, lw=0.5, mutation_scale=1.5))
ax.text(x2[k2]+0.03, y2[k2]-0.03, "1.9 cm", color=ORANGE, fontsize=6.2, va="top")
ax.set_xlim(0, 1.0); ax.set_ylim(0.0, 1.0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("object trajectory (schematic)", labelpad=3)
ax.legend(loc="upper left", frameon=False, handlelength=1.5, borderaxespad=0.0, bbox_to_anchor=(-0.03, 1.05), labelspacing=0.35)
ax.text(0.97, 0.05, "all tiers scored\nagainst $r^{*}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color="#444")

# ---------------- (b) ----------------
ax = axs[1]
T = 300; tt = np.arange(T)
ax.axhspan(0, 10, color=BAND, zorder=0, lw=0)
ax.axhline(10, color=PURPLE, lw=0.6, ls=(0,(3,2)), zorder=1)
def track(base, drop_at, plateau, seed):
    r = np.random.default_rng(seed)
    d = base + 0.9*np.sin(tt/19.0+seed)**2 + r.standard_normal(T).cumsum()*0.03
    d = np.abs(d)
    if drop_at is not None:
        s = (tt-drop_at)/18.0
        d = d + plateau/(1+np.exp(-s))*(tt>drop_at-40)
    return d
# (base, drop frame, plateau, k)
cks = [(2.8, 70, 20.5, 1), (2.5, 115, 17.0, 2), (2.9, None, 0, 3), (2.1, 200, 10.5, 4), (1.5, None, 0, 5), (1.9, 140, 9.0, 6), (1.7, 245, 15.8, 7)]
for base, drop, plat, k in cks:
    d = track(base, drop, plat, k)
    sel = (k == 5)
    ax.plot(tt, d, color=PURPLE if sel else GREY, lw=1.4 if sel else 0.65, alpha=1 if sel else 0.8, zorder=3 if sel else 2)
    yl = d[-1]
    ax.text(T+5, yl, f"$k$={k}", fontsize=5.9, va="center", color=PURPLE if sel else GREY, fontweight="bold" if sel else "normal")
ax.set_ylim(0, 26); ax.set_xlim(0, T+40)
ax.set_xticks([0, 100, 200, 300]); ax.set_yticks([0, 10, 20])
ax.set_xlabel("frame")
ax.set_ylabel(r"$d_k(t)=\|p_{\tau_k}(t)-p_{r}(t)\|$  (cm)", labelpad=2)
ax.text(4, 8.9, "10 cm band around the input $r$", fontsize=6.0, color="#444444", va="top")
ax.text(0.03, 0.975, "select: latest $k$ with $\\max_t d_k(t)<10$ cm\n(reads $r$ only, never $r^{*}$)", transform=ax.transAxes, va="top", fontsize=6.0, color="#2f4f7f")

# ---------------- (c) ----------------
ax = axs[2]
s = np.linspace(0, 350, 400)
auc = 0.465*(1-np.exp(-s/95))**1.6
for a,b,c in [(0,150,"#f5f5f5"),(150,300,"#ececec"),(300,350,"#e1e1e1")]:
    ax.axvspan(a,b,color=c,lw=0,zorder=0)
ax.plot(s, auc, color=BLACK, lw=1.3, zorder=3)
ax.text(150, 0.02, "curriculum stages", fontsize=5.6, color="#777", ha="center")
ax.text(325, 0.02, "final", fontsize=5.6, color="#777", ha="center")
ax.text(60, 0.40, "train on $r$", fontsize=6.3, color=BLACK)
# cold swap
ax.annotate("", xy=(352, 0.295), xytext=(352, 0.46), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=7))
ax.plot([352],[0.289], "o", ms=3.0, color=RED, zorder=4)
ax.text(362, 0.255, "cold swap: $\\pi_\\theta$ evaluated on $r'$\n0.465 $\\rightarrow$ 0.289,  2/30 clips up", fontsize=6.0, color="#444444", va="center")
# warm start
s2 = np.linspace(0, 50, 100)
w = 0.465 + (0.548-0.465)*(1-np.exp(-s2/14))
ax.plot(350+s2, w, color=TEAL, lw=1.3, zorder=3)
ax.plot([350],[0.465],"o",ms=3.0,color=TEAL,zorder=4)
ax.text(300, 0.60, "warm start on $r'$: +50M steps,\ncounter set to 300M\n0.465 $\\rightarrow$ 0.548,  11/19 clips adopted", fontsize=6.0, color="#2f4f7f", va="bottom")
s3 = np.linspace(0, 50, 60)
ax.plot(400+s3, 0.548 - 0.02*(s3/50), color=TEAL, lw=0.9, ls=(0,(2,1.5)), zorder=3)
ax.text(404, 0.515, "2nd gen. on the same $r'$:\n0/3 adopted", fontsize=5.8, color="#2f4f7f", va="top")
ax.set_xlim(0, 470); ax.set_ylim(0, 0.72)
ax.set_xticks([0, 100, 200, 300, 350, 400, 450]); ax.set_xticklabels(["0", "100M", "200M", "300M", "350M", "", "+100M"], fontsize=5.8)
for lbl, h in zip(ax.get_xticklabels(), ["center","center","center","right","left","center","center"]): lbl.set_ha(h)
ax.set_xlabel("environment steps (SAC, one policy per sequence)")
ax.set_ylabel("AUC against $r^{*}$ (mean of 30 clips)", labelpad=2)
ax.set_yticks([0, 0.2, 0.4, 0.6])

for a, lab in zip(axs, ["(a)", "(b)", "(c)"]):
    a.text(-0.06, 1.10, lab, transform=a.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")

fig.savefig("method.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("method.svg", bbox_inches="tight", pad_inches=0.02)
fig.savefig("method.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
