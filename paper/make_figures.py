#!/usr/bin/env python3
"""Publication figures for the Selphi 2 paper. Panel sources:
Figure 2 (figure2_lcwgs_capture): capture-plus-low-pass GIAB libraries, chromosome 22, n = 6
     (HG002-HG007 = NA24385, NA24149, NA24143, NA24631, NA24694, NA24695). Per-sample
     concordance JSONs live in figures/data/chr22/ (copied from the pilot directory) and are
     loaded relative to this script, with a hardcoded fallback dict of the plotted values.
     2a = array-equivalent route vs likelihood engine, non-reference concordance (Table 1;
          Selphi's own site list; *_conc_hard.json vs *_conc_lcwgs.json).
     2b = same comparison, heterozygote recall (Table 1).
     2c = Selphi 2 vs GLIMPSE2 on identical site lists, non-reference concordance (Table 1b;
          *_conc_glimpse2.json / *_conc_selphi_native.json = native pileup + BAQ /
          *_conc_selphi_isec.json = bcftools likelihoods). The pre-BAQ native run
          (*_conc_selphi_bam.json) is an ablation and is not plotted.
     2d = per reference-panel-MAF stratum delta non-reference concordance, Selphi 2 minus
          GLIMPSE2, both Selphi arms (Table 1c).
Figure 3 (figure3_accuracy): 3a = genome-wide (20-autosome) n-weighted per-MAF R^2, five-way
     imputation-only (all tools impute from the identical phased target + panel; Supplementary
     Table S9; OVERALL aggregates in Results text). 3b = MESA per-ancestry per-sample-mean R^2
     (mc=132,676 run; deltas match Table 4b prose). 3c = Table 4c. 3d = Table 2b coverage sweep.
Figure 4 (figure4_efficiency): 4a = lcWGS wall time, whole chr22, 16 threads, two labelled clusters
     (capture libraries, 4,796-hap panel, BAM in -> imputed out, quiet-machine 2026-09-02;
     downsampled 1x, 6,332-hap panel = the former Figure 3c values, unchanged).
     4b/4c = Table 5b whole-genome wall time / peak memory (content unchanged).
Supplementary (figureS_replication): per chromosome (22, 20, 10, 1) the mean paired delta
     non-reference concordance, Selphi 2 native pileup + BAQ minus GLIMPSE2 (pp), overall and by
     panel-MAF stratum, with the six per-sample points and k/6 wins (Table 1d). Per-sample deltas
     are recomputed from figures/data/chrN/*_conc_{selphi_native,glimpse2}.json and cross-checked
     against figures/data/paper_numbers_multichr.json (arms.native)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/data/projects/.claude_home/gt/selphi/mayor/rig/paper/figures"
DATA22 = os.path.join(HERE, "figures", "data", "chr22")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
# Okabe-Ito colorblind-safe palette
C = {"selphi": "#0072B2", "selphi153": "#56B4E9", "beagle": "#E69F00", "impute5": "#009E73",
     "minimac4": "#CC79A7", "glimpse2": "#E69F00", "quilt2": "#009E73",
     # Figure 2 arms
     "sel_native": "#0072B2", "sel_bcftools": "#56B4E9", "array_route": "#999999"}

# =====================================================================================
# ---- Figure 2: genotype-likelihood imputation of capture-plus-low-pass GIAB libraries ----
# =====================================================================================
SAMPLES = [("HG002", "NA24385"), ("HG003", "NA24149"), ("HG004", "NA24143"),
           ("HG005", "NA24631"), ("HG006", "NA24694"), ("HG007", "NA24695")]
HGS = [hg for hg, _ in SAMPLES]
STRATA_KEYS = ["common_maf>=5%", "low_maf_0.5-5%", "rare_maf<0.5%"]
STRATA_LBL = ["MAF ≥ 5%\n(common)", "MAF 0.5–5%", "MAF < 0.5%\n(rare)"]
# arm -> JSON suffix. hard/lcwgs are scored on Selphi's own site list (Table 1);
# glimpse2/selphi_native/selphi_isec on GLIMPSE2's per-sample site list (Tables 1b, 1c).
ARMS = ["hard", "lcwgs", "glimpse2", "selphi_native", "selphi_isec"]

# Hardcoded fallback = the values plotted from the JSONs on 2026-09-02 (nonref /
# het_recall = *_concordance_pct fields; by_af_nonref = by_af[stratum]['nonref_pct'] in
# STRATA_KEYS order). Used only when a JSON is missing or unreadable.
FALLBACK = {
    "hard": {
        "HG002": {"nonref": 71.4389, "het_recall": 57.8827, "by_af_nonref": (72.601, 65.674, 55.701)},
        "HG003": {"nonref": 73.3695, "het_recall": 61.6314, "by_af_nonref": (74.23, 69.807, 58.989)},
        "HG004": {"nonref": 68.2695, "het_recall": 52.5936, "by_af_nonref": (69.989, 54.399, 57.955)},
        "HG005": {"nonref": 73.372, "het_recall": 58.0941, "by_af_nonref": (74.672, 70.467, 51.326)},
        "HG006": {"nonref": 74.8632, "het_recall": 55.8584, "by_af_nonref": (76.396, 68.119, 52.375)},
        "HG007": {"nonref": 75.3753, "het_recall": 61.6517, "by_af_nonref": (77.246, 65.569, 51.581)},
    },
    "lcwgs": {
        "HG002": {"nonref": 98.1728, "het_recall": 97.5148, "by_af_nonref": (98.777, 95.59, 89.219)},
        "HG003": {"nonref": 98.1048, "het_recall": 97.5117, "by_af_nonref": (98.75, 95.453, 87.289)},
        "HG004": {"nonref": 97.861, "het_recall": 97.3017, "by_af_nonref": (98.464, 95.217, 89.915)},
        "HG005": {"nonref": 97.7136, "het_recall": 96.9163, "by_af_nonref": (98.42, 96.442, 85.29)},
        "HG006": {"nonref": 98.0764, "het_recall": 97.0916, "by_af_nonref": (98.815, 95.42, 86.355)},
        "HG007": {"nonref": 97.3235, "het_recall": 96.3844, "by_af_nonref": (98.256, 94.414, 82.517)},
    },
    "glimpse2": {
        "HG002": {"nonref": 97.705, "het_recall": 96.8436, "by_af_nonref": (98.314, 94.892, 88.951)},
        "HG003": {"nonref": 97.9265, "het_recall": 97.2667, "by_af_nonref": (98.599, 94.961, 86.842)},
        "HG004": {"nonref": 97.7072, "het_recall": 97.1006, "by_af_nonref": (98.271, 95.071, 90.517)},
        "HG005": {"nonref": 97.6243, "het_recall": 96.855, "by_af_nonref": (98.242, 95.971, 87.24)},
        "HG006": {"nonref": 98.0913, "het_recall": 97.0875, "by_af_nonref": (98.758, 95.195, 87.997)},
        "HG007": {"nonref": 97.3787, "het_recall": 96.4924, "by_af_nonref": (98.16, 94.831, 84.695)},
    },
    "selphi_native": {
        "HG002": {"nonref": 98.1694, "het_recall": 97.5018, "by_af_nonref": (98.758, 95.59, 89.441)},
        "HG003": {"nonref": 98.1225, "het_recall": 97.5548, "by_af_nonref": (98.744, 95.453, 87.767)},
        "HG004": {"nonref": 97.9235, "het_recall": 97.4168, "by_af_nonref": (98.517, 95.108, 90.445)},
        "HG005": {"nonref": 97.7301, "het_recall": 96.9758, "by_af_nonref": (98.376, 95.885, 87.044)},
        "HG006": {"nonref": 98.176, "het_recall": 97.2493, "by_af_nonref": (98.818, 95.42, 88.409)},
        "HG007": {"nonref": 97.4489, "het_recall": 96.6172, "by_af_nonref": (98.259, 94.539, 84.695)},
    },
    "selphi_isec": {
        "HG002": {"nonref": 98.1882, "het_recall": 97.5427, "by_af_nonref": (98.774, 95.59, 89.58)},
        "HG003": {"nonref": 98.1251, "het_recall": 97.5468, "by_af_nonref": (98.747, 95.453, 87.767)},
        "HG004": {"nonref": 97.8774, "het_recall": 97.3278, "by_af_nonref": (98.461, 95.217, 90.302)},
        "HG005": {"nonref": 97.8191, "het_recall": 97.0919, "by_af_nonref": (98.42, 96.442, 87.37)},
        "HG006": {"nonref": 98.1534, "het_recall": 97.2285, "by_af_nonref": (98.815, 95.42, 87.929)},
        "HG007": {"nonref": 97.4516, "het_recall": 96.5949, "by_af_nonref": (98.256, 94.414, 85.016)},
    },
}


def load_arm(arm):
    """Per-sample {hg: {nonref, het_recall, by_af_nonref}} for one arm; JSON first, fallback second."""
    out = {}
    for hg, na in SAMPLES:
        p = os.path.join(DATA22, f"{na}_conc_{arm}.json")
        try:
            with open(p) as fh:
                d = json.load(fh)
            out[hg] = {"nonref": float(d["nonref_concordance_pct"]),
                       "het_recall": float(d["het_recall_pct"]),
                       "by_af_nonref": tuple(float(d["by_af"][k]["nonref_pct"]) for k in STRATA_KEYS)}
        except (OSError, KeyError, ValueError, TypeError) as e:
            print(f"WARNING: {p}: {e}; using hardcoded fallback", file=sys.stderr)
            out[hg] = FALLBACK[arm][hg]
    return out


D = {arm: load_arm(arm) for arm in ARMS}
for arm in ARMS:  # flag a stale fallback loudly rather than silently diverging from the JSONs
    for hg in HGS:
        for k in ("nonref", "het_recall"):
            if abs(D[arm][hg][k] - FALLBACK[arm][hg][k]) > 5e-4:
                print(f"NOTE: FALLBACK stale for {arm}/{hg}/{k}: json {D[arm][hg][k]} vs "
                      f"fallback {FALLBACK[arm][hg][k]}", file=sys.stderr)

xs = np.arange(len(HGS))
fig0, ax0 = plt.subplots(2, 2, figsize=(9.2, 7.0))


def paired_panel(a, key, ylab, title):
    """(a)/(b): array-equivalent route (grey) vs likelihood engine, per sample, with +delta pp labels.
    The likelihood-engine values are the bcftools-likelihood runs on Selphi's own site list, so
    they take the bcftools-arm colour."""
    arr = np.array([D["hard"][h][key] for h in HGS])
    gl = np.array([D["lcwgs"][h][key] for h in HGS])
    w = 0.38
    a.bar(xs - w / 2, arr, w, color=C["array_route"],
          label=f"Array-equivalent route (mean {arr.mean():.1f}%)")
    a.bar(xs + w / 2, gl, w, color=C["sel_bcftools"],
          label=f"Likelihood engine, Selphi 2 --lcwgs (mean {gl.mean():.1f}%)")
    for i in range(len(HGS)):
        a.text(xs[i], max(arr[i], gl[i]) + 1.2, f"+{gl[i] - arr[i]:.1f} pp", ha="center",
               va="bottom", fontsize=7, fontweight="bold", color=C["sel_native"])
    a.set_ylim(0, 126)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_xticks(xs); a.set_xticklabels(HGS)
    a.set_ylabel(ylab); a.set_title(title, loc="left", fontweight="bold")
    a.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.2)
    wins = int((gl > arr).sum())
    a.annotate(f"mean +{(gl - arr).mean():.1f} pp, {wins}/{len(HGS)} samples",
               xy=(0.99, 0.985), xycoords="axes fraction", ha="right", va="top",
               fontsize=7, style="italic")
    a.grid(alpha=.25, lw=.5, axis="y")


paired_panel(ax0[0, 0], "nonref", "Non-reference concordance (%)",
             "(a) Array-equivalent route vs likelihood engine")
paired_panel(ax0[0, 1], "het_recall", "Heterozygote recall (%)",
             "(b) Heterozygote recall, same comparison")

# (c) Selphi 2 vs GLIMPSE2 on identical site lists: non-ref concordance, three arms
c = ax0[1, 0]
g2 = np.array([D["glimpse2"][h]["nonref"] for h in HGS])
nat = np.array([D["selphi_native"][h]["nonref"] for h in HGS])
bcf = np.array([D["selphi_isec"][h]["nonref"] for h in HGS])
w3 = 0.26
c.bar(xs - w3, g2, w3, color=C["glimpse2"], label="GLIMPSE2")
c.bar(xs, nat, w3, color=C["sel_native"], label="Selphi 2, native pileup + BAQ")
c.bar(xs + w3, bcf, w3, color=C["sel_bcftools"], label="Selphi 2, bcftools likelihoods")
lo = np.floor(min(g2.min(), nat.min(), bcf.min()) * 2) / 2      # zoom to the data range
hi = max(g2.max(), nat.max(), bcf.max())
for i in range(len(HGS)):
    c.text(xs[i], max(g2[i], nat[i], bcf[i]) + 0.025, f"{nat[i] - g2[i]:+.2f}", ha="center",
           va="bottom", fontsize=7, fontweight="bold", color=C["sel_native"])
c.set_ylim(lo, hi + 0.5)
c.set_xticks(xs); c.set_xticklabels(HGS)
c.set_ylabel("Non-reference concordance (%)")
c.set_title("(c) Selphi 2 vs GLIMPSE2, identical site lists", loc="left", fontweight="bold")
c.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.2)
c.annotate("labels: native − GLIMPSE2 (pp); y-axis truncated", xy=(0.99, 0.985),
           xycoords="axes fraction", ha="right", va="top", fontsize=6.5, style="italic")
c.grid(alpha=.25, lw=.5, axis="y")

# (d) per panel-MAF stratum: delta non-ref (Selphi 2 - GLIMPSE2, pp), both Selphi arms,
#     bars = mean over the six samples, points = per sample, k/6 = samples with delta > 0
d = ax0[1, 1]
xs3 = np.arange(len(STRATA_KEYS)); wd = 0.36
jit = np.linspace(-0.09, 0.09, len(HGS))
d_summary = {}
for j, (arm, col, lab) in enumerate([("selphi_native", C["sel_native"], "Selphi 2, native pileup + BAQ"),
                                     ("selphi_isec", C["sel_bcftools"], "Selphi 2, bcftools likelihoods")]):
    off = (j - 0.5) * wd
    per = np.array([[D[arm][h]["by_af_nonref"][s] - D["glimpse2"][h]["by_af_nonref"][s]
                     for h in HGS] for s in range(len(STRATA_KEYS))])       # strata x samples
    means = per.mean(axis=1); wins = (per > 0).sum(axis=1)
    d_summary[arm] = (per, means, wins)
    d.bar(xs3 + off, means, wd, color=col, label=lab, zorder=2)
    for s in range(len(STRATA_KEYS)):
        d.scatter(xs3[s] + off + jit, per[s], s=13, facecolor="white", edgecolor=col,
                  linewidth=0.9, zorder=4)
        top = max(per[s].max(), means[s], 0.0) + 0.05
        d.text(xs3[s] + off, top, f"{wins[s]}/{len(HGS)}", ha="center", va="bottom",
               fontsize=7, fontweight="bold", color=col)
d.axhline(0, color="black", lw=0.8, zorder=1)
allpts = np.concatenate([d_summary[a][0].ravel() for a in d_summary])
d.set_ylim(min(allpts.min(), 0) - 0.25, allpts.max() + 0.45)
d.set_xticks(xs3); d.set_xticklabels(STRATA_LBL)
d.set_ylabel("Δ non-reference concordance (pp)")
d.set_title("(d) Selphi 2 − GLIMPSE2 by reference-panel MAF", loc="left", fontweight="bold")
d.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.2)
d.annotate("bars: mean of six samples; points: per sample; k/6: samples with Δ > 0",
           xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
           fontsize=6.5, style="italic")
d.grid(alpha=.25, lw=.5, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/figure2_lcwgs_capture.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figure2_lcwgs_capture.png", dpi=600, bbox_inches="tight")
print("wrote figure2_lcwgs_capture.pdf/.png")
# plotted-value record (stdout), so the figure can be cross-checked against Tables 1, 1b, 1c
print("  (a) non-ref %  array-route / likelihood engine / delta pp:",
      "; ".join(f"{h} {D['hard'][h]['nonref']:.2f}/{D['lcwgs'][h]['nonref']:.2f}/"
                f"{D['lcwgs'][h]['nonref'] - D['hard'][h]['nonref']:+.2f}" for h in HGS))
print("  (b) het recall % array-route / likelihood engine / delta pp:",
      "; ".join(f"{h} {D['hard'][h]['het_recall']:.2f}/{D['lcwgs'][h]['het_recall']:.2f}/"
                f"{D['lcwgs'][h]['het_recall'] - D['hard'][h]['het_recall']:+.2f}" for h in HGS))
print("  (c) non-ref %  GLIMPSE2 / native+BAQ / bcftools-GL (native-G2 pp):",
      "; ".join(f"{h} {g:.3f}/{n:.3f}/{b:.3f} ({n - g:+.3f})" for h, g, n, b in zip(HGS, g2, nat, bcf)))
for arm, (per, means, wins) in d_summary.items():
    print(f"  (d) {arm:14s} delta non-ref pp by stratum (mean, wins):",
          "; ".join(f"{k} {m:+.3f} {wn}/{len(HGS)}" for k, m, wn in zip(STRATA_KEYS, means, wins)))

# =====================================================================================
# ---- Figure 3: array-imputation accuracy (4 panels) ----
# =====================================================================================
maf_lbl = ["0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50"]
x = np.arange(len(maf_lbl))

# ---- data (from paper tables) ----
# 3a: 1KG genome-wide (20 autosomes) R2 by MAF, five-way, imputation-only (n-weighted;
#     all tools impute from the identical phased target + panel; Supplementary Table S9)
t1 = {"selphi":   [.3528,.4222,.5235,.6250,.6910,.7618,.8525,.8962,.9254],
      "selphi153":[.3375,.4106,.5149,.6207,.6898,.7629,.8539,.8973,.9262],
      "beagle":   [.3570,.4151,.5092,.6083,.6742,.7460,.8414,.8885,.9196],
      "impute5":  [.3458,.4062,.5005,.5991,.6654,.7381,.8355,.8840,.9160],
      "minimac4": [.3529,.4132,.5048,.6013,.6663,.7367,.8323,.8803,.9112]}
# 3c: HGDP out-of-panel per-region per-sample R2 (Table 4c genome-wide)
regions = ["Oceanian*","Mid-East*","African","E.Asian","C/S.Asian","European","Adm.Amer."]
hgdp_s = [.8984,.9386,.8778,.9506,.9465,.9572,.9626]
hgdp_b = [.8880,.9326,.8652,.9431,.9418,.9532,.9569]
# 3d: lcWGS coverage sweep per-sample R2 (Table 2b) — Selphi native --bam errmod GL (beats GLIMPSE2 and QUILT2 at every coverage)
cov = [0.5,1,2,4]
lc = {"selphi":[.9924,.9950,.9971,.9979],"glimpse2":[.9916,.9945,.9967,.9975],"quilt2":[.9919,.9944,.9968,.9973]}

fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))

# (a) 1KG five-way, imputation-only (all tools from the identical phased target + panel)
a = ax[0,0]
for k,lab,m,lw in [("impute5","IMPUTE5","^",1.4),("minimac4","Minimac4","D",1.4),
                   ("beagle","Beagle 5.5","s",1.4),("selphi153","Selphi 1.5.3","v",1.4),
                   ("selphi","Selphi 2","o",2.1)]:
    a.plot(x, t1[k], marker=m, ms=4, lw=lw, color=C[k], label=lab,
           zorder=5 if k=="selphi" else 3)
a.set_xticks(x); a.set_xticklabels(maf_lbl, rotation=45, ha="right")
a.set_ylabel("Imputation R²"); a.set_xlabel("Minor allele frequency (%)")
a.set_title("(a) 1000 Genomes genome-wide, imputation-only", loc="left", fontweight="bold")
a.legend(frameon=False, loc="lower right"); a.grid(alpha=.25, lw=.5)

# (b) MESA per-sample R2 by ancestry: Selphi 2 vs Beagle 5.5 (competitive, per-sample)
# per-sample-mean R2 by MESA ancestry group at the panel-adaptive mc=132,676 run;
# per-group deltas (+0.019..+0.032) match the Table 4b / Results prose.
b = ax[0,1]
pop_lbl = ["African-\nAmerican","Hispanic","European\n(White)","East-Asian"]
sel_r2  = [0.8936, 0.8975, 0.9026, 0.8925]
bea_r2  = [0.8733, 0.8763, 0.8835, 0.8600]
xp = np.arange(4); wp = 0.38
b.bar(xp-wp/2, sel_r2, wp, color=C["selphi"], label="Selphi 2")
b.bar(xp+wp/2, bea_r2, wp, color=C["beagle"], label="Beagle 5.5")
for i in range(4):
    b.text(xp[i], max(sel_r2[i],bea_r2[i])+0.0015, f"+{sel_r2[i]-bea_r2[i]:.3f}",
           ha="center", va="bottom", fontsize=7, color=C["selphi"], fontweight="bold")
b.set_ylim(0.84, 0.918); b.set_xticks(xp); b.set_xticklabels(pop_lbl, fontsize=7.5)
b.set_ylabel("Per-sample R² (MESA, chr20)")
b.set_title("(b) Per-sample accuracy by ancestry, MESA", loc="left", fontweight="bold")
b.legend(frameon=False, loc="upper right", ncol=1); b.grid(alpha=.25, lw=.5, axis="y")

# (c) HGDP out-of-panel
c = ax[1,0]
xr = np.arange(len(regions)); w=0.38
c.bar(xr-w/2, hgdp_s, w, color=C["selphi"], label="Selphi 2")
c.bar(xr+w/2, hgdp_b, w, color=C["beagle"], label="Beagle 5.5")
c.set_xticks(xr); c.set_xticklabels(regions, rotation=45, ha="right")
c.set_ylim(0.84,0.975); c.set_ylabel("Per-sample R²")
c.set_title("(c) HGDP out-of-panel, genome-wide", loc="left", fontweight="bold")
c.legend(frameon=False, loc="upper left"); c.grid(alpha=.25, lw=.5, axis="y")
c.annotate("*absent from panel", xy=(0.02,0.02), xycoords="axes fraction", fontsize=7, style="italic")

# (d) lcWGS coverage sweep
d = ax[1,1]
for k,lab,m in [("selphi","Selphi 2","o"),("glimpse2","GLIMPSE2","s"),("quilt2","QUILT2","^")]:
    d.plot(cov, lc[k], marker=m, ms=5, lw=1.6, color=C[k], label=lab)
d.set_xticks(cov); d.set_xlabel("Coverage (×)"); d.set_ylabel("Per-sample R²")
d.set_title("(d) lcWGS coverage sweep (GIAB)", loc="left", fontweight="bold")
d.legend(frameon=False, loc="lower right"); d.grid(alpha=.25, lw=.5)

plt.tight_layout()
plt.savefig(f"{OUT}/figure3_accuracy.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figure3_accuracy.png", dpi=600, bbox_inches="tight")
print("wrote figure3_accuracy.pdf/.png")

# =====================================================================================
# ---- Figure 4: efficiency (3 panels) ----
# =====================================================================================
fig2, (axl, axw, axm) = plt.subplots(1, 3, figsize=(11.5, 3.6))
# (a) lcWGS wall time per run, whole chr22, 16 threads, log scale, two labelled clusters.
#     Capture-library cluster (4,796-hap panel, BAM in -> imputed out), quiet machine,
#     2026-09-02, after the shared conditioning pack (commit 9d4b398): Selphi 2 native
#     pileup 104 s (5.0 GB peak) vs GLIMPSE2 327 s for one sample; 170 s vs 389 s for six
#     samples in one run.
#     Downsampled cluster (1x, 6,332-hap panel, 1 sample): Selphi 115 s, GLIMPSE2 287 s
#     (15 native chunks + ligate), QUILT2 1,729 s (tiled) -- the former Figure 3c values,
#     unchanged. No scaling.
tool_col = {"Selphi 2": C["selphi"], "GLIMPSE2": C["glimpse2"], "QUILT2": C["quilt2"]}
speed_groups = [  # (x0, tick label, [(tool, seconds), ...])
    (0.0, "1 sample",            [("Selphi 2", 104), ("GLIMPSE2", 327)]),
    (2.4, "6 samples\n(one run)", [("Selphi 2", 170), ("GLIMPSE2", 389)]),
    (5.3, "1 sample",            [("Selphi 2", 115), ("GLIMPSE2", 287), ("QUILT2", 1729)]),
]
bw = 0.78; tick_pos = []; tick_lbl = []; seen = set()
for x0, glab, bars_ in speed_groups:
    for i, (tool, sec) in enumerate(bars_):
        xb = x0 + i
        axl.bar(xb, sec, bw, color=tool_col[tool], label=tool if tool not in seen else None)
        seen.add(tool)
        axl.text(xb, sec * 1.10, f"{sec:,} s", ha="center", va="bottom", fontsize=7)
    tick_pos.append(x0 + (len(bars_) - 1) / 2); tick_lbl.append(glab)
axl.set_yscale("log"); axl.set_ylim(50, 7000)
axl.set_xticks(tick_pos); axl.set_xticklabels(tick_lbl, fontsize=7.5)
axl.set_xlim(-0.6, 7.9)
axl.axvline(4.55, color="#888888", lw=0.7, ls="--")
axl.annotate("Capture libraries\n4,796-hap panel, BAM in → imputed out", xy=(1.7, -0.19),
             xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7)
axl.annotate("Downsampled 1×\n6,332-hap panel", xy=(6.3, -0.19),
             xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7)
axl.set_ylabel("Wall time (s, log scale)")
axl.set_title("(a) lcWGS speed, whole chr22, 16 threads", loc="left", fontweight="bold")
axl.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2, borderaxespad=0.2)
axl.grid(alpha=.25, lw=.5, axis="y")
tt = ["Selphi 2", "Beagle 5.5"]; xt = np.arange(2); tcol2 = [C["selphi"], C["beagle"]]
# (b) whole-genome wall time: 6-sample array, 22 autosomes, full pipeline, 16 threads
wg_wall = [27.7, 36.4]
axw.bar(xt, wg_wall, 0.55, color=tcol2)
for i,v in enumerate(wg_wall): axw.text(i, v+0.4, f"{v} min", ha="center", fontsize=8.5, fontweight="bold")
axw.set_xticks(xt); axw.set_xticklabels(tt); axw.set_ylabel("Whole-genome wall time (min)")
axw.set_ylim(0, 41); axw.set_title("(b) Whole-genome wall time", loc="left", fontweight="bold")
axw.grid(alpha=.25, lw=.5, axis="y")
# (c) peak memory, one chromosome at a time (peak = largest chromosome)
wg_mem = [25, 40]
axm.bar(xt, wg_mem, 0.55, color=tcol2)
for i,v in enumerate(wg_mem): axm.text(i, v+0.5, f"{v} GB", ha="center", fontsize=8.5, fontweight="bold")
axm.set_xticks(xt); axm.set_xticklabels(tt); axm.set_ylabel("Peak memory (GB, per chromosome)")
axm.set_ylim(0, 47); axm.set_title("(c) Peak memory", loc="left", fontweight="bold")
axm.grid(alpha=.25, lw=.5, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/figure4_efficiency.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figure4_efficiency.png", dpi=600, bbox_inches="tight")
print("wrote figure4_efficiency.pdf/.png")
print("  (a) capture 1 sample: Selphi 104 s / GLIMPSE2 327 s; 6 samples one run: 170 s / 389 s;"
      " downsampled 1x: 115 / 287 / 1,729 s")

# =====================================================================================
# ---- Supplementary figure: replication across chromosomes ----
#      Selphi 2 native pileup + BAQ minus GLIMPSE2, delta non-ref concordance (pp), identical
#      per-sample site lists, typed sites excluded; overall + three panel-MAF strata.
# =====================================================================================
CHRS = ["chr22", "chr20", "chr10", "chr1"]
CAT_LBL = ["Overall", "MAF ≥ 5%", "MAF 0.5–5%", "MAF < 0.5%"]
# Fallback = per-sample (overall, common, low, rare) delta pp recomputed from the JSONs on
# 2026-09-02; used only when a per-sample JSON pair is missing or unreadable.
FALLBACK_S = {
    "chr22": {"HG002": (0.4644, 0.444, 0.698, 0.49), "HG003": (0.196, 0.145, 0.492, 0.925), "HG004": (0.2163, 0.246, 0.037, -0.072), "HG005": (0.1058, 0.134, -0.086, -0.196), "HG006": (0.0847, 0.06, 0.225, 0.412), "HG007": (0.0702, 0.099, -0.292, 0.0)},
    "chr20": {"HG002": (0.5064, 0.457, 0.882, 0.858), "HG003": (0.1221, 0.101, -0.08, 1.021), "HG004": (0.0726, 0.062, 0.092, 0.289), "HG005": (0.163, 0.16, 0.446, -0.268), "HG006": (0.1224, 0.052, 0.306, 1.331), "HG007": (0.1142, 0.14, 0.06, -0.338)},
    "chr10": {"HG002": (0.441, 0.397, 0.683, 0.86), "HG003": (0.0605, 0.038, 0.183, 0.254), "HG004": (0.1698, 0.152, 0.29, 0.29), "HG005": (0.2615, 0.261, 0.191, 0.431), "HG006": (0.171, 0.152, 0.16, 0.608), "HG007": (0.1456, 0.173, -0.074, 0.032)},
    "chr1": {"HG002": (0.6514, 0.582, 1.043, 1.089), "HG003": (0.1284, 0.095, 0.213, 0.608), "HG004": (0.1733, 0.158, 0.19, 0.464), "HG005": (0.1778, 0.13, 0.237, 0.939), "HG006": (0.2914, 0.29, 0.393, 0.088), "HG007": (0.1425, 0.108, 0.162, 0.744)},
}


def load_chr_delta(chrom):
    """{hg: {"delta": (overall, common, low, rare) pp, "sites": n or None}} for native+BAQ - GLIMPSE2."""
    out = {}
    for hg, na in SAMPLES:
        base = os.path.join(HERE, "figures", "data", chrom)
        pn = os.path.join(base, f"{na}_conc_selphi_native.json")
        pg = os.path.join(base, f"{na}_conc_glimpse2.json")
        try:
            with open(pn) as fh:
                n = json.load(fh)
            with open(pg) as fh:
                g = json.load(fh)
            if n["sites_evaluated"] != g["sites_evaluated"]:
                raise ValueError(f"site lists differ ({n['sites_evaluated']} vs {g['sites_evaluated']})")
            delta = (float(n["nonref_concordance_pct"]) - float(g["nonref_concordance_pct"]),) + tuple(
                float(n["by_af"][k]["nonref_pct"]) - float(g["by_af"][k]["nonref_pct"]) for k in STRATA_KEYS)
            out[hg] = {"delta": delta, "sites": int(n["sites_evaluated"])}
        except (OSError, KeyError, ValueError, TypeError) as e:
            print(f"WARNING: {chrom}/{na}: {e}; using hardcoded fallback", file=sys.stderr)
            out[hg] = {"delta": FALLBACK_S[chrom][hg], "sites": None}
    return out


DS = {ch: load_chr_delta(ch) for ch in CHRS}
try:  # cross-check the recomputed means / wins against the summary JSON (arms.native)
    with open(os.path.join(HERE, "figures", "data", "paper_numbers_multichr.json")) as fh:
        MJ = json.load(fh)["chromosomes"]
except (OSError, KeyError, ValueError) as e:
    print(f"NOTE: paper_numbers_multichr.json not cross-checked: {e}", file=sys.stderr); MJ = None

figS, axS = plt.subplots(1, len(CHRS), figsize=(11.5, 3.4), sharey=True)
xs4 = np.arange(len(CAT_LBL)); jit4 = np.linspace(-0.13, 0.13, len(HGS))
s_summary = {}
for pi, (ch, a) in enumerate(zip(CHRS, axS)):
    per = np.array([DS[ch][h]["delta"] for h in HGS])          # samples x 4
    means = per.mean(axis=0); wins = (per > 0).sum(axis=0)
    s_summary[ch] = (per, means, wins)
    if MJ is not None and ch in MJ:
        A = MJ[ch]["arms"]["native"]
        jm = [A["nonref_delta"]["mean"]] + [A[f"nonref_delta_{k}"]["mean"] for k in STRATA_KEYS]
        jw = [A["nonref_delta"]["wins"]] + [A[f"nonref_delta_{k}"]["wins"] for k in STRATA_KEYS]
        for k, (m1, m2, w1, w2) in enumerate(zip(means, jm, wins, jw)):
            if abs(m1 - m2) > 2e-3 or int(w1) != int(w2):
                print(f"NOTE: {ch} {CAT_LBL[k]}: recomputed {m1:+.4f} ({w1}/6) vs summary JSON "
                      f"{m2:+.4f} ({w2}/6)", file=sys.stderr)
    a.bar(xs4, means, 0.62, color=C["sel_native"], zorder=2)
    for k in range(len(CAT_LBL)):
        a.scatter(xs4[k] + jit4, per[:, k], s=13, facecolor="white", edgecolor=C["sel_native"],
                  linewidth=0.9, zorder=4)
        a.text(xs4[k], max(per[:, k].max(), means[k], 0.0) + 0.06, f"{wins[k]}/{len(HGS)}",
               ha="center", va="bottom", fontsize=7, fontweight="bold", color=C["sel_native"])
    a.axhline(0, color="black", lw=0.8, zorder=1)
    a.set_xticks(xs4); a.set_xticklabels(CAT_LBL, rotation=30, ha="right", fontsize=7.5)
    a.set_title(f"({'abcd'[pi]}) {ch}", loc="left", fontweight="bold")
    sites = [DS[ch][h]["sites"] for h in HGS]
    if all(v is not None for v in sites):
        a.annotate(f"{min(sites):,}–{max(sites):,} sites/sample", xy=(0.98, 0.98),
                   xycoords="axes fraction", ha="right", va="top", fontsize=6.5, style="italic")
    a.grid(alpha=.25, lw=.5, axis="y")
    if pi > 0:
        a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
allS = np.concatenate([s_summary[ch][0].ravel() for ch in CHRS])
axS[0].set_ylim(min(allS.min(), 0) - 0.25, allS.max() + 0.35)
axS[0].set_ylabel("Δ non-reference concordance (pp)\nSelphi 2 native + BAQ − GLIMPSE2")
plt.tight_layout()
figS.text(0.005, -0.03, "bars: mean of six samples; points: per sample; k/6: samples with Δ > 0; "
          "identical per-sample site lists, typed sites excluded", ha="left", va="top",
          fontsize=6.5, style="italic")
plt.savefig(f"{OUT}/figureS_replication.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figureS_replication.png", dpi=600, bbox_inches="tight")
print("wrote figureS_replication.pdf/.png")
for ch, (per, means, wins) in s_summary.items():
    print(f"  {ch:5s} mean delta pp (wins):",
          "; ".join(f"{lab} {m:+.3f} ({w}/{len(HGS)})" for lab, m, w in zip(CAT_LBL, means, wins)))
