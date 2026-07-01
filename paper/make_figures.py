#!/usr/bin/env python3
"""Publication figures for the Selphi 2 paper. All values are the measured
numbers in the paper tables (1, 2b, 2c, 6b, 5/3b)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/data/projects/.claude_home/gt/selphi/mayor/rig/paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
# Okabe-Ito colorblind-safe palette
C = {"selphi": "#0072B2", "beagle": "#E69F00", "impute5": "#009E73",
     "minimac4": "#CC79A7", "glimpse2": "#E69F00", "quilt2": "#009E73",
     "afram": "#D55E00", "hisp": "#CC79A7", "white": "#0072B2", "eas": "#009E73"}

maf_lbl = ["0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-20", "20-50"]
x = np.arange(len(maf_lbl))

# ---- data (from paper tables) ----
# 2a: 1KG genome-wide (20 autosomes) R2 by MAF, four-way (n-weighted)
t1 = {"selphi":[.3578,.4275,.5274,.6273,.6926,.7630,.8531,.8965,.9255],
      "beagle":[.3619,.4196,.5117,.6090,.6742,.7458,.8410,.8881,.9192],
      "impute5":[.3458,.4061,.5005,.5991,.6654,.7381,.8355,.8840,.9160],
      "minimac4":[.3530,.4131,.5048,.6013,.6663,.7367,.8323,.8803,.9112]}
# 2b: MESA candidate-cap RECOVERY (adaptive - fixed) by ancestry x MAF (Table 2b)
t2b = {"afram":[.1391,.1211,.0885,.0642,.0528,.0436,.0343,.0266,.0173],
       "hisp":[.1715,.1554,.1268,.0968,.0692,.0436,.0255,.0199,.0125],
       "white":[.1195,.0858,.0587,.0381,.0242,.0141,.0128,.0114,.0063],
       "eas":[.0170,.0123,.0055,.0008,-.0034,-.0055,-.0060,-.0059,-.0038]}
# 2c: HGDP out-of-panel per-region per-sample R2 (Table 2c genome-wide)
regions = ["Oceanian*","Mid-East*","African","E.Asian","C/S.Asian","European","Adm.Amer."]
hgdp_s = [.8984,.9386,.8778,.9506,.9465,.9572,.9626]
hgdp_b = [.8880,.9326,.8652,.9431,.9418,.9532,.9569]
# 2d: lcWGS coverage sweep per-sample R2 (Table 6b) — Selphi native --bam errmod GL (beats GLIMPSE2 and QUILT2 at every coverage)
cov = [0.5,1,2,4]
lc = {"selphi":[.9924,.9950,.9971,.9979],"glimpse2":[.9916,.9945,.9967,.9975],"quilt2":[.9919,.9944,.9968,.9973]}

fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))

# (a) 1KG four-way
a = ax[0,0]
for k,lab,m in [("selphi","Selphi 2","o"),("beagle","Beagle 5.5","s"),("impute5","IMPUTE5","^"),("minimac4","Minimac4","D")]:
    a.plot(x, t1[k], marker=m, ms=4, lw=1.6, color=C[k], label=lab)
a.set_xticks(x); a.set_xticklabels(maf_lbl, rotation=45, ha="right")
a.set_ylabel("Imputation R²"); a.set_xlabel("Minor allele frequency (%)")
a.set_title("(a) 1000 Genomes genome-wide, four-way", loc="left", fontweight="bold")
a.legend(frameon=False, loc="lower right"); a.grid(alpha=.25, lw=.5)

# (b) MESA per-sample R2 by ancestry: Selphi 2 vs Beagle 5.5 (competitive, per-sample)
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
plt.savefig(f"{OUT}/figure2_accuracy.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figure2_accuracy.png", dpi=600, bbox_inches="tight")
print("wrote figure2_accuracy.pdf/.png")

# ---- Figure 3: efficiency (3 panels) ----
fig2, (axw, axm, axl) = plt.subplots(1, 3, figsize=(11.5, 3.6))
tt = ["Selphi 2", "Beagle 5.5"]; xt = np.arange(2); tcol2 = [C["selphi"], C["beagle"]]
# (a) whole-genome wall time: 6-sample array, 22 autosomes, full pipeline, 16 threads
wg_wall = [27.7, 36.4]
axw.bar(xt, wg_wall, 0.55, color=tcol2)
for i,v in enumerate(wg_wall): axw.text(i, v+0.4, f"{v} min", ha="center", fontsize=8.5, fontweight="bold")
axw.set_xticks(xt); axw.set_xticklabels(tt); axw.set_ylabel("Whole-genome wall time (min)")
axw.set_ylim(0, 41); axw.set_title("(a) Whole-genome wall time", loc="left", fontweight="bold")
axw.grid(alpha=.25, lw=.5, axis="y")
# (b) peak memory, one chromosome at a time (peak = largest chromosome)
wg_mem = [25, 40]
axm.bar(xt, wg_mem, 0.55, color=tcol2)
for i,v in enumerate(wg_mem): axm.text(i, v+0.5, f"{v} GB", ha="center", fontsize=8.5, fontweight="bold")
axm.set_xticks(xt); axm.set_xticklabels(tt); axm.set_ylabel("Peak memory (GB, per chromosome)")
axm.set_ylim(0, 47); axm.set_title("(b) Peak memory", loc="left", fontweight="bold")
axm.grid(alpha=.25, lw=.5, axis="y")
# (c) lcWGS per-sample wall time, log scale. ALL three tools measured on the WHOLE
# chr22 (1 sample, 1x, 16 threads, same machine): Selphi 115 s, GLIMPSE2 287 s
# (15 native chunks + ligate), QUILT2 1729 s (tiled). No scaling.
tools=["Selphi 2","GLIMPSE2","QUILT2"]
twall=[115,287,1729]; tcol=[C["selphi"],C["glimpse2"],C["quilt2"]]
bars=axl.bar(np.arange(3), twall, 0.6, color=tcol)
axl.set_yscale("log"); axl.set_xticks(np.arange(3)); axl.set_xticklabels(tools)
axl.set_ylabel("Wall time per sample (s, log scale)")
axl.set_title("(c) lcWGS speed, whole chr22 (1 sample)", loc="left", fontweight="bold")
lbls=["115 s","287 s","1,729 s"]
for b,v,t in zip(bars,twall,lbls): axl.text(b.get_x()+b.get_width()/2, v*1.12, t, ha="center", fontsize=8)
axl.annotate("whole chr22, 1× coverage, 16 threads", xy=(0.02,0.93), xycoords="axes fraction", fontsize=6.5, style="italic")
axl.grid(alpha=.25, lw=.5, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/figure3_efficiency.pdf", bbox_inches="tight")
plt.savefig(f"{OUT}/figure3_efficiency.png", dpi=600, bbox_inches="tight")
print("wrote figure3_efficiency.pdf/.png")
