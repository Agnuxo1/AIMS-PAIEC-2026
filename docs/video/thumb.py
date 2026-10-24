"""Miniatura 1280x720 para YouTube, generada desde los resultados reales."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch

REPO=r"E:\AIMS-sprint\github\AIMS-PAIEC-2026"
INK="#0f1b2d"; ACCENT="#1b6ca8"; WARN="#c1121f"; GOOD="#2a9d8f"; MUTED="#8d99ae"; GRID="#e8ebf0"
s=pd.read_csv(os.path.join(REPO,"results","summary.csv")).set_index("method")
B=[0,1,3,7,15,31]; x=np.log2(np.array(B)+1)

fig=plt.figure(figsize=(12.8,7.2),dpi=100); fig.patch.set_facecolor("white")
ax=fig.add_axes([0.07,0.155,0.52,0.585])
ax.plot(x,[s.loc["empirical_mean",f"B{b}"] for b in B],"-o",color=WARN,lw=6,ms=13)
ax.plot(x,[s.loc["original_beta3",f"B{b}"] for b in B],"-o",color=ACCENT,lw=7,ms=13)
ax.set_xticks(x); ax.set_xticklabels([str(b) for b in B],fontsize=15)
ax.set_ylim(0.17,0.37); ax.set_yticks([0.20,0.25,0.30,0.35])
ax.tick_params(labelsize=14); ax.grid(color=GRID,lw=1.4); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
ax.set_xlabel("labels revealed",fontsize=17,color=INK)
ax.set_ylabel("Brier score",fontsize=17,color=INK)
ax.annotate("the reference\nbreaks here",xy=(x[1],s.loc["empirical_mean","B1"]),
            xytext=(x[2]+0.1,0.335),fontsize=19,color=WARN,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=WARN,lw=3))
fig.text(0.07,0.885,"Four lines of arithmetic",fontsize=50,fontweight="bold",color=INK,va="top")
fig.text(0.07,0.795,"beat the official reference by 13%",fontsize=29,color=MUTED,va="top")
bx=fig.add_axes([0.635,0.12,0.30,0.58]); bx.axis("off")
for i,(lab,val,col) in enumerate([("official reference",f"{s.loc['empirical_mean','brier_alc']:.4f}",WARN),
                                  ("submitted",f"{s.loc['original_beta3','brier_alc']:.4f}",ACCENT)]):
    y=0.86-i*0.30
    bx.text(0,y,lab,fontsize=17,color=MUTED,va="top",transform=bx.transAxes)
    bx.text(0,y-0.08,val,fontsize=52,color=col,va="top",fontweight="bold",transform=bx.transAxes)
bx.add_patch(FancyBboxPatch((0.0,0.02),0.98,0.19,boxstyle="round,pad=0.02",
             facecolor="#eef7f5",edgecolor=GOOD,lw=2.5,transform=bx.transAxes))
bx.text(0.49,0.155,"133 of 151 benchmarks",ha="center",fontsize=19,color=GOOD,
        fontweight="bold",transform=bx.transAxes)
bx.text(0.49,0.065,"NeurIPS 2026 competition entry",ha="center",fontsize=14,
        color=MUTED,transform=bx.transAxes)
fig.text(0.635,0.045,"local validation, not an official score",fontsize=14,color=MUTED)
fig.savefig(r"E:\AIMS-sprint\video\out\thumbnail.png",facecolor="white")
print("miniatura 1280x720 escrita")
