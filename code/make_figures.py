import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, glob, numpy as np, os, cv2

TW = os.path.join(os.path.dirname(__file__), "..", "results")
RV = os.environ.get("YCB_REF_VIEWS", "ref_views_16")   # dat bien moi truong tro toi du lieu YCB
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.grid":True,"grid.alpha":0.3,
 "grid.linestyle":":","figure.dpi":220,"axes.linewidth":0.8,"legend.frameon":True,
 "legend.fontsize":7.2,"legend.borderpad":0.35})
Ks=[1,3,5,10,20,50]

b=pd.read_csv(f"{TW}/breadth.csv"); b=b[b.ok]
D=pd.concat([pd.read_csv(f) for f in glob.glob(f"{TW}/decisive_ob*.csv")]); D=D[D.ok]
a=pd.read_csv(f"{TW}/anchor_full.csv"); a=a[a.ok]
AS, AT, AK = 100*a.adds_pass.mean(), a.time_ms.mean(), 252

dS=D.groupby(['strategy','top_k']).adds_pass.mean().unstack()*100
bS=b.groupby(['strategy','top_k']).adds_pass.mean().unstack()*100
bT=b.groupby(['strategy','top_k']).time_ms.mean().unstack()
dT=D.groupby(['strategy','top_k']).time_ms.mean().unstack()

C={'farthest':'#1f77b4','random':'#2ca02c','dino':'#d62728'}

# ================= FIG 1: accuracy — KHONG CO CHU TRONG HINH =================
fig,ax=plt.subplots(figsize=(3.4,2.55))
ax.axhline(AS,color='k',ls='--',lw=1.3,label='Full grid, $K$=252')
ax.plot(Ks,[dS.loc['farthest',k] for k in Ks],'-o',ms=4,lw=1.4,color=C['farthest'],label='Farthest (proposed)')
ax.plot(Ks,[dS.loc['random',k]   for k in Ks],'-s',ms=4,lw=1.2,color=C['random'],  label='Random')
ax.plot(Ks,[dS.loc['dino',k]     for k in Ks],'-^',ms=4,lw=1.2,color=C['dino'],    label='DINOv2 BoW')
ax.set_xscale('log'); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.set_xlabel('Retained hypotheses $K$'); ax.set_ylabel('ADD-S recall (%)')
ax.set_ylim(20,104); ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(f"{OUT}/rfig1_accuracy.png",bbox_inches='tight'); plt.close(fig)

# ================= FIG 2: time — mo hinh di qua anchor =================
c0, c1 = 192.0, 17.0        # mo hinh cong bo, di qua anchor K=252 (192+17*252=4476 ~ 4486)
print(f"FIT: t = {c0:.0f} + {c1:.1f} K   (K=20 -> {c0+20*c1:.0f} ms, thuc te {bT.loc['farthest',20]:.0f} ms)")

fig,ax=plt.subplots(figsize=(3.4,2.55))
xs=np.logspace(0,np.log10(260),200)
ax.plot(xs,c0+c1*xs,'--',lw=1.3,color='0.35',label=f'Model $t\\approx{c0:.0f}+{c1:.1f}K$')
for s,mk,lb in [('farthest','-o','Farthest (proposed)'),('random','-s','Random'),('dino','-^','DINOv2 BoW')]:
    ax.plot(Ks,[dT.loc[s,k] for k in Ks],mk,ms=4,lw=1.2,color=C[s],label=lb)
ax.plot([AK],[AT],'D',ms=6,color='k',label=f'Full grid, $K$=252')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([1,3,5,10,20,50,252]); ax.set_xticklabels([1,3,5,10,20,50,252])
ax.set_xlabel('Retained hypotheses $K$'); ax.set_ylabel('Frame-0 registration time (ms)')
ax.legend(loc='upper left'); fig.tight_layout()
fig.savefig(f"{OUT}/rfig2_time.png",bbox_inches='tight'); plt.close(fig)

# ================= FIG 3: Pareto — bo dau sao, duong full grid dam =================
fig,ax=plt.subplots(figsize=(3.4,2.55))
ax.axhline(AS,color='k',ls='--',lw=1.8,zorder=1,
           label='Full grid, $K$=252')
ax.plot(bT.loc['farthest',Ks],bS.loc['farthest',Ks],'-o',ms=4.5,lw=1.5,color=C['farthest'],
        zorder=3,label='Farthest (proposed)')
ax.plot(bT.loc['random',Ks],bS.loc['random',Ks],'-s',ms=4,lw=1.2,color=C['random'],
        zorder=2,label='Random')
ax.set_xscale('log'); ax.set_xlabel('Frame-0 registration time (ms)')
ax.set_ylabel('ADD-S recall (%)'); ax.set_ylim(30,104)
ax.set_xlim(230,6000)
ax.set_xticks([300,500,1000,2000,4486]); ax.set_xticklabels(['300','500','1000','2000','4486'])
ax.tick_params(axis='x',labelsize=7)
ax.legend(loc='lower right'); fig.tight_layout()
fig.savefig(f"{OUT}/rfig3_pareto.png",bbox_inches='tight'); plt.close(fig)

# ================= FIG 4: per-object bar — khong chu de len nhau =================
po=(b[(b.strategy=='farthest')&(b.top_k==20)].groupby('obj').adds_pass.mean()*100)
po.index=[f"ob{o[-2:]}" for o in po.index]; po=po.sort_index()
fig,ax=plt.subplots(figsize=(3.4,2.55))
cols=['#d62728' if v<100 else '#1f77b4' for v in po.values]
ax.bar(range(len(po)),po.values,color=cols,width=0.72)
ax.set_xticks(range(len(po))); ax.set_xticklabels(po.index,rotation=90,fontsize=5.6)
ax.set_ylim(85,101); ax.set_ylabel('ADD-S recall (%)')
ax.set_xlabel('Object'); ax.axhline(100,color='0.45',ls=':',lw=0.9)
fig.tight_layout(); fig.savefig(f"{OUT}/rfig4_perobj.png",bbox_inches='tight'); plt.close(fig)

# ================= FIG 5 (MOI): anh 21 vat the =================
def crop(oid):
    d=f"{RV}/ob_{oid:07d}"
    rf=sorted(glob.glob(f"{d}/rgb/*.png")); mf=sorted(glob.glob(f"{d}/mask/*.png"))
    if not rf: return None
    img=cv2.cvtColor(cv2.imread(rf[0]),cv2.COLOR_BGR2RGB)
    if mf:
        m=cv2.imread(mf[0],0)>0
        if m.any():
            ys,xs=np.where(m); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
            pad=int(0.12*max(y1-y0,x1-x0))
            y0=max(0,y0-pad); x0=max(0,x0-pad)
            y1=min(img.shape[0]-1,y1+pad); x1=min(img.shape[1]-1,x1+pad)
            img=img[y0:y1+1,x0:x1+1]
            mm=m[y0:y1+1,x0:x1+1]
            img=img.copy(); img[~mm]=(255,255,255)      # nen trang cho sach
    h,w=img.shape[:2]; s=max(h,w)
    sq=np.full((s,s,3),255,np.uint8); sq[(s-h)//2:(s-h)//2+h,(s-w)//2:(s-w)//2+w]=img
    return cv2.resize(sq,(150,150))

if not os.path.isdir(RV):
    print("[bo qua] Fig. objects: khong tim thay du lieu YCB tai", RV,
          "\n           dat bien moi truong YCB_REF_VIEWS neu muon dung hinh nay.")
    raise SystemExit(0)

HARD={15,18}
fig,axes=plt.subplots(3,7,figsize=(7.0,3.3))
for i,ax in enumerate(axes.ravel(),start=1):
    im=crop(i); ax.set_xticks([]); ax.set_yticks([])
    if im is not None: ax.imshow(im)
    lbl=f"ob{i:02d}"
    if i in HARD:
        for sp in ax.spines.values(): sp.set_edgecolor('#d62728'); sp.set_linewidth(2.0)
        ax.set_xlabel(lbl+"*",fontsize=7,color='#d62728',labelpad=1.5,fontweight='bold')
    else:
        for sp in ax.spines.values(): sp.set_edgecolor('0.75'); sp.set_linewidth(0.6)
        ax.set_xlabel(lbl,fontsize=7,labelpad=1.5)
fig.subplots_adjust(wspace=0.06,hspace=0.28)
fig.savefig(f"{OUT}/rfig5_objects.png",bbox_inches='tight',dpi=220); plt.close(fig)
print("all figures written")
