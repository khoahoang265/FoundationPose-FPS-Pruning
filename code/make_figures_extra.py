import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, trimesh
import os
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":8.5,"figure.dpi":220})

# ============ FIG A: so do pipeline ============
fig,ax=plt.subplots(figsize=(7.0,2.15)); ax.set_xlim(0,116); ax.set_ylim(0,32); ax.axis('off')
def box(x,y,w,h,txt,fc,ec='0.3',fs=7.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc=fc,ec=ec,lw=1.3,zorder=2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,zorder=3,linespacing=1.35,
            fontweight='bold' if bold else 'normal')
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,
                 lw=1.2,color='0.35',zorder=1))

box( 1,12,20,9.5,"Object model\n(CAD or NOF)","#eef3f8")
box(25,12,22,9.5,"Template pool\n$|P|=504$","#eef3f8")
box(51,10,24,13.5,"Hypothesis\nselector\nkeep top-$K$","#ffe9e9",ec='#d62728',fs=8,bold=True)
box(79,12,16,9.5,"Refiner\n(5 iter.)","#eaf4ea")
box(99,12,16,9.5,"Scorer\n$\\rightarrow$ pose","#eaf4ea")
for a,b in [(21,25),(47,51),(75,79),(95,99)]: arrow(a,16.75,b,16.75)

box(51,1.5,24,6,"RGB-D frame + mask","#f7f7f7",fs=7.4)
arrow(63,7.5,63,10)

ax.text(63,29.5,"random / FPS / DINOv2 BoW",ha='center',fontsize=7.4,color='#d62728',style='italic')
ax.text(63,26.6,"(only this block differs)",ha='center',fontsize=7.0,color='#d62728',style='italic')
ax.annotate("",xy=(63,23.8),xytext=(63,25.7),arrowprops=dict(arrowstyle='-|>',color='#d62728',lw=1.1))
ax.plot([79,115],[24.2,24.2],color='0.55',lw=0.8,ls=':')
ax.text(97,25.4,"unchanged FoundationPose",ha='center',fontsize=7.0,color='0.4',style='italic')
fig.savefig(f"{OUT}/rfigA_pipeline.png",bbox_inches='tight'); plt.close(fig)

# ============ FIG B: ban kinh phu tren SO(3) ============
Ns,Ni=42,12
sph=trimesh.creation.icosphere(subdivisions=2)
cam=sph.vertices[::max(1,len(sph.vertices)//Ns)][:Ns]
P=[]
for c in cam:
    z=-c/(np.linalg.norm(c)+1e-8); up=np.array([0,1,0.])
    if abs(z@up)>0.99: up=np.array([1,0,0.])
    x=np.cross(up,z); x/=np.linalg.norm(x)+1e-8; y=np.cross(z,x)
    Rb=np.stack([x,y,z],1)
    for j in range(Ni):
        a=j*2*np.pi/Ni; ca,sa=np.cos(a),np.sin(a)
        P.append(Rb.T@np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.]]).T)
R=np.array(P); M=R.reshape(len(R),9)
Dm=np.degrees(np.arccos(np.clip((M@M.T-1)/2,-1,1)))
def fps(k,seed=0):
    rng=np.random.RandomState(seed); s=[int(rng.randint(len(R)))]; mind=Dm[s[0]].copy()
    for _ in range(k-1):
        n=int(np.argmax(mind)); s.append(n); mind=np.minimum(mind,Dm[n])
    return s
cov=lambda idx: Dm[:,idx].min(1).max()
Ks=[1,3,5,10,20,50]
cf=[cov(fps(k)) for k in Ks]
cr=[np.mean([cov(np.random.RandomState(s).choice(len(R),k,replace=False)) for s in range(200)]) for k in Ks]
print("FPS   :",[round(v,1) for v in cf]); print("Random:",[round(v,1) for v in cr])

fig,ax=plt.subplots(figsize=(3.4,2.55))
ax.plot(Ks,cf,'-o',ms=4.5,lw=1.5,color='#1f77b4',label='Farthest (proposed)')
ax.plot(Ks,cr,'-s',ms=4,lw=1.2,color='#2ca02c',label='Random (mean of 200)')
ax.axvline(5,color='0.5',ls='--',lw=1.0)
ax.set_xscale('log'); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.set_xlabel('Retained hypotheses $K$')
ax.set_ylabel('Covering radius on $SO(3)$ (deg)')
ax.grid(alpha=0.3,ls=':'); ax.set_ylim(35,190)
ax.legend(loc='lower left',fontsize=6.9,borderpad=0.3,handlelength=1.6,framealpha=0.95)
fig.tight_layout(); fig.savefig(f"{OUT}/rfigB_coverage.png",bbox_inches='tight'); plt.close(fig)
print("done")
