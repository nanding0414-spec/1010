"""Fixed front/back distance trimming. Independent research; no production edits."""
from fractions import Fraction
import hashlib,json
import numpy as np

ARMS=['A_pre10','B_post10','C_pre10_post10','D_pre5_post5']
SEEDS=[42,123,2026]
RATIOS={'A_pre10':('0.10','0'),'B_post10':('0','0.10'),
        'C_pre10_post10':('0.10','0.10'),'D_pre5_post5':('0.05','0.05')}

def trim(x,ids,ratio,background=False):
    """Mirror production mean/FP32 L2/np.argsort[::-1], preserve kept order."""
    x=np.asarray(x);ids=np.asarray(ids,str)
    if x.ndim!=2 or len(x)!=len(ids) or len(set(ids))!=len(ids):raise ValueError('Invalid class population')
    if x.dtype!=np.float32 or not np.isfinite(x).all() or (len(x) and (np.linalg.norm(x,axis=1)<=1e-6).any()):raise ValueError('Invalid source vectors')
    p=Fraction(str(ratio))
    if p not in [Fraction(0),Fraction(1,20),Fraction(1,10)]:raise ValueError('Fixed ratios only')
    # Keep historical floating-floor semantics, including rounding at products.
    count=0 if background else int(np.floor(len(x)*float(p)))
    center=x.mean(0) if len(x) else np.zeros(x.shape[1],np.float32)
    distance=np.sqrt(((x-center[None])**2).sum(1))
    order=np.argsort(distance)[::-1]
    dropped=set(order[:count].tolist());keep=np.asarray([i for i in range(len(x)) if i not in dropped],np.int64)
    report=dict(N=len(x),kept=len(keep),dropped=count,ratio=str(ratio),background=background,
                kept_ids=ids[keep].tolist(),dropped_ids=ids[order[:count]].tolist(),
                distance_dtype=str(distance.dtype),tie_policy='production numpy argsort descending; source order pinned')
    return keep,report

def paired(metrics):
    """Each input scalar is mIoU fraction; output uses percentage points."""
    import statistics
    out={}
    for arm in ARMS:
        values=[float(metrics[arm][str(s)])*100 for s in SEEDS]
        delta=[values[i]-float(metrics[ARMS[0]][str(s)])*100 for i,s in enumerate(SEEDS)]
        out[arm]=dict(seeds=values,mean=statistics.mean(values),sampleSD=statistics.stdev(values),
                      delta_A_pp=delta,mean_delta_A_pp=statistics.mean(delta),positive_seeds=sum(v>0 for v in delta))
    return out
