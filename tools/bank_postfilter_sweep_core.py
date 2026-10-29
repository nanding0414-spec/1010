"""Fixed development sweep of projected-gallery trimming; no training or GT selection."""
import statistics
import numpy as np
PERCENTAGES=[0,5,10,15,20,25,30]
NEW_PERCENTAGES=[0,5,15,20,25,30]
SEEDS=[42,123,2026]
CASES=['clean','HCR']
def name(p):
    if type(p) is not int or p not in PERCENTAGES:raise ValueError('Unregistered post-filter percentage')
    return f'p{p:02d}'
def trim(x,ids,percent,background=False):
    name(percent);x=np.asarray(x);ids=np.asarray(ids,str)
    if x.ndim!=2 or len(ids)!=len(x) or len(set(ids))!=len(ids):raise ValueError('Invalid/duplicate members')
    if x.dtype!=np.float32 or not np.isfinite(x).all() or (len(x) and (np.linalg.norm(x,axis=1)<=1e-6).any()):raise ValueError('Invalid projected vectors')
    center=x.mean(0) if len(x) else np.zeros(x.shape[1],np.float32)
    distances=np.sqrt(((x-center[None])**2).sum(1));count=0 if background else int(np.floor(len(x)*(percent/100.)))
    order=np.argsort(distances)[::-1];drop=set(order[:count].tolist());keep=np.asarray([i for i in range(len(x)) if i not in drop],np.int64)
    return keep,dict(N=len(x),kept=len(keep),dropped=count,percent=percent,background=bool(background),source_ids=ids.tolist(),kept_ids=ids[keep].tolist(),dropped_ids=ids[order[:count]].tolist(),saved_distances=distances.astype(float).tolist(),tie_policy='original numpy argsort descending; original retained order')
def summarize(metrics):
    result={}
    for case in CASES:
        baseline=[100*metrics[f'p10_{s}_{case}']['mIoU'] for s in SEEDS];result[case]={}
        for p in PERCENTAGES:
            a=name(p);values=[100*metrics[f'{a}_{s}_{case}']['mIoU'] for s in SEEDS];delta=[x-y for x,y in zip(values,baseline)]
            result[case][a]=dict(percent=p,seeds=values,mean=statistics.mean(values),sampleSD=statistics.stdev(values),delta10_pp=delta,mean_delta10_pp=statistics.mean(delta),paired_sampleSD_pp=statistics.stdev(delta),positive_seeds=sum(x>0 for x in delta))
    return result
def winner(summary):
    # Exact mean comparison; smallest deletion percentage wins an exact tie.
    return min(PERCENTAGES,key=lambda p:(-summary['HCR'][name(p)]['mean'],p))
