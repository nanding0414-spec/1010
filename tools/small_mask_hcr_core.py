"""Fixed small-mask controls. No model, GT, IO, or threshold selection."""
import statistics
import numpy as np

SEEDS=[42,123,2026]
ARMS=['A_current','B_gap_hcr','C_area100']
NEW_ARMS=ARMS[1:]
STAGES=['HCR','E']
RULES=dict(baseline_area=300,small_min_area=100,small_max_area_exclusive=300,
           direct_area=100,proposal=.4,retrieval=.95,margin=.1,novel=.1,proposal_weight=.2)

def _nms(rows,minimum,iou=.4):
    groups={}
    for r in rows:groups.setdefault(r['cls_name'],[]).append(r)
    kept=[]
    for group in groups.values():
        selected=[]
        for r in sorted(group,key=lambda z:z['score'],reverse=True):
            if int(r['mask'].sum())<minimum:continue
            if any(float(np.logical_and(r['mask'],p['mask']).sum())/(float(np.logical_or(r['mask'],p['mask']).sum())+1e-8)>iou for p in selected):continue
            selected.append(r)
        kept.extend(selected)
    return kept

def _raster(rows,shape,mapping):
    sem=np.zeros(shape,np.uint8);owner=np.full(shape,-1,np.int32)
    for r in sorted(rows,key=lambda x:x['score'],reverse=True):
        cls=mapping[r['cls_name']]
        if cls==0:continue
        add=r['mask']&(sem==0);sem[add]=cls;owner[add]=r['pool_index']
    return sem,owner

def assemble(masks,proposals,flags,labels,retrieval,margins,names,arm):
    from src.semantic_rescue import HCRConfig,rescue_instances
    assert arm in ARMS and len(names)==21 and names[0]=='background'
    masks=np.asarray(masks,bool);n=len(masks)
    proposals,retrieval,margins=[np.asarray(a,np.float64) for a in (proposals,retrieval,margins)]
    flags=np.asarray(flags,bool);labels=np.asarray(labels,np.int16)
    assert masks.ndim==3 and all(x.shape==(n,) for x in [proposals,retrieval,margins,flags,labels])
    assert np.isin(labels,np.arange(21)).all() and all(np.isfinite(a).all() for a in [proposals,retrieval,margins])
    assert ((proposals>=0)&(proposals<=1)).all() and (margins>=0).all()
    rows=[dict(mask=masks[i],cls_name=names[int(labels[i])],score=.8*float(retrieval[i])+.2*float(proposals[i]),
               proposal_score=float(proposals[i]),retrieval_score=float(retrieval[i]),retrieval_margin=float(margins[i]),pool_index=i) for i in range(n)]
    original=[r for r in rows if not flags[r['pool_index']]];extras=[r for r in rows if flags[r['pool_index']]]
    minimum=100 if arm=='C_area100' else 300
    strong=_nms(original,minimum)
    mapping={name:i for i,name in enumerate(names)}
    clean,owner_clean=_raster(strong,masks.shape[1:],mapping)
    stats={};hcr=rescue_instances(strong,extras,HCRConfig(),names[1:],stats)
    old,old_owner=_raster(hcr,masks.shape[1:],mapping)
    trace=[];repair_stats={};base=old
    if arm=='B_gap_hcr':
        small=[r for r in original if 100<=int(r['mask'].sum())<300]
        extended=rescue_instances(hcr,small,HCRConfig(),names[1:],repair_stats)
        added={r['pool_index']:r['mask'] for r in extended[len(hcr):]}
        occupied=old!=0
        for r in sorted(small,key=lambda z:z['score'],reverse=True):
            i=r['pool_index'];area=int(r['mask'].sum());avail=r['mask']&~occupied;count=int(avail.sum())
            reason=('background' if labels[i]==0 else 'proposal' if proposals[i]<.4 else 'retrieval' if retrieval[i]<.95 else 'margin' if margins[i]<.1 else 'novel' if count==0 or count/area<.1 else 'accepted')
            accepted=i in added
            assert accepted==(reason=='accepted')
            if accepted:assert np.array_equal(added[i],avail);occupied|=avail
            trace.append(dict(index=i,mask_area=area,available_pixels=count,novel_fraction=count/area,
                              accepted=accepted,reason=reason,added_pixels=count if accepted else 0))
        hcr=extended
    result,owner=_raster(hcr,masks.shape[1:],mapping)
    if arm=='B_gap_hcr':assert np.array_equal(result[old!=0],old[old!=0])
    fields=dict(clean=clean,HCR=result,owner_clean=owner_clean,owner_HCR=owner,
                before_repair=base,owner_before_repair=old_owner)
    report=dict(arm=arm,classified_min_area=minimum,clean_kept=[r['pool_index'] for r in strong],
                HCR_stats=stats,repair_stats=repair_stats,repair_trace=trace,
                total_small_clean=sum((not flags[i]) and 100<=int(masks[i].sum())<300 for i in range(n)),
                HCR_added_pixels=int(np.count_nonzero((result!=0)&(clean==0))))
    return fields,report

def summarize(metrics):
    out={}
    for stage in STAGES:
        out[stage]={};baseline=[100*metrics[f'A_current_{s}_{stage}']['mIoU'] for s in SEEDS]
        for a in ARMS:
            values=[100*metrics[f'{a}_{s}_{stage}']['mIoU'] for s in SEEDS];delta=[x-y for x,y in zip(values,baseline)]
            out[stage][a]=dict(seeds=values,mean=statistics.mean(values),sampleSD=statistics.stdev(values),delta_A_pp=delta,mean_delta_A_pp=statistics.mean(delta),positive_seeds=sum(x>0 for x in delta))
    return out
