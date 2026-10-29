"""Compose accepted strict R3 then original-view loose E; no learned policy."""
import statistics
import numpy as np
SEEDS=[42,123,2026]
STAGES=['strict','E']
CASES=['base','strict','E']
RULES=dict(proposal=.4,strict_score=.95,strict_margin=.1,loose_score=.90,loose_margin=.05,novel=.1,proposal_weight=.2,center=.2,postfilter_percent=5)

def compose(base,masks,q,image):
    from tools.oracle_guided_r import strict_background_rescue
    from tools.r3_crossview_control_core import compose as loose
    from tools.r3_crossview_core import strict_owners
    a,report=strict_background_rescue(base,masks,q)
    owner=strict_owners(base,masks,q['candidate_ids'],report)
    row=dict(A=a,masks=masks,candidate_ids=q['candidate_ids'],labels=q['class_ids'],scores=q['retrieval_scores'],margins=q['margins'],proposals=q['proposal_scores'],accepted_A={x['candidate_id'] for x in report['accepted_candidates']})
    e=loose({image:row},0,0)
    pred=e['predictions'][image];oe=e['owners'][image]
    assert np.array_equal(pred[base!=0],base[base!=0]) and np.array_equal(pred[a!=0],a[a!=0])
    return dict(strict=a,E=pred,owner_strict=owner,owner_E=oe),dict(strict=report,loose=e['traces'][image],loose_accepted=e['accepted'])

def summarize(metrics):
    out={}
    for case in CASES:
        values=[100*metrics[f'{case}_{s}']['mIoU'] for s in SEEDS]
        out[case]=dict(seeds=values,mean=statistics.mean(values),sampleSD=statistics.stdev(values))
    pairs={}
    for a,b in [('strict','base'),('E','base'),('E','strict')]:
        d=[x-y for x,y in zip(out[a]['seeds'],out[b]['seeds'])]
        pairs[a+'-'+b]=dict(seeds_pp=d,mean_pp=statistics.mean(d),sampleSD_pp=statistics.stdev(d),positive_seeds=sum(x>0 for x in d))
    return dict(methods=out,paired=pairs)
