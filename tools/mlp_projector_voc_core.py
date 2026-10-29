"""Saved-count evaluation for the explicitly authorized six-arm VOC follow-up."""
import statistics
import numpy as np

ARMS=['E0_prod_aligned','E1_linear_l2','E2_rmsnorm','E3_residual_bottleneck','E4_dual_head','E5_residual_bankmargin']
SEEDS=[42,123,2026]
CASES=['clean','HCR']

def confusion(gt,pred):
    gt=np.asarray(gt);pred=np.asarray(pred)
    if gt.shape!=pred.shape or not np.isin(pred,np.arange(21)).all():
        raise ValueError('Prediction shape/classes differ')
    valid=(gt>=0)&(gt<21)
    return np.bincount(gt[valid].astype(np.int64)*21+pred[valid].astype(np.int64),minlength=441).reshape(21,21)

def calculate(cm,mapping):
    cm=np.asarray(cm)
    if cm.shape!=(21,21) or cm.dtype.kind not in 'iu' or (cm<0).any():raise ValueError('Invalid saved counts')
    union=cm.sum(0)+cm.sum(1)-cm.diagonal()
    iou=np.divide(cm.diagonal(),union,out=np.zeros(21,float),where=union>0)
    inverse={v:int(k) for k,v in mapping.items()}
    return dict(mIoU=float(iou[union>0].mean()) if (union>0).any() else None,
                per_class_IoU={mapping[str(i)]:float(iou[i]) for i in range(21)},confusion_matrix=cm.tolist(),
                BG_FP=int(cm[0,1:].sum()),FN=int(cm[1:,0].sum()),
                wrong_FG=int(cm[1:,1:].sum()-np.trace(cm[1:,1:])),
                chair_to_sofa=int(cm[inverse['chair'],inverse['sofa']]),sofa_to_chair=int(cm[inverse['sofa'],inverse['chair']]))

def summarize(metrics):
    result={}
    for case in CASES:
        baseline=[100*metrics[f'{ARMS[0]}_{s}_{case}']['mIoU'] for s in SEEDS]
        result[case]={}
        for arm in ARMS:
            x=[100*metrics[f'{arm}_{s}_{case}']['mIoU'] for s in SEEDS]
            delta=[a-b for a,b in zip(x,baseline)]
            result[case][arm]=dict(per_seed=x,mean=statistics.mean(x),sample_SD=statistics.stdev(x),
                paired_delta_pp=delta,paired_mean_pp=statistics.mean(delta),paired_sample_SD_pp=statistics.stdev(delta),positive_seeds=sum(v>0 for v in delta))
    return result
