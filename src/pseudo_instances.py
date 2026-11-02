"""Preserve genuine SAM instances and confidence before semantic rasterization."""
import numpy as np


def compose_instances(masks,class_ids,scores,edge_options=None,class_names=None,policy='score'):
    masks=np.asarray(masks,bool);class_ids=np.asarray(class_ids,int);scores=np.asarray(scores,np.float32)
    if masks.ndim!=3 or class_ids.shape!=(len(masks),) or scores.shape!=(len(masks),):
        raise ValueError('Instance arrays must be aligned')
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any() or ((class_ids<1)|(class_ids>254)).any():
        raise ValueError('Invalid instance classes or confidence')
    repaired=masks.copy()
    if edge_options:
        from src.mask_edge_repair import edge_postprocess_mask
        for i,cid in enumerate(class_ids):
            name=class_names[cid-1] if class_names else ''
            iterations=edge_options['diningtable_close_iterations'] if name in ('table','diningtable','dining table') else edge_options['close_iterations']
            repaired[i]=edge_postprocess_mask(masks[i],edge_options['close_kernel'],iterations,edge_options['dilate_pixels'])
    if policy=='score': order=np.argsort(-scores,kind='stable')
    elif policy=='class_order': order=np.argsort(class_ids,kind='stable')
    else: raise ValueError('Invalid overlap policy')
    semantic=np.zeros(masks.shape[1:],np.uint8);assigned=np.zeros_like(masks)
    for i in order:
        assigned[i]=repaired[i]&(semantic==0)
        semantic[assigned[i]]=class_ids[i]
    return semantic,assigned,repaired


def instance_regions(path,semantic,min_area=300):
    """Keep instance identities; reject semantic edits instead of guessing scores."""
    with np.load(path,allow_pickle=False) as data:
        masks=data['assigned_masks'];classes=data['class_ids'];scores=data['scores']
    if masks.shape[1:] != semantic.shape or masks.shape[0]!=len(classes) or scores.shape!=(len(classes),):
        raise ValueError('Invalid saved instance cache')
    reconstructed=np.zeros_like(semantic)
    result=[]
    for i,(mask,cid,score) in enumerate(zip(masks,classes,scores)):
        mask=np.asarray(mask,bool)
        if (reconstructed[mask]!=0).any(): raise ValueError('Assigned instances overlap')
        reconstructed[mask]=cid
        if mask.sum()>=min_area:
            result.append(dict(mask=mask,cls_id=int(cid),instance_id=i,confidence=float(score)))
    if not np.array_equal(reconstructed,semantic):
        raise ValueError('Semantic labels changed after instance export; explicit reconciliation is required')
    return result
