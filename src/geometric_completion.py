"""Conservative geometry-only completion of retained clean foreground masks.

No feature/model/GT inputs. Existing semantic foreground is never overwritten.
Enclosure is the fraction of inner component boundary pixels within one pixel
(8-neighbourhood) of the ORIGINAL source mask, not of previous completions.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import cv2
import numpy as np
from configs.experiment import GEOMETRIC_COMPLETION


@dataclass(frozen=True)
class GeometricCompletionConfig:
    enabled: bool = GEOMETRIC_COMPLETION['enabled']
    mode: str = GEOMETRIC_COMPLETION['mode']
    kernel_ratio: float = GEOMETRIC_COMPLETION['kernel_ratio']
    min_kernel: int = GEOMETRIC_COMPLETION['min_kernel']
    max_kernel: int = GEOMETRIC_COMPLETION['max_kernel']
    min_mask_density: float = GEOMETRIC_COMPLETION['min_mask_density']
    min_component_area: int = GEOMETRIC_COMPLETION['min_component_area']
    max_component_ratio: float = GEOMETRIC_COMPLETION['max_component_ratio']
    max_total_add_ratio: float = GEOMETRIC_COMPLETION['max_total_add_ratio']
    min_enclosure: float = GEOMETRIC_COMPLETION['min_enclosure']
    bbox_only: bool = GEOMETRIC_COMPLETION['bbox_only']
    background_only: bool = GEOMETRIC_COMPLETION['background_only']
    debug_visualize: bool = GEOMETRIC_COMPLETION['debug_visualize']
    debug_image_limit: int = GEOMETRIC_COMPLETION['debug_image_limit']
    local_min_thickness: float = GEOMETRIC_COMPLETION['local_min_thickness']
    local_max_gap_ratio: float = GEOMETRIC_COMPLETION['local_max_gap_ratio']
    local_min_alignment: float = GEOMETRIC_COMPLETION['local_min_alignment']
    support_search_pixels: int = GEOMETRIC_COMPLETION['support_search_pixels']
    support_depth_pixels: int = GEOMETRIC_COMPLETION['support_depth_pixels']
    support_alignment_padding: int = GEOMETRIC_COMPLETION['support_alignment_padding']
    support_thickness_quantile: float = GEOMETRIC_COMPLETION['support_thickness_quantile']
    support_gap_quantile: float = GEOMETRIC_COMPLETION['support_gap_quantile']

    def __post_init__(self):
        if self.mode not in ('legacy','local_support'): raise ValueError('Unknown completion mode')
        for name in ('enabled', 'bbox_only', 'background_only', 'debug_visualize'):
            if type(getattr(self, name)) is not bool: raise ValueError(name + ' must be boolean')
        for name in ('min_kernel', 'max_kernel', 'min_component_area', 'debug_image_limit',
                     'support_search_pixels','support_depth_pixels','support_alignment_padding'):
            if type(getattr(self, name)) is not int: raise ValueError(name + ' must be integer')
        if self.min_kernel<1 or self.max_kernel<self.min_kernel or self.min_kernel%2!=1 or self.max_kernel%2!=1:
            raise ValueError('Kernel bounds must be positive odd integers in order')
        for name in ('kernel_ratio','min_mask_density','max_component_ratio','max_total_add_ratio','min_enclosure',
                     'local_min_alignment','support_thickness_quantile','support_gap_quantile'):
            value=getattr(self,name)
            if not math.isfinite(value) or not 0<=value<=1: raise ValueError(name+' must be finite in [0,1]')
        if self.min_component_area<1 or self.debug_image_limit<0: raise ValueError('Invalid area/debug limit')
        if min(self.support_search_pixels,self.support_depth_pixels)<1 or self.support_alignment_padding<0:
            raise ValueError('Invalid local support window')
        for name in ('local_min_thickness','local_max_gap_ratio'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError('Invalid '+name)
        if not self.bbox_only or not self.background_only:
            raise ValueError('This conservative version requires bbox_only and background_only')


def adaptive_kernel(extent, config):
    size=max(config.min_kernel,min(config.max_kernel,int(round(extent*config.kernel_ratio))))
    return min(config.max_kernel,size if size%2 else size+1)


def directional_envelopes(mask, horizontal, vertical):
    pad=max(horizontal,vertical)//2+1
    padded=np.pad(np.asarray(mask,np.uint8),pad,constant_values=0)
    def close(kernel):
        out=cv2.morphologyEx(padded,cv2.MORPH_CLOSE,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        return out[pad:-pad,pad:-pad].astype(bool)
    h=close(np.ones((1,horizontal),np.uint8));v=close(np.ones((vertical,1),np.uint8))
    original=np.asarray(mask,bool)
    return original|h|v,h&~original,v&~original


def completion_envelope(mask, horizontal, vertical):
    return directional_envelopes(mask,horizontal,vertical)[0]


def euclidean_support_distance(mask):
    """Exact L2 distances to original background, including outside-bbox zeros."""
    padded=np.pad(np.asarray(mask,np.uint8),1,constant_values=0)
    return cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]


def local_structural_support(original, component, horizontal_new, vertical_new, distance, config):
    """Measure support on scanlines, using original mask pixels exclusively.

    Dominant closing provenance selects ONE axis (horizontal on ties). Each
    support starts at the nearest original foreground within search distance;
    thickness is twice the maximum EDT along its contiguous outward run,
    capped by support_depth_pixels. A nearby disconnected mass is not sampled.
    Side thicknesses use the fixed lower quantile of paired component scanlines;
    gap uses the fixed upper quantile. Alignment is IoU of side projections on
    the orthogonal axis in the padded component window, not image-space IoU.
    """
    h_count=int((component&horizontal_new).sum());v_count=int((component&vertical_new).sum())
    direction='horizontal' if h_count>=v_count else 'vertical'
    result=dict(direction=direction,horizontal_provenance_pixels=h_count,vertical_provenance_pixels=v_count,
        gap_length=None,left_thickness=None,right_thickness=None,support_thickness=None,
        gap_to_thickness_ratio=None,alignment=0.,both_supports=False,paired_scanlines=0,
        negative_side='left' if direction=='horizontal' else 'top',
        positive_side='right' if direction=='horizontal' else 'bottom')
    mask=original if direction=='horizontal' else original.T
    region=component if direction=='horizontal' else component.T
    dt=distance if direction=='horizontal' else distance.T
    yy,xx=np.nonzero(region)
    if not len(yy): return result,'missing_support'
    y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
    negatives=set();positives=set();left=[];right=[];gaps=[]
    for y in range(max(0,y0-config.support_alignment_padding),min(mask.shape[0],y1+config.support_alignment_padding)):
        positions=np.flatnonzero(region[y]);lo=int(positions[0]) if len(positions) else x0
        hi=int(positions[-1])+1 if len(positions) else x1
        a=np.flatnonzero(mask[y,max(0,lo-config.support_search_pixels):lo])
        b=np.flatnonzero(mask[y,hi:min(mask.shape[1],hi+config.support_search_pixels)])
        lx=max(0,lo-config.support_search_pixels)+int(a[-1]) if len(a) else None
        rx=hi+int(b[0]) if len(b) else None
        if lx is not None: negatives.add(y)
        if rx is not None: positives.add(y)
        if not len(positions) or lx is None or rx is None: continue
        values=[]
        for start,step in ((lx,-1),(rx,1)):
            radii=[]
            for offset in range(config.support_depth_pixels):
                x=start+step*offset
                if not 0<=x<mask.shape[1] or not mask[y,x]: break
                radii.append(float(dt[y,x]))
            values.append(2*max(radii) if radii else 0.)
        left.append(values[0]);right.append(values[1]);gaps.append(rx-lx-1)
    result['support_projection_iou']=len(negatives&positives)/max(len(negatives|positives),1)
    result['paired_support_coverage']=len(gaps)/max(y1-y0,1)
    result['alignment']=min(result['support_projection_iou'],result['paired_support_coverage'])
    result['paired_scanlines']=len(gaps)
    if not gaps: return result,'missing_support'
    l=float(np.quantile(left,config.support_thickness_quantile));r=float(np.quantile(right,config.support_thickness_quantile))
    gap=float(np.quantile(gaps,config.support_gap_quantile));thickness=min(l,r)
    result.update(both_supports=True,gap_length=gap,left_thickness=l,right_thickness=r,
        support_thickness=thickness,gap_to_thickness_ratio=gap/thickness if thickness>0 else None)
    reason=('thin_support' if thickness<config.local_min_thickness else
        'gap_to_thickness' if gap/thickness>config.local_max_gap_ratio else
        'support_alignment' if result['alignment']<config.local_min_alignment else None)
    return result,reason


def _debug_save(directory, before, after, index=None, bounds=None, arrays=None):
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if index is None:
        Image.fromarray(before).save(directory/'before_semantic.png')
        Image.fromarray(after).save(directory/'after_semantic.png')
        return
    y0,y1,x0,x1=bounds
    for name,value in arrays.items():
        canvas=np.zeros(before.shape,np.uint8);canvas[y0:y1,x0:x1]=np.asarray(value,np.uint8)*255
        Image.fromarray(canvas).save(directory/f'{index:03d}_{name}.png')


def complete_semantic_map(semantic_map, clean_instances, config, *, image_id=None,
                          class_to_id=None, debug_dir=None):
    """Return a new map and per-instance diagnostics; callers explicitly pass clean sources.

    Entire components are accepted or skipped, never clipped to fit the budget.
    Ties use stable source/component order. Original masks are never mutated.
    """
    before=np.asarray(semantic_map)
    if before.ndim!=2 or not np.issubdtype(before.dtype,np.integer): raise ValueError('Expected integer HxW map')
    out=before.copy()
    report=dict(image_id=image_id,enabled=config.enabled,parameters=asdict(config),instances=[],added_pixels=0)
    if not config.enabled: return out,report
    three=np.ones((3,3),np.uint8)
    indexed=list(enumerate(clean_instances))
    for _,row in indexed:
        if not math.isfinite(float(row['score'])): raise ValueError('Nonfinite source score')
    for source_index,row in sorted(indexed,key=lambda pair:-float(pair[1]['score'])):
        if row.get('geometric_completion_source') is False or row.get('is_rescue',False): continue
        cid=int(row.get('class_id',(class_to_id or {}).get(row.get('cls_name'),0)))
        if cid<=0 or row.get('cls_name')=='background': continue
        if cid>np.iinfo(before.dtype).max: raise ValueError('Class ID exceeds semantic dtype')
        mask=np.asarray(row['mask'])
        if mask.shape!=before.shape or not np.all((mask==0)|(mask==1)): raise ValueError('Invalid source mask')
        mask=mask.astype(bool);area=int(mask.sum())
        info=dict(source_index=source_index,class_id=cid,score=float(row['score']),original_mask_area=area,
                  candidate_area=0,accepted_area=0,mask_density=0.,components=[],added_ratio=0.,skip_reason=None)
        report['instances'].append(info)
        if area==0: info['skip_reason']='empty';continue
        ys,xs=np.nonzero(mask);y0,y1=int(ys.min()),int(ys.max())+1;x0,x1=int(xs.min()),int(xs.max())+1
        density=area/((y1-y0)*(x1-x0));info.update(mask_density=density,bbox_xyxy=[x0,y0,x1,y1])
        if density<config.min_mask_density: info['skip_reason']='low_density';continue
        original=mask[y0:y1,x0:x1];view=out[y0:y1,x0:x1]
        horizontal=adaptive_kernel(x1-x0,config);vertical=adaptive_kernel(y1-y0,config)
        envelope,horizontal_new,vertical_new=directional_envelopes(original,horizontal,vertical)
        distance=euclidean_support_distance(original) if config.mode=='local_support' else None
        candidate=envelope & ~original & (view==0)
        info.update(candidate_area=int(candidate.sum()),horizontal_kernel=horizontal,vertical_kernel=vertical)
        count,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),connectivity=8)
        adjacent=cv2.dilate(original.astype(np.uint8),three,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        eligible=[]
        for label in range(1,count):
            component=labels==label;size=int(stats[label,cv2.CC_STAT_AREA])
            inner=component & ~cv2.erode(component.astype(np.uint8),three,
                borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            enclosure=float(np.count_nonzero(inner&adjacent)/max(int(inner.sum()),1))
            reason=('small' if size<config.min_component_area else
                    'component_budget' if size>area*config.max_component_ratio else
                    'enclosure' if enclosure<config.min_enclosure else None)
            record=dict(component_id=label,area=size,enclosure=enclosure,accepted=False,reject_reason=reason)
            if config.mode=='local_support':
                support,support_reason=local_structural_support(original,component,horizontal_new,vertical_new,distance,config)
                record.update(support)
                record.update(support_gate_reason=support_reason,support_gate_passed=support_reason is None)
                if reason is None: record['reject_reason']=reason=support_reason
            else:
                record.update(direction=None,gap_length=None,left_thickness=None,right_thickness=None,
                    support_thickness=None,gap_to_thickness_ratio=None,alignment=None)
            info['components'].append(record)
            if reason is None: eligible.append((enclosure,label,record))
        accepted=np.zeros_like(original)
        for enclosure,label,record in sorted(eligible,key=lambda item:(-item[0],item[1])):
            if info['accepted_area']+record['area']>area*config.max_total_add_ratio:
                record['reject_reason']='total_budget';continue
            component=(labels==label) & (view==0)
            assert int(component.sum())==record['area']
            view[component]=cid;accepted|=component
            record['accepted']=True;info['accepted_area']+=record['area']
        info['added_ratio']=info['accepted_area']/area
        report['added_pixels']+=info['accepted_area']
        if config.debug_visualize and debug_dir is not None:
            _debug_save(debug_dir,before,out,source_index,(y0,y1,x0,x1),
                dict(original_mask=original,completion_envelope=envelope,candidate=candidate,accepted_completion=accepted))
    if not np.array_equal(out[before!=0],before[before!=0]): raise AssertionError('Completion overwrote foreground')
    if config.debug_visualize and debug_dir is not None:
        _debug_save(debug_dir,before,out)
        (Path(debug_dir)/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return out,report


def summarize_completion(reports):
    classes={}
    for report in reports:
        for row in report['instances']:
            r=classes.setdefault(str(row['class_id']),dict(source_instances=0,triggered_instances=0,
                added_pixels=0,sum_added_ratio=0.,original_area=0))
            r['source_instances']+=1;r['triggered_instances']+=int(row['accepted_area']>0)
            r['added_pixels']+=row['accepted_area'];r['sum_added_ratio']+=row['added_ratio'];r['original_area']+=row['original_mask_area']
    for row in classes.values():
        row['mean_added_ratio_triggered']=row['sum_added_ratio']/max(row['triggered_instances'],1)
        row['mean_added_ratio_all_sources']=row['sum_added_ratio']/max(row['source_instances'],1)
        row['area_weighted_added_ratio']=row['added_pixels']/max(row['original_area'],1)
    return dict(images=len(reports),added_pixels=sum(r['added_pixels'] for r in reports),per_class=classes)
