"""Fixed-mask SAM3 semantic assistance; isolated, read-only historical inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_bank_provenance as provenance
from tools.experimental_background_partition_recovery import AliasAwareStages
from tools.experimental_tag_constraint_recovery import evaluation_environment
ab, queue, gmp, old = provenance.ab, provenance.queue, provenance.gmp, provenance.old
PROTOCOL = 'fixed_mask_semantic_rank_v1'
METHODS = ('baseline', 'sam_label', 'rank_bonus', 'guarded')
ACTIVE = dict(preview=False, bucket_size=4, neighbor_tokens=2, hypernym_tokens=2,
              context_strength=1., confidence_threshold=.15, mask_nms_threshold=.7,
              min_mask_area=100, min_mask_region_area=0, max_mask_area_ratio=.95, max_masks=0)
RULE = dict(min_iou=.5, rank_bonus=.05, retrieval_margin=.05,
            semantic_min=.5, semantic_gap=.1, ordering='original retrieval scalar and margin')


def unique_records(records):
    result = {}
    for r in records:
        v = dict(r, path=str(Path(r['path']).resolve()))
        if v['path'] in result and result[v['path']] != v:
            raise ValueError('Conflicting file identities: ' + v['path'])
        result[v['path']] = v
    return [result[p] for p in sorted(result)]


def mask_array(masks):
    x = np.asarray(masks)
    if x.ndim != 3 or not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
        raise ValueError('Expected finite binary N,H,W masks')
    return x.astype(bool, copy=False)


def semantic_evidence(query, candidates, classes, scores, chunk=8):
    """E_rc=max(q_j*IoU), only geometrically corresponding class c candidates."""
    query, candidates = mask_array(query), mask_array(candidates)
    classes, scores = np.asarray(classes), np.asarray(scores, np.float64)
    if chunk < 1 or classes.shape != (len(candidates),) or scores.shape != classes.shape:
        raise ValueError('Invalid semantic rows/chunk')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('SAM scores must be finite in [0,1]')
    if not np.isfinite(classes).all() or not np.equal(classes, classes.astype(int)).all() or ((classes < 1) | (classes > 20)).any():
        raise ValueError('Class IDs must be integers 1..20')
    if len(query) and len(candidates) and query.shape[1:] != candidates.shape[1:]:
        raise ValueError('Mask geometry differs')
    evidence = np.zeros((len(query), 20), np.float32)
    overlap = np.zeros_like(evidence)
    confidence = np.zeros_like(evidence)
    qa = query.sum(axis=(1, 2))
    for candidate, c, score in zip(candidates, classes.astype(int), scores):
        area = int(candidate.sum())
        for start in range(0, len(query), chunk):
            inter = (query[start:start+chunk] & candidate).sum(axis=(1, 2))
            union = qa[start:start+chunk] + area - inter
            iou = np.divide(inter, union, out=np.zeros(len(inter), np.float64), where=union > 0)
            values = np.where(iou >= RULE['min_iou'], score * iou, 0).astype(np.float32)
            view = evidence[start:start+chunk, c-1]
            better = values > view
            view[better] = values[better]
            overlap[start:start+chunk, c-1][better] = iou[better]
            confidence[start:start+chunk, c-1][better] = score
    return evidence, overlap, confidence


def decide(base_scores, semantic, classes, method):
    """Only choose labels: do not recalibrate or change instance ordering scores."""
    scores, semantic = np.asarray(base_scores, np.float32), np.asarray(semantic, np.float32)
    names = list(classes)
    if method not in METHODS or scores.shape != (len(names),) or semantic.shape != (20,):
        raise ValueError('Invalid decision shapes/method')
    if len(names) != 21 or set(names) != set(provenance.evidence.VOC_CLASSES):
        raise ValueError('Unexpected VOC class mapping')
    if not np.isfinite(scores).all() or not np.isfinite(semantic).all() or ((semantic < 0) | (semantic > 1)).any():
        raise ValueError('Nonfinite/out-of-range class evidence')
    base_order = np.argsort(scores)[::-1]
    baseline = names[base_order[0]]
    sem_order = np.argsort(-semantic, kind='stable')
    positive = [int(i) for i in sem_order if semantic[i] > 0]
    top = provenance.evidence.VOC_CLASSES[positive[0]+1] if positive else None
    second = float(semantic[positive[1]]) if len(positive) > 1 else 0.
    info = dict(baseline=baseline, semantic=top, matched=bool(positive),
                semantic_score=float(semantic[positive[0]]) if positive else 0.,
                semantic_gap=float(semantic[positive[0]])-second if positive else 0.,
                retrieval_margin=float(scores[base_order[0]]-scores[base_order[1]]))
    if method == 'baseline': return baseline, info
    if method == 'sam_label': return top or 'background', info  # abstention, not learned background
    if not positive: return baseline, info
    if method == 'guarded':
        eligible = (info['retrieval_margin'] <= RULE['retrieval_margin']
                    and info['semantic_score'] >= RULE['semantic_min']
                    and info['semantic_gap'] >= RULE['semantic_gap'])
        return (top if eligible else baseline), dict(info, gate_passed=eligible)
    adjusted = scores.copy()
    for rank, i in enumerate(positive, 1):
        adjusted[names.index(provenance.evidence.VOC_CLASSES[i+1])] += RULE['rank_bonus']/rank
    return names[np.argsort(adjusted)[::-1][0]], info


def semantic_identity(job, stem):
    return old.canonical(dict(protocol=PROTOCOL, input=job['cache_identity'][stem],
        checkpoint_sha256=job['checkpoint_sha256'], source=job['source'],
        active=ACTIVE, vocabulary=job['active_classes'], rule=RULE, precision=ab.POLICY,
        categories='all 20, no image-level or pixel labels', source_sha=job['semantic_source_sha']))


def read_semantic(job, stem, masks):
    data = queue.cache_read(Path(job['semantic_cache'])/stem, semantic_identity(job, stem))
    if data is None or data['mask_hashes'] != queue.mask_hashes(masks):
        raise queue.GlobalFailure('Missing/misaligned semantic cache: ' + stem)
    with np.load(data['file'], allow_pickle=False) as f:
        values = f['evidence'].copy()
    if values.shape != (len(masks),20) or not np.isfinite(values).all() or ((values < 0)|(values > 1)).any():
        raise queue.GlobalFailure('Invalid semantic cache values')
    return values


def extract_worker(job):
    import torch
    from PIL import Image
    from src.feature_extractor_factory import build_feature_extractor
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    from run_inference import load_strict_proposals
    model = build_feature_extractor(torch.device('cuda:0'), use_interactivity=False)
    if model.feature_signature() != job['source']: raise ValueError('Live feature signature changed')
    vocab = SAM3PseudoMaskGenerator(dataset_type='voc', load_model=False, output_dir=job['semantic_cache'])
    if old.canonical([asdict(c) for c in vocab.active_classes]) != job['active_classes']:
        raise ValueError('Active vocabulary changed')
    proposer = SAM3ActiveProposer(model.model, torch.device('cuda:0'), vocab.active_classes, **ACTIVE)
    for n, stem in enumerate(job['ids'], 1):
        queue.ensure_resources(job['deadline'])
        if ab.digest(job['images'][stem]) != job['cache_identity'][stem]['image_sha256']:
            raise ValueError('Image changed')
        if ab.digest(Path(job['proposals'])/(stem+'.npz')) != job['cache_identity'][stem]['mask_source_sha256']:
            raise ValueError('Query masks changed')
        with Image.open(job['images'][stem]) as im: rgb = np.asarray(im.convert('RGB'))
        masks, _ = load_strict_proposals(job['proposals'], stem, *rgb.shape[:2])
        folder = Path(job['semantic_cache'])/stem
        committed = queue.cache_read(folder, semantic_identity(job, stem))
        if committed:
            read_semantic(job, stem, masks)
            continue
        started = time.monotonic()
        with torch.inference_mode(): output = proposer(np.ascontiguousarray(rgb[...,::-1]), class_indices=None)
        candidates = output['instances'].pred_masks.numpy()
        scores = output['instances'].scores.numpy()
        class_ids = proposer.last_result['class_ids']
        values, ious, confidence = semantic_evidence(masks, candidates, class_ids, scores)
        if job['stage'] == 'semantic_probe':
            repeated = semantic_evidence(masks, candidates, class_ids, scores, chunk=1)
            if not all(np.array_equal(a,b) for a,b in zip(repeated,(values,ious,confidence))):
                raise ValueError('Semantic chunk equivalence failed')
        attempt = queue.next_attempt(folder); file = attempt/'evidence.npz'
        np.savez_compressed(file, evidence=values, iou=ious, confidence=confidence)
        report = dict(image=stem, candidate_count=len(candidates), query_count=len(masks),
            matched_queries=int((values.max(1)>0).sum()) if len(values) else 0,
            seconds=time.monotonic()-started, all_class_ids=proposer.last_result['report']['candidate_class_ids'])
        if report['all_class_ids'] != list(range(1,21)): raise ValueError('Not all VOC classes were queried')
        ab.write(attempt/'report.json', report)
        ab.write(folder/'complete.json', dict(identity=semantic_identity(job,stem), file=str(file),
            mask_hashes=queue.mask_hashes(masks), artifacts=ab.identities([file,attempt/'report.json'])))
        if not np.array_equal(read_semantic(job,stem,masks),values): raise ValueError('Online/cache mismatch')
        print(f'[Semantic] {n}/{len(job["ids"])} {stem} matched={report["matched_queries"]}/{len(masks)}',flush=True)
        del output, candidates; proposer.last_result = None


def install_evaluation(job):
    original = queue.install_cached_eval
    def hook(j):
        original(j)
        import run_inference as inference
        import src.inference as inf
        import src.retrieval as retrieval
        from PIL import Image
        load, build, classify, search, raster = (inference.load_strict_proposals,
            inference.build_feature_extractor, inf.classify_queries, retrieval._search, inference.rasterize_instances)
        state = {}; trace = []; counts = Counter(); seen = []
        def load_hook(directory, stem, h, w):
            masks, scores = load(directory, stem, h, w)
            state.clear(); state.update(stem=stem,masks=masks,rows=[])
            if job['semantic_method'] != 'baseline': state['semantic'] = read_semantic(job,stem,masks)
            return masks, scores
        def build_hook(*args, **kwargs):
            model = build(*args,**kwargs); extract = model.extract_features_batch
            def record(rgb,masks):
                vectors = extract(rgb,masks)
                state['valid'] = [i for i,v in enumerate(vectors) if np.isfinite(v).all() and np.linalg.norm(v)>=1e-6]
                return vectors
            model.extract_features_batch = record
            return model
        def search_hook(*args, **kwargs):
            value = search(*args,**kwargs); state['search'] = value; return value
        def classify_hook(*args, **kwargs):
            result = classify(*args,**kwargs)
            if kwargs.get('mode') != 'balanced': raise ValueError('Balanced retrieval required')
            ss, ii = state.pop('search'); ctx = kwargs['context']
            labels = np.asarray([retrieval.canonical_label(v) for v in kwargs['all_labels']])
            _, _, fused, _ = provenance.audit.decompose(kwargs['query_feats'],ctx,labels,ss,ii,kwargs['class_topm'],kwargs['centroid_weight'])
            if len(result) != len(state['valid']): raise ValueError('Candidate alignment failed')
            output = []
            for k, (r, index) in enumerate(zip(result,state['valid'])):
                order = np.argsort(fused[k])[::-1]
                if r.label != ctx.labels[order[0]] or abs(r.score-float(fused[k,order[0]]))>2e-6 or abs(r.margin-float(fused[k,order[0]]-fused[k,order[1]]))>2e-6:
                    raise ValueError('Retrieval decomposition differs')
                semantic = state['semantic'][index] if 'semantic' in state else np.zeros(20,np.float32)
                label, detail = decide(fused[k],semantic,ctx.labels,job['semantic_method'])
                # Explicit label-only ablation. Original NMS/raster rules and ordering scalars retained.
                output.append(retrieval.RetrievalPrediction(label=label,score=r.score,margin=r.margin))
                state['rows'].append(dict(index=index,prediction=label,**detail))
            return output
        def raster_hook(rows,h,w):
            prediction = raster(rows,h,w)  # GT is accessed ONLY AFTER all decisions and composition.
            stem = state['stem']; seen.append(stem)
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im: gt=np.asarray(im)
            for row in state['rows']:
                detail = provenance.histogram(state['masks'][row['index']],gt)
                row.update(image=stem,**detail); trace.append(row); counts['candidates']+=1
                if row['matched']: counts['matched']+=1
                if detail['purity'] is None or detail['purity'] < .9: continue
                truth = provenance.evidence.VOC_CLASSES[detail['majority']]
                counts['pure']+=1
                b, new = row['baseline']==truth, row['prediction']==truth
                counts['base_correct']+=int(b); counts['new_correct']+=int(new)
                counts['corrected']+=int(not b and new); counts['harmed']+=int(b and not new)
                if row['semantic'] is None: counts['semantic_abstain_pure']+=1
                else:
                    s = row['semantic']==truth
                    counts['matched_pure']+=1; counts['sam_correct']+=int(s)
                    counts['retrieval_wrong_sam_right']+=int(not b and s)
                    counts['retrieval_right_sam_wrong']+=int(b and not s)
            return prediction
        inference.load_strict_proposals=load_hook; inference.build_feature_extractor=build_hook
        inference.rasterize_instances=raster_hook; inf.classify_queries=classify_hook; retrieval._search=search_hook
        job['_trace'],job['_counts'],job['_seen'] = trace,counts,seen
    queue.install_cached_eval = hook


def worker(job):
    os.chdir(ROOT); sys.path.insert(0,str(ROOT/'sam3-main'))
    import torch
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    started=time.monotonic(); status='failed'; category='local'; error=None
    try:
        if job['protocol'] != PROTOCOL or job['rule'] != RULE or job['active'] != ACTIVE:
            raise ValueError('Worker protocol mismatch')
        if job['stage'] in ('semantic_probe','semantic_extract'):
            ab.install_precision()
            from project_utils.seed_utils import set_global_seed
            set_global_seed(42); extract_worker(job)
        elif job['stage']=='eval':
            queue.KEY_IMAGES=[]; install_evaluation(job); old.previous.worker(job)
            if job['_seen'] != job['val_ids'] or len(job['_seen'])!=1449: raise ValueError('Incomplete full evaluation')
            ab.write(Path(job['attempt'])/'diagnostics.json',dict(counts=dict(job['_counts']),rows=job['_trace'],
                note='GT is post-hoc only; pure means >=90%; abstentions are not confident background predictions'))
        else: raise ValueError('Unsupported stage')
        torch.cuda.synchronize(); status='complete'
    except queue.Deferred as exc: category='deferred'; error=str(exc); raise
    except torch.cuda.OutOfMemoryError as exc: category='oom'; error=str(exc); raise
    except (ValueError,KeyError,FileNotFoundError,queue.GlobalFailure) as exc: category='global'; error=str(exc); raise
    except BaseException as exc: error=str(exc); raise
    finally:
        ab.write(Path(job['attempt'])/'queue_worker.json',dict(status=status,category=category,error=error,
            stage=job['stage'],seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),precision_policy=ab.POLICY))


def compact_job(job):
    keys = ('dataset_root image_dir checkpoint train_list val_list imagelevel_json gt_dir images train_ids val_ids '
            'source cache cache_identity checkpoint_sha256 pseudo proposals raw_signature weights bank cleanup '
            'stage group seed role variant').split()
    result = {k:copy.deepcopy(job[k]) for k in keys}
    ids = result['val_ids']
    result['images'] = {i:result['images'][i] for i in ids}
    result['cache_identity'] = {i:result['cache_identity'][i] for i in ids}
    return result


def history(previous):
    print('[History] checking frozen source and deduplicating historical SHA identities',flush=True)
    manifest=ab.read(previous/'manifest.json'); summary=ab.read(previous/'summary.json')
    if summary['status']!='complete' or not summary['groups']['historical']['complete']:
        raise ValueError('Completed clean reference required')
    gmp.verify(manifest['code'],'Previous frozen source')
    records=unique_records(manifest['records']); del manifest
    gmp.verify(records,'Previous inputs')
    prepared_path=provenance.DEPLOY/'prepared.json'; prepared=ab.read(prepared_path)
    refs={}
    for s in old.SEEDS:
        source=prepared['jobs'][f'baseline_{s}']; ref=summary['results'][f'historical_{s}']
        job=compact_job(source['job'])
        metric=ab.read(source['metrics']); replay=ab.read(ref['metrics'])
        if metric['confusion_matrix']!=replay['confusion_matrix']: raise ValueError('Historical baselines disagree')
        old.require_image_replay(source['reference']['per_image'],ref['per_image'])
        if job['role']!='gmp2' or job['group']!='C' or job['variant']!='d2': raise ValueError('Unexpected baseline')
        if len(job['val_ids'])!=1449 or len(set(job['val_ids']))!=1449: raise ValueError('Expected 1449 unique validation IDs')
        refs[s]=dict(job=job,environment=source['command']['environment'],metrics=ref['metrics'],per_image=ref['per_image'])
        records += ab.identities([ref['metrics'],ref['per_image']])
    del prepared
    recorded={r['path'] for r in records}
    used=ab.files_under([Path(refs[s]['job'][k]) for s in old.SEEDS for k in ('weights','bank')])
    used += [Path(first_path) for first_path in (refs[42]['job']['checkpoint'],)]
    if any(str(p.resolve()) not in recorded for p in used):
        raise ValueError('Unrecorded historical model/bank file')
    # The selected completed stages bind these files to their original SHA identities.
    for s in old.SEEDS:
        marker=previous/f'eval_historical_{s}'/'complete.json'; value=ab.read(marker)
        gmp.verify(unique_records(value['artifacts']),'Historical evaluation')
        records += value['artifacts'] + ab.identities([marker])
    records += ab.identities([previous/'manifest.json',previous/'summary.json',prepared_path])
    first=refs[42]['job']
    if any(refs[s]['job']['cache_identity']!=first['cache_identity'] or refs[s]['job']['val_ids']!=first['val_ids'] or refs[s]['job']['proposals']!=first['proposals'] for s in old.SEEDS):
        raise ValueError('Seeds use different query inputs')
    for stem in first['val_ids']:
        entry=queue.cache_read(Path(first['cache'])/'val'/stem,first['cache_identity'][stem])
        if entry is None: raise ValueError('Missing query cache')
        records += entry['artifacts'] + ab.identities([Path(first['cache'])/'val'/stem/'complete.json'])
    return refs, unique_records(records)


def summarize(results):
    groups={}; contrasts={}
    for method in METHODS:
        scores={str(s):results[f'{method}_{s}']['miou'] for s in old.SEEDS if f'{method}_{s}' in results}
        complete=len(scores)==3
        groups[method]=dict(complete=complete,seeds=scores,mean=float(np.mean(list(scores.values()))) if complete else None,
            sample_std=float(np.std(list(scores.values()),ddof=1)) if complete else None)
        if method!='baseline' and complete and groups['baseline']['complete']:
            delta={str(s):100*(scores[str(s)]-groups['baseline']['seeds'][str(s)]) for s in old.SEEDS}
            contrasts[method+'-baseline']=dict(paired_pp=delta,mean_pp=float(np.mean(list(delta.values()))))
    return dict(groups=groups,contrasts=contrasts,results=results,
        note='VOC val development experiment, no significance or deployment claim; label-only SAM control retains original ranking scalar')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--version',default='sam3_semantic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--previous',type=Path,default=ROOT/'experiments/background_partition_resume_20261006_002500')
    parser.add_argument('--gpus',nargs=2,type=int,default=[1,0]); parser.add_argument('--max-hours',type=float,default=24)
    parser.add_argument('--preflight-only',action='store_true'); parser.add_argument('--resume',action='store_true')
    opts=parser.parse_args(); os.chdir(ROOT)
    if opts.worker: return worker(ab.read(opts.worker))
    ab.parse_args(['--version',opts.version])
    if len(set(opts.gpus))!=2 or min(opts.gpus)<0 or not math.isfinite(opts.max_hours) or opts.max_hours<=0: raise ValueError('Invalid GPU/time')
    refs,records=history(opts.previous.resolve())
    from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
    vocab=SAM3PseudoMaskGenerator(dataset_type='voc',load_model=False)
    classes=old.canonical([asdict(c) for c in vocab.active_classes])
    code=ab.source_identity()
    identity=old.canonical(dict(protocol=PROTOCOL,previous=str(opts.previous.resolve()),records=records,code=code,
        active=ACTIVE,rule=RULE,active_classes=classes,methods=METHODS,seeds=old.SEEDS,gpus=opts.gpus))
    output=ROOT/'experiments'/opts.version; cache=ROOT/'storage/proposals/voc'/opts.version/'semantic_evidence'; logs=ROOT/'logs'/opts.version
    if output.exists() and (not opts.resume or ab.read(output/'manifest.json')!=identity): raise ValueError('Existing version differs or needs --resume')
    if not output.exists() and (cache.parent.exists() or logs.exists()): raise FileExistsError('Version paths used')
    for gpu in opts.gpus: ab.check_gpu(gpu)
    queue.ensure_resources(float('inf'))
    if queue.shutil.disk_usage(ROOT).free<queue.FLOOR+2*2**30: raise queue.Deferred('Need 2 GiB allowance plus 5 GiB reserve')
    print('[Preflight passed] 3 exact historical replays; full-vocabulary semantic cache; 9 label-only evaluations; no training/HCR/GT class constraints',flush=True)
    if opts.preflight_only: return
    if not output.exists(): ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    budget=output/'budget.json'
    if not budget.exists(): ab.write(budget,dict(start=time.time()))
    started=ab.read(budget)['start']; deadline=started+opts.max_hours*3600
    cancel=threading.Event(); guard=threading.Lock(); stages=AliasAwareStages(output,True)
    status='running'; error=None; results={}; failures={}
    entry=queue.__file__; queue.__file__=__file__
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_): cancel.set(); raise KeyboardInterrupt('Interrupted')
    for s in handlers: signal.signal(s,stop)
    def flush():
        ab.write(output/'status.json',dict(status=status,error=error,completed=list(results),failures=failures,expected_evaluations=12))
        ab.write(output/'summary.json',dict(status=status,error=error,failures=failures,**summarize(results)))
    def launch(key,job,gpu):
        def action(attempt):
            job.update(attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline,
                protocol=PROTOCOL,rule=RULE,active=ACTIVE,active_classes=classes,semantic_cache=str(cache),semantic_source_sha=old.object_hash(code))
            env=os.environ.copy(); env.update(refs[job['seed']]['environment']); env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            env=evaluation_environment(job,env)
            if job['stage'].startswith('semantic_'): env['OVERRIDE_FEATURE_ADAPTER_CKPT']=''
            log=logs/f'{key}_{attempt.name}.log'
            queue.launch(job,attempt,env,log,cancel,deadline)
            artifacts=[attempt,log]
            if job['stage'].startswith('semantic_'):
                for stem in job['ids']:
                    marker=cache/stem/'complete.json'
                    artifacts += [marker,*[Path(r['path']) for r in ab.read(marker)['artifacts']]]
            return dict(attempt=str(attempt)),artifacts
        return stages.run(key,action)
    def evaluate(method,seed,gpu):
        job=copy.deepcopy(refs[seed]['job']); job.update(stage='eval',semantic_method=method)
        value=launch(f'eval_{method}_{seed}',job,gpu); attempt=Path(value['attempt'])
        metric=ab.read(attempt/'metrics.json'); ref=ab.read(refs[seed]['metrics'])
        current=provenance.evidence.checked_copy(metric)[0]; reference=provenance.evidence.checked_copy(ref)[0]
        old.validate_eval(current,reference,job['val_ids'],ref['proposal_count_after'],.2)
        old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),metric)
        if method=='baseline':
            old.require_image_replay(refs[seed]['per_image'],str(attempt/'per_image_confusion.npz'))
            if metric['confusion_matrix']!=ref['confusion_matrix']: raise queue.GlobalFailure('Historical replay differs')
        cm=np.asarray(metric['confusion_matrix'],np.int64)
        result=dict(miou=current['mIoU'],metrics=str(attempt/'metrics.json'),per_image=str(attempt/'per_image_confusion.npz'),
            per_class_iou=current['per_class_iou'],candidate_diagnostics=ab.read(attempt/'diagnostics.json')['counts'],
            pixels=dict(background_fp=int(cm[0,1:].sum()),foreground_fn=int(cm[1:,0].sum()),
                foreground_confusion=int(cm[1:,1:].sum()-np.trace(cm[1:,1:]))),worker=ab.read(attempt/'queue_worker.json'))
        with guard: results[f'{method}_{seed}']=result; flush()
    def fatal(item,exc): raise exc
    try:
        flush(); stages.verify_all()
        queue.run_round(list(old.SEEDS),opts.gpus,lambda s,g:evaluate('baseline',s,g),cancel,deadline,fatal)
        common=copy.deepcopy(refs[42]['job'])
        launch('semantic_probe',dict(common,stage='semantic_probe',ids=common['val_ids'][:1]),opts.gpus[0])
        queue.run_round([0,1],opts.gpus,lambda lane,g:launch(f'semantic_extract_{lane}',dict(common,stage='semantic_extract',ids=common['val_ids'][lane::2]),g),cancel,deadline,fatal)
        def failed(item,exc):
            with guard: failures[str(item)]=str(exc); flush()
        queue.run_round([(m,s) for s in old.SEEDS for m in METHODS[1:]],opts.gpus,
            lambda item,g:evaluate(*item,g),cancel,deadline,failed)
        gmp.verify(records,'Historical inputs after execution'); gmp.verify(code,'Frozen source'); stages.verify_all()
        status='complete' if len(results)==12 else 'partial'
    except queue.Deferred as exc: status='deferred'; error=str(exc)
    except BaseException as exc: status='failed'; error=str(exc); cancel.set(); raise
    finally:
        flush(); ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(cache.parent),str(logs)],automatic_deletion=False))
        queue.__file__=entry
        for s,h in handlers.items(): signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(opts.version,status,str(output),time.time()-started)


if __name__=='__main__': main()
