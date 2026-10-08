"""Read-only, three-seed gallery provenance audit. GT is diagnostic only."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experimental_mechanism_audit as audit
from tools import experimental_rescue_evidence as evidence
ab, queue, gmp, old = audit.ab, audit.queue, audit.gmp, audit.old
DEPLOY = ROOT / 'experiments/hcr_mainline_20261004_214541'


def histogram(mask, gt):
    mask = np.asarray(mask, bool)
    if mask.shape != gt.shape or not np.isin(gt, list(range(21))+[255]).all():
        raise ValueError('Invalid mask/GT geometry or labels')
    h = np.bincount(gt[mask & (gt != 255)], minlength=21)
    return dict(histogram=h.tolist(), valid_pixels=int(h.sum()),
                majority=int(h.argmax()) if h.sum() else None,
                purity=float(h.max()/h.sum()) if h.sum() else None)


def keyed_plan(plan):
    result = {}
    for stem, rows in plan.items():
        for row in rows:
            key = (row['cls'], row['id'])
            if key in result: raise ValueError('Duplicate class/sample identity')
            result[key] = dict(row, image=stem)
    return result


def gallery(bank):
    from src.feature_bank_storage import load_feature_matrix_npz
    mapping = ab.read(Path(bank)/'meta/class_mapping.json')['id_to_dirname']
    names = [mapping[k] for k in sorted(mapping, key=int) if int(k) != 0]
    names += [mapping['0']]
    feats, keys = [], []
    for name in dict.fromkeys(names):
        files = list((Path(bank)/'features'/name).glob('*.npz'))
        if len(files) != 1: raise ValueError('Missing/ambiguous class bank: '+name)
        x, ids = load_feature_matrix_npz(files[0])
        if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1)<1e-6).any():
            raise ValueError('Invalid bank vectors')
        feats.append(x); keys.extend((name, str(s)) for s in ids)
    if len(keys) != len(set(keys)): raise ValueError('Duplicate gallery rows')
    x = np.vstack(feats).astype(np.float32)
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8), keys


def catalog(job, attempt):
    """Replay exact region identity; never infer a component from ID suffix."""
    from PIL import Image
    from src.region_extractors import get_regions_from_pseudo
    from configs.config import DATASET_CLASSES
    plan = ab.read(job['plan']); keyed_plan(plan)
    rows, records = [], []
    for pos, stem in enumerate(job['train_ids'], 1):
        queue.ensure_resources(job['deadline'])
        identity = job['cache_identity'][stem]
        pseudo = Path(job['pseudo'])/(stem+'.png')
        if ab.digest(pseudo) != identity['mask_source_sha256']: raise ValueError('Pseudo changed: '+stem)
        if ab.digest(job['images'][stem]) != identity['image_sha256']: raise ValueError('Image changed: '+stem)
        regions = get_regions_from_pseudo(job['images'][stem], 300, job['pseudo'])
        masks = [regions[r['index']]['mask'] for r in plan[stem]]
        if queue.mask_hashes(masks) != [r['mask_sha256'] for r in plan[stem]]:
            raise ValueError('Region geometry differs: '+stem)
        folder = Path(job['cache'])/'train'/stem
        value = queue.cache_read(folder, identity)
        if value is None or value['mask_hashes'] != queue.mask_hashes(masks):
            raise ValueError('Training cache identity differs')
        with np.load(value['file'], allow_pickle=False) as data:
            if list(data['ids']) != [r['id'] for r in plan[stem]] or list(data['classes']) != [r['cls'] for r in plan[stem]]:
                raise ValueError('Training cache row identities differ')
        gtfile = Path(job['gt_dir'])/(stem+'.png')
        if not gtfile.exists(): raise FileNotFoundError('Training diagnostic GT unavailable: '+str(gtfile))
        with Image.open(gtfile) as im: gt = np.asarray(im)
        records += ab.identities([gtfile, pseudo, job['images'][stem], folder/'complete.json']) + value['artifacts']
        for r, mask in zip(plan[stem], masks):
            cid = regions[r['index']]['cls_id']
            label = 'background' if cid == 0 else DATASET_CLASSES[cid-1]
            if label != r['cls']: raise ValueError('Region class changed')
            h = histogram(mask, gt)
            rows.append(dict(r, image=stem, **h, area=int(mask.sum()),
                label_fraction=h['histogram'][cid]/h['valid_pixels'] if h['valid_pixels'] else None,
                provenance='exact_region_index_and_mask_sha256', gt_role='diagnostic_only'))
        if pos % 200 == 0: print(f'[Catalog] {pos}/{len(job["train_ids"])}', flush=True)
    ab.write(attempt/'regions.json', dict(rows=rows, records=records))
    return dict(catalog=str(attempt/'regions.json')), [attempt]


def install_trace(job):
    audit.install(job)
    original = queue.install_cached_eval
    def install(j):
        original(j)
        import faiss
        faiss.omp_set_num_threads(4)
        import run_inference as inference
        import src.inference as inf
        import src.retrieval as retrieval
        from src.feature_adapter import project_features
        from PIL import Image
        projected, keys = gallery(job['bank'])
        labels = np.asarray([k[0] for k in keys])
        provenance = {(r['cls'],r['id']):r for r in ab.read(job['catalog'])['rows']}
        plan = ab.read(job['plan']); raw_by_key = {}
        for stem, rows in plan.items():
            x = queue.cache_vectors(Path(job['cache'])/'train'/stem, job['cache_identity'][stem], 'C', [r['mask_sha256'] for r in rows])
            for r, v in zip(rows, x): raw_by_key[(r['cls'],r['id'])] = v
        if set(keys) != set(raw_by_key) or set(keys) != set(provenance): raise ValueError('Raw/projected/provenance membership differs')
        raw = np.asarray([raw_by_key[k] for k in keys],np.float32)
        del raw_by_key
        raw /= np.linalg.norm(raw,axis=1,keepdims=True)+1e-8
        ctx = retrieval.build_retrieval_context(raw, labels)
        index = faiss.IndexFlatIP(raw.shape[1]); index.add(raw)
        state = {}; trace = []; checked = False
        load, build, classify = inference.load_strict_proposals, inference.build_feature_extractor, inf.classify_queries
        def load_hook(directory, stem, h, w):
            masks, scores = load(directory, stem, h, w)
            state.clear(); state.update(stem=stem,masks=masks,scores=scores)
            return masks, scores
        def build_hook(*args, **kwargs):
            model = build(*args, **kwargs)
            job['_trace_model'] = model
            return model
        def classify_hook(*args, **kwargs):
            nonlocal checked
            result = classify(*args, **kwargs)
            if not checked:
                if list(kwargs['all_labels']) != list(labels) or not np.array_equal(kwargs['all_feats'],projected):
                    raise ValueError('Production gallery order differs')
                model = job['_trace_model']
                reproduction = project_features(model.adapter,raw,model.device)
                error = float(np.max(np.abs(reproduction-projected)))
                if not np.allclose(reproduction,projected,atol=2e-5,rtol=2e-5):
                    raise ValueError('MLP export does not reproduce from raw rows')
                job['_projection_error'] = error; checked = True
            stem = state['stem']; masks = state['masks']
            qraw = queue.cache_vectors(Path(job['cache'])/'val'/stem,job['cache_identity'][stem],'C',queue.mask_hashes(masks))
            model = job['_trace_model']; qp = project_features(model.adapter,qraw,model.device)
            valid = np.flatnonzero(np.isfinite(qp).all(1)&(np.linalg.norm(qp,axis=1)>=1e-6))
            expected = qp[valid]/(np.linalg.norm(qp[valid],axis=1,keepdims=True)+1e-8)
            if not np.array_equal(expected,kwargs['query_feats']): raise ValueError('Query row alignment differs')
            qr = qraw[valid]/(np.linalg.norm(qraw[valid],axis=1,keepdims=True)+1e-8)
            search_k = min(max(kwargs['topk'],kwargs['class_topm']*len(ctx.labels)*kwargs['search_multiplier']),len(keys))
            spaces = {}
            for name,q,x,ix,context in [('raw',qr,raw,index,ctx),('mlp',kwargs['query_feats'],projected,kwargs['faiss_index'],kwargs['context'])]:
                sims, inds = retrieval._search(q,x,ix,search_k)
                local,center,fused,support = audit.decompose(q,context,labels,sims,inds,kwargs['class_topm'],kwargs['centroid_weight'])
                spaces[name] = (sims,inds,local,center,fused,support,context)
            with Image.open(Path(job['gt_dir'])/(stem+'.png')) as im: gt = np.asarray(im)
            class_names = ['background']+list(inf.DATASET_CLASSES)
            for n, (pi,pred) in enumerate(zip(valid,result)):
                h = histogram(masks[pi],gt); truth = class_names[h['majority']] if h['majority'] is not None else None
                row = dict(image=stem,proposal_index=int(pi),mask_sha256=queue.mask_hashes([masks[pi]])[0],
                    sam3_score=float(state['scores'][pi]),truth_majority=truth,**h,spaces={})
                for name, (sims,inds,local,center,fused,support,context) in spaces.items():
                    def neighbors(ii):
                        return [dict(gallery_row=int(k),label=keys[k][0],sample_id=keys[k][1],
                            image=provenance[keys[k]]['image'], similarity=float(s),
                            source_label_fraction=provenance[keys[k]]['label_fraction']) for k,s in ii]
                    pairs=list(zip(inds[n].tolist(),sims[n].tolist()))
                    winner=context.labels[np.argsort(fused[n])[::-1][0]]
                    row['spaces'][name]=dict(classes=list(context.labels),local=local[n].tolist(),center=center[n].tolist(),
                        fused=fused[n].tolist(),support=support[n].tolist(),prediction=winner,
                        local_prediction=context.labels[np.argsort(local[n])[::-1][0]],
                        neighbors=neighbors(pairs[:20]),
                        decision_neighbors={c:neighbors([(k,s) for k,s in pairs if labels[k]==c][:5]) for c in dict.fromkeys([winner,truth]) if c is not None})
                if row['spaces']['mlp']['prediction']!=pred.label: raise ValueError('Trace prediction differs')
                trace.append(row)
            return result
        inference.load_strict_proposals=load_hook; inference.build_feature_extractor=build_hook; inf.classify_queries=classify_hook
        job['_trace']=trace
    queue.install_cached_eval=install


def summarize_trace(rows):
    counts=Counter(); hubs=Counter(); error_hubs=Counter()
    for r in rows:
        counts['candidates']+=1
        for n in r['spaces']['mlp']['neighbors'][:5]: hubs[(n['label'],n['sample_id'])]+=1
        if r['purity'] is None or r['purity']<.9: continue
        counts['pure_candidates']+=1
        truth=r['truth_majority']; before=r['spaces']['raw']; after=r['spaces']['mlp']
        wrong=after['prediction']!=truth
        counts['pure_wrong']+=int(wrong)
        counts['raw_correct_mlp_wrong']+=int(before['prediction']==truth and wrong)
        counts['raw_wrong_mlp_correct']+=int(before['prediction']!=truth and not wrong)
        counts['local_correct_fused_wrong']+=int(after['local_prediction']==truth and wrong)
        counts['local_wrong_fused_correct']+=int(after['local_prediction']!=truth and not wrong)
        if wrong:
            counts['foreground_to_background']+=int(truth!='background' and after['prediction']=='background')
            for n in after['decision_neighbors'][after['prediction']]: error_hubs[(n['label'],n['sample_id'])]+=1
    def top(c): return [dict(label=k[0],sample_id=k[1],query_count=v) for k,v in c.most_common(100)]
    return dict(counts=dict(counts),frequent_neighbors=top(hubs),wrong_class_supporters=top(error_hubs),
                note='Overlapping candidate counts, NOT unique erroneous pixels or mIoU contributions; local scores retain centroid fallback. GT diagnostic only.')


def worker(job):
    install_trace(job); old.previous.worker(job)
    dest=Path(job['attempt'])
    with (dest/'query_provenance.jsonl').open('x',encoding='utf8') as f:
        for row in job['_trace']: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    ab.write(dest/'diagnostics.json',dict(summarize_trace(job['_trace']),projection_max_abs=job['_projection_error']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',default='bank_provenance_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--worker',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    if args.worker:return worker(ab.read(args.worker))
    ab.parse_args(['--version',args.version])
    prepared=ab.read(DEPLOY/'prepared.json');deployed=ab.read(DEPLOY/'deployment.json')
    if ab.read(DEPLOY/'status.json')['status']!='complete':raise ValueError('Mainline regression incomplete')
    gmp.verify(deployed['code_after'],'Deployed source')
    changed={str(ROOT/x) for x in ('run_inference.py','src/inference.py')}
    retained=[r for r in prepared['records'] if r['path'] not in changed]
    gmp.verify(retained,'Historical inputs and unchanged source')
    records=retained+ab.identities([DEPLOY/'prepared.json',DEPLOY/'deployment.json',DEPLOY/'status.json'])
    job0=prepared['jobs']['baseline_42']['job']
    records+=ab.identities([job0['plan']])
    queue.ensure_resources(float('inf'))
    if queue.shutil.disk_usage(ROOT).free<6*2**30:raise ValueError('Need 6 GiB free including 5 GiB reserve')
    identity=dict(protocol='bank_provenance_v1',code=ab.source_identity(),records=records,seeds=list(old.SEEDS),
        scope='clean proposals; HCR unchanged and excluded from attribution replay',purity=.9,top_neighbors=20)
    output=ROOT/'experiments'/args.version
    if output.exists() and (not args.resume or ab.read(output/'manifest.json')!=identity):raise ValueError('Existing version differs or --resume missing')
    print('[Preflight passed] exact region catalog + three-seed clean baseline replay and raw/MLP retrieval trace',flush=True)
    if args.preflight_only:return
    if not output.exists():ab.write(output/'manifest.json',identity)
    lock=output/'launcher.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    cancel=threading.Event();start=time.time();deadline=start+24*3600;status='running';error=None;reports={}
    saved={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(*_):cancel.set();raise KeyboardInterrupt('Audit interrupted')
    for s in saved:signal.signal(s,stop)
    def flush():ab.write(output/'status.json',dict(status=status,error=error,completed=list(reports)))
    entry=queue.__file__;queue.__file__=__file__
    try:
        flush();stages=ab.Stages(output,True);stages.verify_all()
        j=copy.deepcopy(job0);j['deadline']=deadline
        cat=stages.run('catalog',lambda attempt:catalog(j,attempt))
        catalog_data=ab.read(cat['catalog']);gmp.verify(catalog_data['records'],'Catalog inputs')
        guard=threading.Lock()
        def execute(seed,gpu):
            source=prepared['jobs'][f'baseline_{seed}']
            def action(attempt):
                job=copy.deepcopy(source['job']);job.update(attempt=str(attempt),metrics=str(attempt/'metrics.json'),deadline=deadline,catalog=cat['catalog'])
                env=os.environ.copy();env.update(source['command']['environment']);env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                log=ROOT/'logs'/args.version/f'seed{seed}_{attempt.name}.log'
                queue.launch(job,attempt,env,log,cancel,deadline)
                current=ab.read(job['metrics']);ref=ab.read(source['metrics'])
                old.validate_eval(evidence.checked_copy(current)[0],evidence.checked_copy(ref)[0],job['val_ids'],ref['proposal_count_after'],.2)
                old.seeds.validate_image_matrices(str(attempt/'per_image_confusion.npz'),current)
                old.require_image_replay(source['reference']['per_image'],str(attempt/'per_image_confusion.npz'))
                if current['confusion_matrix']!=ref['confusion_matrix']:raise ValueError('Baseline replay differs')
                return dict(diagnostics=ab.read(attempt/'diagnostics.json'),trace=str(attempt/'query_provenance.jsonl'),baseline_reproduced=True),[attempt,log]
            try:r=stages.run(f'seed_{seed}',action)
            except BaseException:traceback.print_exc();raise
            with guard:reports[str(seed)]=r;flush();ab.write(output/'summary.json',dict(status='running',results=reports,catalog=cat['catalog']))
        queue.run_round(list(old.SEEDS),(1,0),execute,cancel,deadline,lambda item,exc:(_ for _ in ()).throw(exc))
        gmp.verify(identity['code'],'Source');gmp.verify(records,'Sources');gmp.verify(catalog_data['records'],'Catalog')
        status='complete'
    except BaseException as exc:status='failed';error=str(exc);cancel.set();raise
    finally:
        flush();ab.write(output/'summary.json',dict(status=status,error=error,results=reports,gt_role='diagnostic_only',evaluation_ids=job0['val_ids']))
        ab.write(output/'artifacts.json',dict(owned_roots=[str(output),str(ROOT/'logs'/args.version)],historical_inputs_read_only=True))
        queue.__file__=entry
        for s,h in saved.items():signal.signal(s,h)
        lock.unlink()
        from project_utils.experiment_notify import notify_experiment
        notify_experiment(args.version,status,str(output),time.time()-start)


if __name__=='__main__':main()
