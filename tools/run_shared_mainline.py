"""Portable VOC mainline: one frozen SAM3 object, same-image encoding reuse.

Build raw regions, train with train_feature_adapter.py, postfilter with
postfilter_lsa_bank.py, then predict here. Predictions never read GT.
"""
from pathlib import Path
import argparse,json,os,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'sam3-main')]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(path):path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['build','predict']);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--image-list',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--imagelevel-json',type=Path);p.add_argument('--bank',type=Path);p.add_argument('--adapter',type=Path);p.add_argument('--gpu',type=int,default=1);p.add_argument('--seed',type=int,choices=[42,123,2026],default=42);a=p.parse_args(argv)
 if a.phase=='build' and a.imagelevel_json is None:p.error('build requires --imagelevel-json')
 if a.phase=='predict' and (a.bank is None or a.adapter is None):p.error('predict requires --bank and --adapter')
 if a.output_dir.exists():p.error('Use a fresh output directory')
 return a
def visible_gpu(a):
 if os.environ.get('SLURM_JOB_ID'):
  visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
  if not visible or ',' in visible or visible=='-1':raise RuntimeError('Slurm must allocate exactly one visible GPU; keep its device assignment')
  return visible
 return str(a.gpu)
def create_engine(a):
 import torch,numpy as np,random
 from src.sam3_runtime import build_sam3_model
 from src.feature_extractor_sam3 import SAM3FeatureExtractor
 from src.sam3_shared_encoding import use_model_for_extractor,ImageEncodingReuse
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU is required')
 device=torch.device('cuda:0');model=build_sam3_model(device,checkpoint_path=str(a.checkpoint),enable_inst_interactivity=False)
 with use_model_for_extractor(model):ex=SAM3FeatureExtractor(model_path=str(a.checkpoint),device=device,feature_source='trunk_fpn',token_norm='layernorm',context_subtract=0.,use_interactivity=False)
 if ex.model is not model or getattr(ex,'image_token_cache',None) is not None:raise RuntimeError('Shared object required; full-image disk cache disabled')
 return model,device,ex,ImageEncodingReuse(ex)
def build(a,ids):
 import torch,numpy as np
 from PIL import Image
 from tools.generate_pseudo_masks_sam3 import SAM3PseudoMaskGenerator
 from tools.proposal_demo.active_proposer import SAM3ActiveProposer
 from src.pseudo_label_cleanup import clean_semantic_labels
 from src.region_extractors import _extract_class_regions
 from src.hard_center_gmp import aggregate
 from src.feature_bank_storage import save_feature_matrix_npz
 from src.feature_signature import save_feature_signature
 model,device,ex,reuse=create_engine(a)
 gen=SAM3PseudoMaskGenerator(dataset_type='voc',dataset_root=a.dataset_root,ckpt=str(a.checkpoint),device=str(device),load_model=False,use_bf16=False,conf_thr=.5,min_mask_area=100,force_presence_one=True,edge_repair=True,mask_logit_threshold=.4,close_kernel=3,close_iterations=1,dilate_pixels=1,diningtable_close_iterations=2,overlap_policy='class_order')
 gen.active_proposer=SAM3ActiveProposer(model,device,gen.active_classes,preview=True,bucket_size=4,neighbor_tokens=2,hypernym_tokens=2,context_strength=1.,confidence_threshold=.5,min_mask_area=100,min_mask_region_area=0,max_mask_area_ratio=0.,max_masks=0,mask_nms_threshold=.7,mask_probability_threshold=.4,force_presence_one=True)
 if len(gen.classes)!=20:raise ValueError('This portable release targets the accepted VOC protocol')
 rows=json.loads(a.imagelevel_json.read_text(encoding='utf-8'))['images'];labels={Path(x['file_name']).stem:x['labels'] for x in rows}
 if not set(ids)<=set(labels):raise ValueError('Missing image-level labels')
 features={i:[] for i in range(21)};sample_ids={i:[] for i in range(21)};paths=[]
 for pos,stem in enumerate(ids,1):
  image=a.dataset_root/'JPEGImages'/(stem+'.jpg');rgb=np.asarray(Image.open(image).convert('RGB')).copy()
  with torch.inference_mode():
   with reuse.capture():pseudo=gen.generate_mask(image,labels[stem])
   tokens,grid=reuse.tokens(rgb);pseudo,_=clean_semantic_labels(pseudo,num_classes=21);regions=_extract_class_regions(pseudo,300)
   for i,r in enumerate(regions):
    vector=aggregate(tokens[0],r['mask'],1008,grid,.1).cpu().numpy();assert vector.shape==(1280,) and np.isfinite(vector).all() and np.linalg.norm(vector)>1e-6;features[r['cls_id']].append(vector);sample_ids[r['cls_id']].append(stem+'_'+str(i).zfill(6))
  p=a.output_dir/'pseudo'/(stem+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:Image.fromarray(pseudo.astype(np.uint8)).save(f,format='PNG')
  paths.append(p);del tokens,regions,rgb,pseudo
  print('BUILD',pos,len(ids),flush=True)
 bank=a.output_dir/'raw_bank';names=['background']+gen.classes;write(bank/'meta/class_mapping.json',dict(id_to_dirname={str(i):n for i,n in enumerate(names)}));save_feature_signature(bank/'meta',ex.feature_signature())
 for i,name in enumerate(names):
  if not features[i]:raise ValueError('Missing class '+name)
  save_feature_matrix_npz(bank/'features'/name/(name+'_features.npz'),np.stack(features[i]),sample_ids[i])
 return paths+sorted(p for p in bank.rglob('*') if p.is_file())
def predict(a,ids):
 import torch,numpy as np
 from PIL import Image
 from tools.oracle_guided_r import _classes,decode_image
 from tools.proposal_demo.active_proposer import SAM3ActiveProposer
 from src.feature_adapter import load_adapter,adapted_signature,project_features
 from src.inference import load_class_mapping,load_class_features,build_faiss_index
 from src.semantic_bank_filter import validate_filtered_bank,gallery_class_order
 from src.retrieval import build_retrieval_context,classify_queries
 from src.proposal_cleanup import CleanupConfig,clean_proposals
 from src.semantic_rescue import HCRConfig,prepare_pool
 from src.hard_center_gmp import aggregate
 from tools.small_mask_hcr_core import assemble
 from tools.r3_crossview_core import query_fields
 from src.progressive_retrieval_recovery import recover_from_queries
 model,device,ex,reuse=create_engine(a);post=validate_filtered_bank(a.bank);assert len(post['classes'])==21
 mapping,fg,bg=load_class_mapping(str(a.bank/'meta'));order=gallery_class_order(a.bank,fg+bg);d=load_class_features(str(a.bank/'features'),order);gallery=np.vstack(list(d.values())).astype(np.float32);gallery=np.ascontiguousarray(gallery/(np.linalg.norm(gallery,axis=1,keepdims=True)+1e-8));labels=[k for k,v in d.items() for _ in v];index=build_faiss_index(gallery);context=build_retrieval_context(gallery,labels);adapter,_=load_adapter(str(a.adapter),device,ex.feature_signature());assert json.loads((a.bank/'meta/feature_signature.json').read_text())==adapted_signature(ex.feature_signature(),str(a.adapter),adapter.architecture)
 proposer=SAM3ActiveProposer(model,device,_classes({}),preview=True,bucket_size=4,neighbor_tokens=2,hypernym_tokens=2,context_strength=1.,confidence_threshold=.15,min_mask_area=100,min_mask_region_area=0,max_mask_area_ratio=.95,max_masks=0,mask_nms_threshold=.7)
 cropper=SAM3ActiveProposer(model,device,_classes({}),preview=False,bucket_size=4,neighbor_tokens=2,hypernym_tokens=2,context_strength=1.,confidence_threshold=.15,mask_probability_threshold=.5);paths=[];traces=[]
 for pos,stem in enumerate(ids,1):
  image=a.dataset_root/'JPEGImages'/(stem+'.jpg');rgb=np.asarray(Image.open(image).convert('RGB')).copy();start=time.monotonic()
  with torch.inference_mode():
   with reuse.capture():inst=proposer(np.ascontiguousarray(rgb[...,::-1]))['instances']
   tokens,grid=reuse.tokens(rgb);raw=inst.pred_masks.cpu().numpy().astype(bool);scores=inst.scores.cpu().numpy().astype(np.float32);crop,geometry=decode_image(cropper,rgb,stem,'R3');clean,cs,_,_=clean_proposals(raw,scores,CleanupConfig());pool,ps,flags=prepare_pool(clean,cs,raw,scores,CleanupConfig(),HCRConfig());pool=np.asarray(pool,bool).reshape(-1,*rgb.shape[:2]);ps=np.asarray(ps,np.float32);flags=np.asarray(flags,bool);masks=list(pool)+list(crop['masks']);vectors=[aggregate(tokens[0],mask,1008,grid,.1) for mask in masks];x=torch.nn.functional.normalize(torch.stack(vectors).unsqueeze(0).mean(0),dim=1).cpu().numpy() if vectors else np.zeros((0,1280),np.float32);z=project_features(adapter,x,device);z=np.ascontiguousarray(z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-8));rows=classify_queries(z,gallery,labels,index,context,25,'balanced',5,2,.2) if len(z) else [];inv={v:k for k,v in mapping.items()};cl=np.asarray([inv[r.label] for r in rows],np.int16);rr=np.asarray([r.score for r in rows],np.float32);mm=np.asarray([r.margin for r in rows],np.float32);base,_=assemble(pool,ps,flags,cl[:len(pool)],rr[:len(pool)],mm[:len(pool)],[mapping[i] for i in range(21)],'C_area100');q=query_fields(dict(class_ids=cl[len(pool):],retrieval_scores=rr[len(pool):],margins=mm[len(pool):]),crop['scores'],crop['candidate_ids'],crop['source_classes']);out=recover_from_queries(base['HCR'],crop['masks'],q,stem)[0]['E'];assert np.array_equal(out[base['HCR']!=0],base['HCR'][base['HCR']!=0])
  p=a.output_dir/'predictions'/(stem+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:Image.fromarray(out.astype(np.uint8)).save(f,format='PNG')
  paths.append(p);traces.append(dict(image_id=stem,seconds=time.monotonic()-start,original_encoding_shared=True,crop_encoding_separate=True,full_image_features_saved=False));del tokens,x,z,vectors,masks,raw,pool,crop,inst,rgb
  print('PREDICT',pos,len(ids),flush=True)
 p=a.output_dir/'trace.json';write(p,traces);return paths+[p]
def main(argv=None):
 a=parse_args(argv);ids=a.image_list.read_text().split()
 if not ids or len(set(ids))!=len(ids) or any(Path(i).name!=i or '/' in i or '\\' in i or i in ['.','..'] for i in ids):raise ValueError('Image list requires unique plain image stems')
 os.environ.update(CUDA_VISIBLE_DEVICES=visible_gpu(a),OVERRIDE_DATASET_TYPE='voc',OVERRIDE_IMAGE_TOKEN_CACHE_DIR='',OVERRIDE_SAM3_POOLING='hard_center_gmp',OVERRIDE_SAM3_GMP_RHO='.1',OVERRIDE_SAM3_RESOLUTION='1008',OVERRIDE_SAM3_USE_BF16='false',OVERRIDE_SAM3_FEATURE_SOURCE='trunk_fpn',OVERRIDE_SAM3_FEATURE_LEVEL='-1',OVERRIDE_SAM3_TOKEN_NORM='layernorm',OVERRIDE_SAM3_CONTEXT_SUBTRACT='0',OVERRIDE_FEATURE_ADAPTER_CKPT='',OVERRIDE_BUILD_SPLIT_CONNECTED_COMPONENTS='true',NVIDIA_TF32_OVERRIDE='0');a.output_dir.mkdir(parents=True,exist_ok=False);start=time.time();write(a.output_dir/'request.json',dict(phase=a.phase,checkpoint=record(a.checkpoint),image_list=record(a.image_list),images=len(ids),seed=a.seed,full_image_feature_cache=False))
 paths=build(a,ids) if a.phase=='build' else predict(a,ids);write(a.output_dir/'completion.json',dict(status='complete',images=len(ids),GT_decoded=False,artifacts=[record(p) for p in paths],start_unix=start,end_unix=time.time(),full_image_feature_cache=False))
if __name__=='__main__':main()
