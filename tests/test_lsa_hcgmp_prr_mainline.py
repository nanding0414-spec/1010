"""CPU contracts for the published mainline; no GPU quality claim."""
from pathlib import Path
import hashlib,json,tempfile,unittest
import numpy as np
from src.semantic_bank_filter import export_filtered_bank,select_members,validate_filtered_bank,gallery_class_order
from src.feature_bank_storage import save_feature_matrix_npz,load_feature_matrix_npz
from src.progressive_retrieval_recovery import recover_from_queries
from tools.bank_postfilter_sweep_core import trim
from tools.r3_postfilter5_e_core import compose
ROOT=Path(__file__).resolve().parents[1]

class MainlineTests(unittest.TestCase):
    def test_gallery_byte_preservation_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';target=Path(td)/'post05';(source/'meta').mkdir(parents=True)
            mapping={str(i):'background' if i==0 else f'c{i:02}' for i in range(21)}
            (source/'meta/class_mapping.json').write_text(json.dumps(dict(id_to_dirname=mapping)))
            (source/'meta/feature_signature.json').write_text(json.dumps(dict(adapter={'sha256':'fixture'})))
            rng=np.random.default_rng(42)
            for i,name in mapping.items():
                x=rng.normal(size=(40,256)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
                save_feature_matrix_npz(source/'features'/name/(name+'_features.npz'),x,[f'image_{i}_{j}' for j in range(40)])
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            report=export_filtered_bank(source,target)
            for cid,name in mapping.items():
                x,ids=load_feature_matrix_npz(source/'features'/name/(name+'_features.npz'))
                keep,_=trim(x,ids,5,cid=='0');actual,names=load_feature_matrix_npz(target/'features'/name/(name+'_features.npz'))
                np.testing.assert_array_equal(actual,x[keep]);np.testing.assert_array_equal(names,ids[keep])
                self.assertEqual(len(actual),40 if cid=='0' else 38)
            self.assertEqual(report['total_kept'],800)
            self.assertEqual(validate_filtered_bank(target)['total_kept'],800)
            self.assertEqual(gallery_class_order(target,list(reversed(mapping.values()))),sorted(mapping.values()))
            self.assertEqual(gallery_class_order(source,list(reversed(mapping.values()))),list(reversed(mapping.values())))
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):export_filtered_bank(source,target)
            with self.assertRaises(ValueError):export_filtered_bank(target,Path(td)/'twice')

    def test_fixed_filter_nonfinite_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):select_members(np.ones((2,3),np.float32),['same','same'])
        with self.assertRaises(ValueError):select_members(np.full((2,3),np.nan,np.float32),['a','b'])

    def test_prr_delegates_original_operator_and_protects_foreground(self):
        base=np.zeros((30,30),np.uint8);base[:5,:5]=9
        masks=np.zeros((3,30,30),bool);masks[0,:12,:12]=1;masks[1,12:25,12:25]=1;masks[2,:5,:5]=1
        q=dict(candidate_ids=np.array(['strict','loose','occupied']),class_ids=np.array([18,9,18]),
            retrieval_scores=np.array([.99,.93,.99],np.float32),margins=np.array([.2,.08,.2],np.float32),
            proposal_scores=np.array([.8,.8,.8],np.float32),source_classes=np.array([9,18,9]))
        q['fused_scores']=.8*q['retrieval_scores']+.2*q['proposal_scores']
        a,ar=recover_from_queries(base,masks,q,'synthetic');b,br=compose(base,masks,q,'synthetic')
        self.assertEqual(ar,br)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(a['E'][base!=0],base[base!=0])
        self.assertTrue((a['E'][12:25,12:25]==9).all())

    def test_formal_defaults_are_production_aligned(self):
        from configs.experiment import TRAINING,MAINLINE
        from tools.run_mainline_full import environment
        env=environment(dict(dataset='VOC',seed=42,checkpoint='sam3.pt'),ROOT,1)
        self.assertEqual(MAINLINE['recipe_id'],'lsa_hcgmp_prr_v1')
        self.assertEqual(TRAINING['selection_metric'],'balanced');self.assertEqual(TRAINING['eval_topk'],25)
        self.assertEqual(TRAINING['eval_centroid_weight'],.2);self.assertEqual(TRAINING['num_layers'],2)
        self.assertEqual(env['OVERRIDE_FILTER_FOREGROUND_DROP_RATIO'],'0')
        self.assertEqual(env['OVERRIDE_MIN_MASK_AREA'],'100')
        recipe=json.loads((ROOT/MAINLINE['recipe']).read_text())
        self.assertEqual(recipe['prr']['stages'],['HCR','Strict R3','Loose E'])
        self.assertEqual(recipe['bank']['postfilter_percent'],5)

if __name__=='__main__':unittest.main()
