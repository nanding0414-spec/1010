import unittest
import os
import json
from pathlib import Path
import shutil
import tempfile
from collections import Counter
import numpy as np
from tools import experimental_multiview_bank as experiment


class SelectionTests(unittest.TestCase):
    def test_counts_background_family_caps_and_repeatability(self):
        real={('a','cat'):[0,1,2],('b','dog'):[0],('a','background'):[0]}
        candidates={('a','cat'):[dict(family='a',cls='cat',view=v,row=i) for i,v in enumerate(['L30','R30','L60','R60'])],
                    ('b','dog'):[dict(family='b',cls='dog',view='L60',row=0)]}
        plan=experiment.select_rows(real,candidates)
        self.assertEqual(plan,experiment.select_rows(real,candidates))
        self.assertEqual(plan['total_added'],2)
        for m,rows in plan['rows'].items():
            self.assertEqual(Counter((r['family'],r['cls']) for r in rows),{('a','cat'):2})
            if m!='duplicate':self.assertEqual(len({r['row'] for r in rows}),2)
        self.assertTrue(all(r['view'] in ('L30','R30') for r in plan['rows']['view30']))

    def test_empty_support_rejected(self):
        with self.assertRaises(ValueError):experiment.select_rows({('a','cat'):[0]}, {})

    def test_summary_incomplete_has_no_mean(self):
        summary=experiment.summarize({'view30_clean_42':dict(miou=.86)})
        self.assertIsNone(summary['groups']['view30_clean']['mean'])


@unittest.skipUnless(os.environ.get('MODUSEG_TEST_REAL_HISTORY')=='1','Actual cached-model projection test requires server history')
class BankTests(unittest.TestCase):
    def test_actual_model_projection_and_matched_export(self):
        from src.feature_bank_storage import load_feature_matrix_npz,save_feature_matrix_npz
        root=experiment.ROOT
        summary=json.loads((root/'experiments/view_aug_20261008_checked/summary.json').read_text())
        trained=summary['results']['view_weighted_42']['training']
        job=json.loads((Path(trained['attempt'])/'job.json').read_text())
        marker=root/'storage/raw_banks/voc/view_aug_20261008_checked/generated_regions/2007_000032_L30/complete.json'
        entry=json.loads(marker.read_text())
        with np.load(entry['file'],allow_pickle=False) as f:label=int(f['labels'][0])
        cls=experiment.views.provenance.evidence.VOC_CLASSES[label]
        with tempfile.TemporaryDirectory(prefix='mv_gallery_test_') as tmp:
            folder=Path(tmp);bank=folder/'real';(bank/'meta').mkdir(parents=True)
            for name in ('class_mapping.json','feature_signature.json'):
                shutil.copy2(Path(trained['bank'])/'meta'/name,bank/'meta'/name)
            original={}
            for name in ('background',cls):
                file=list((Path(trained['bank'])/'features'/name).glob('*.npz'))[0]
                x,ids=load_feature_matrix_npz(file);original[name]=x[:1]
                save_feature_matrix_npz(bank/'features'/name/file.name,x[:1],ids[:1])
            row=dict(family='2007_000032',cls=cls,view='L30',file=entry['file'],row=0,marker=str(marker))
            plan=dict(rule=experiment.RULE,rows={'duplicate':[dict(family='2007_000032',cls=cls,real_row=0)],
                                               'view30':[row],'view_weighted':[row]})
            plan_path=folder/'selection.json';experiment.ab.write(plan_path,plan)
            experiment.build_banks(dict(job,bank=str(bank),weights=trained['weights'],plan_file=str(plan_path),
                plan_sha256=experiment.ab.digest(plan_path),bank_root=str(folder/'built'),attempt_name='attempt_001'))
            for method in experiment.METHODS[1:]:
                new=folder/'built'/method/'attempt_001'
                data,ids=load_feature_matrix_npz(list((new/'features'/cls).glob('*.npz'))[0])
                self.assertEqual(data.shape,(2,256));np.testing.assert_array_equal(data[0],original[cls][0])
                self.assertTrue(np.isfinite(data).all())
                self.assertAlmostEqual(float(np.linalg.norm(data[1])),1,places=5)
                bg,_=load_feature_matrix_npz(list((new/'features/background').glob('*.npz'))[0])
                np.testing.assert_array_equal(bg,original['background'])
                if method=='duplicate':np.testing.assert_array_equal(data[0],data[1])


if __name__=='__main__':unittest.main()
