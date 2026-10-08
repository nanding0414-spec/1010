"""Exercise the entire launcher with synthetic worker products, without GPU/SAM3."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tools import run_span2016_full as pipeline
from src.feature_adapter import adapted_signature
from configs.experiment import CLEANUP
from configs import config  # Load real configuration before the test changes cwd.


class CleanPipelineTests(unittest.TestCase):
    def test_pipeline_requires_no_historical_experiment_and_resumes(self):
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'data/VOC2012'
            images=dataset/'JPEGImages'
            span=root/'span'
            proposals=root/'proposals'
            for folder in (images,span,proposals,dataset/'SegmentationClassAug'):
                folder.mkdir(parents=True)
            for stem in ('a','b','c','d'):
                Image.new('RGB',(8,8)).save(images/f'{stem}.jpg')
                Image.new('L',(8,8),1).save(dataset/f'SegmentationClassAug/{stem}.png')
            for stem in ('a','b'):
                Image.new('RGB',(32,32)).save(span/f'{stem}.png')
            for stem in ('c','d'):
                np.savez(proposals/f'{stem}.npz',masks=np.ones((1,8,8),bool),scores=[.9])
            split=dataset/'ImageSets/Segmentation'
            split.mkdir(parents=True)
            (split/'trainaug.txt').write_text('a\nb\n')
            (split/'val.txt').write_text('c\nd\n')
            pipeline.write(dataset/'ImageSets/ImageLevel/trainaug_imagelevel.json',
                           {'images':[{'file_name':f'{stem}.jpg','labels':[1]} for stem in ('a','b')]})
            ckpt=root/'sam3.pt'; ckpt.write_bytes(b'fake checkpoint')
            rawsig={'resolution':2016,'feature_grid':[144,144],'embed_dim':4}
            calls=[]

            def worker_products(jobs):
                for command,env,log in jobs:
                    calls.append(command)
                    def arg(flag):
                        return command[command.index(flag)+1]
                    entry=Path(command[2]).name
                    if entry=='run_pseudo_two_gpus.py':
                        from tools.generate_pseudo_masks_sam3 import parse_args
                        parse_args(command[3:])
                        dest=Path(arg('--output-dir')); dest.mkdir(parents=True)
                        for stem in ('a','b'):
                            Image.new('L',(32,32),1).save(dest/f'{stem}.png')
                    elif entry=='run_span2016_full.py':
                        dest=Path(env['OVERRIDE_OUTPUT_ROOT'])
                        pipeline.write(dest/'meta/feature_signature.json',rawsig)
                    elif entry=='train_feature_adapter.py':
                        self.assertEqual(arg('--num-layers'),'3')
                        weights=Path(arg('--output-dir')); weights.mkdir(parents=True)
                        architecture=dict(input_dim=4,hidden_dim=8,output_dim=2,num_layers=3)
                        torch.save({'architecture':architecture},weights/'adapter.pt')
                        pipeline.write(weights/'training_report.json',{'synthetic':True})
                        signature=adapted_signature(rawsig,weights/'adapter.pt',architecture)
                        pipeline.write(Path(arg('--export-bank'))/'meta/feature_signature.json',signature)
                    elif entry=='run_inference.py':
                        self.assertEqual(env['OVERRIDE_MIN_MASK_AREA'],'300')
                        self.assertEqual(env['OVERRIDE_SAM3_RESOLUTION'],'2016')
                        self.assertNotIn('OVERRIDE_BUILD_MASK_EROSION_ITERATIONS',env)
                        shard=int(arg('--shard-index'))
                        matrix=np.zeros((21,21),dtype=int); matrix[1,1]=64
                        result=dict(feature_signature=pipeline.read(Path(env['OVERRIDE_INF_FEATURE_BANK_ROOT'])/'meta/feature_signature.json'),
                            feature_bank=env['OVERRIDE_INF_FEATURE_BANK_ROOT'],cleanup=CLEANUP,
                            proposals_dir=str(proposals),inference_settings={},feature_image_dir=str(images),feature_image_scale=1,
                            per_class_iou={str(i):0 for i in range(21)},confusion_matrix=matrix.tolist(),
                            evaluated_ids=[('c','d')[shard]],proposal_count_before=1,proposal_count_after=1)
                        pipeline.write(Path(arg('--results-json')),result)
                    else:
                        self.fail(f'Unexpected entry: {entry}')

            command=['run_span2016_full.py','--version','test_clean','--train-images',str(span),
                     '--val-images',str(images),'--proposals-dir',str(proposals),'--checkpoint',str(ckpt)]
            try:
                with patch.object(pipeline,'ROOT',root), patch.object(pipeline,'parallel',side_effect=worker_products), \
                     patch.object(pipeline.subprocess,'check_output',return_value='0, 0'),patch.object(sys,'argv',command):
                    pipeline.main()
                    self.assertEqual(len(calls),7)
                    result=pipeline.read(root/'experiments/test_clean/result.json')
                    self.assertEqual(result['evaluated_count'],2)
                    self.assertAlmostEqual(result['mIoU'],1/21)
                    manifest=pipeline.read(root/'experiments/test_clean/manifest.json')
                    self.assertEqual(manifest['recipe']['prompt_backend'],'active')
                    self.assertEqual(manifest['pseudo_output'],'native_x4')
                    with patch.object(sys,'argv',command+['--resume']):
                        pipeline.main()
                    self.assertEqual(len(calls),7)
            finally:
                os.chdir(original)
