import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class MainlinePromotionTests(unittest.TestCase):
    def test_default_recipe_and_explicit_ablation_switches(self):
        from tools.run_mainline_full import parse_args
        from tools.generate_pseudo_masks_sam3 import parse_args as pseudo_args
        a=parse_args([])
        self.assertTrue(a.edge_repair and a.force_presence_one and a.semantic_cleanup and a.save_image_features)
        a=parse_args(['--no-edge-repair','--no-force-presence-one','--no-semantic-cleanup','--no-save-image-features'])
        self.assertFalse(a.edge_repair or a.force_presence_one or a.semantic_cleanup or a.save_image_features)
        p=pseudo_args([]);self.assertTrue(p.edge_repair and p.force_presence_one)
        self.assertFalse(pseudo_args(['--no-force-presence-one']).force_presence_one)

    def test_standalone_cleanup_matches_validated_semantic_algorithm_and_resume_guards(self):
        from tools.clean_pseudo_labels import main
        from src.pseudo_label_cleanup import clean_semantic_labels
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()),patch('tools.run_mainline_full.free_space'):
            root=Path(directory);source=root/'raw';source.mkdir()
            labels=np.zeros((100,100),np.uint8);labels[20:80,20:80]=1;labels[30:32,30:32]=0;labels[1:3,1:3]=1
            Image.fromarray(labels).save(source/'a.png');original=(source/'a.png').read_bytes()
            listing=root/'ids.txt';listing.write_text('a\n')
            args=['--input-dir',str(source),'--output-dir',str(root/'clean'),'--image-list',str(listing)]
            self.assertEqual(main(args),0);self.assertEqual(main(args),0)
            expected,_=clean_semantic_labels(labels)
            with Image.open(root/'clean/a.png') as image:np.testing.assert_array_equal(np.asarray(image),expected)
            self.assertEqual((source/'a.png').read_bytes(),original)
            with self.assertRaises(ValueError):main(args+['--min-island-area','65'])
            Image.fromarray(np.zeros_like(labels)).save(source/'a.png')
            with self.assertRaises(ValueError):main(args)

    def test_cleanup_prevents_source_overwrite(self):
        from tools.clean_pseudo_labels import main
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stderr(io.StringIO()):
            root=Path(directory);listing=root/'ids.txt';listing.write_text('a\n')
            with self.assertRaises(SystemExit):main(['--input-dir',directory,'--output-dir',directory,'--image-list',str(listing)])

    def make_selection(self,root):
        from configs.experiment import MAINLINE
        from tools.run_mainline_full import identity
        artifact=root/'adapter.pt';artifact.write_bytes(b'weights')
        selector=root/'mainline_active.json'
        selector.write_text(json.dumps(dict(recipe_id=MAINLINE['recipe_id'],status='active',seed=42,
            paths=dict(bank='bank',adapter='adapter.pt',checkpoint='sam3.pt',dataset_root='dataset',
                clean_proposals_dir='clean',raw_proposals_dir='raw'),artifact_bindings=[identity(artifact)])))
        return types.SimpleNamespace(mainline_manifest=selector,seed=None,bank=None,adapter=None,checkpoint=None,
            dataset_root=None,clean_proposals_dir=None,raw_proposals_dir=None)

    def test_activation_resolves_artifacts_and_rejects_changed_weights_or_wrong_seed(self):
        from project_utils.mainline_selection import resolve_mainline_inputs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.make_selection(root);resolve_mainline_inputs(args,root)
            self.assertEqual(args.adapter,root/'adapter.pt');self.assertEqual(args.seed,42)
            args=self.make_selection(root);args.seed=123
            with self.assertRaises(ValueError):resolve_mainline_inputs(args,root)
            args=self.make_selection(root);(root/'adapter.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):resolve_mainline_inputs(args,root)

    def test_fresh_clone_requires_explicit_inputs_when_no_activation_exists(self):
        from project_utils.mainline_selection import resolve_mainline_inputs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.make_selection(root);args.mainline_manifest.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):resolve_mainline_inputs(args,root)

    def test_activation_rejects_legacy_or_changed_recipes(self):
        from tools.activate_mainline import checked_recipe
        from configs.experiment import PSEUDO_CLEANUP
        from src.pseudo_label_cleanup import POLICY
        generation=dict(pseudo={},pseudo_extra_cli=['--edge-repair','--force-presence-one','--mask-logit-threshold','0.4'])
        manifest=dict(source_version='edge',cleanup_policy=POLICY,label_cleanup=PSEUDO_CLEANUP)
        checked_recipe(manifest,generation)
        for changed in [dict(generation,pseudo_extra_cli=['--force-presence-one']),
                        dict(generation,pseudo_extra_cli=['--edge-repair','--force-presence-one','--mask-logit-threshold','0.25'])]:
            with self.assertRaises(ValueError):checked_recipe(manifest,changed)
        with self.assertRaises(ValueError):checked_recipe(dict(semantic_cleanup=False),generation)


if __name__=='__main__':unittest.main()
