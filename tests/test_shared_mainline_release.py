import importlib.util
from pathlib import Path
import tempfile,unittest,contextlib,io
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_shared_entry',ROOT/'tools/run_shared_mainline.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
class SharedCLIContracts(unittest.TestCase):
 def base(self,phase,out):return [phase,'--checkpoint','sam3.pt','--dataset-root','VOC2012','--image-list','ids.txt','--output-dir',str(out)]
 def test_build_requires_image_level_metadata(self):
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):entry.parse_args(self.base('build',Path(d)/'new'))
 def test_predict_requires_matched_model_and_bank(self):
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):entry.parse_args(self.base('predict',Path(d)/'new'))
 def test_existing_results_are_protected(self):
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):entry.parse_args(self.base('predict',d)+['--bank','bank','--adapter','adapter.pt'])
 def test_default_gpu_and_seed(self):
  with tempfile.TemporaryDirectory() as d:
   a=entry.parse_args(self.base('predict',Path(d)/'new')+['--bank','bank','--adapter','adapter.pt']);self.assertEqual(a.gpu,1);self.assertEqual(a.seed,42);self.assertFalse((Path(d)/'new').exists())
 def test_slurm_gpu_assignment_is_preserved(self):
  with patch.dict(entry.os.environ,{'SLURM_JOB_ID':'test','CUDA_VISIBLE_DEVICES':'GPU-allocated'},clear=True):self.assertEqual(entry.visible_gpu(SimpleNamespace(gpu=1)),'GPU-allocated')
 def test_slurm_without_allocated_gpu_rejected(self):
  with patch.dict(entry.os.environ,{'SLURM_JOB_ID':'test'},clear=True):
   with self.assertRaises(RuntimeError):entry.visible_gpu(SimpleNamespace(gpu=1))
if __name__=='__main__':unittest.main()
