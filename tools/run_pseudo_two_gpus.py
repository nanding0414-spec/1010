"""VOC pseudo labels: disjoint workers on visible GPUs 0 and 1, then full mIoU.

Accepts the same arguments as generate_pseudo_masks_sam3.py. Output and experiment
directories must be new; partial runs are retained for diagnosis, never deleted.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def partition_records(records):
    stems = [Path(row['file_name']).stem for row in records]
    if len(stems) < 2 or len(set(stems)) != len(stems):
        raise ValueError('Need at least two images with unique output stems')
    return [records[::2], records[1::2]]


def main(argv=None):
    from tools.generate_pseudo_masks_sam3 import parse_args, SAM3PseudoMaskGenerator
    from tools.pseudo_mask_evaluation import find_ground_truth
    import torch
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if (args.dataset != 'voc' or not args.output_dir or not args.experiment_dir
            or not args.imagelevel_json or not args.ckpt):
        raise ValueError('Requires VOC and explicit --output-dir, --experiment-dir, --imagelevel-json, --ckpt')
    if args.eval_only or not args.evaluate or args.limit or args.image_ids or args.metadata_dir:
        raise ValueError('Two-GPU entry processes and evaluates the full JSON; omit subset/eval-only/metadata flags')
    if args.output_dir.exists() or args.experiment_dir.exists():
        raise FileExistsError('Use new output and experiment directories; existing runs are never overwritten')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expose exactly two GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1')
    # Validate all image/reference/GT paths before either expensive worker starts.
    reader = SAM3PseudoMaskGenerator(dataset_type='voc', dataset_root=args.dataset_root,
        imagelevel_json=args.imagelevel_json, image_dir=args.image_dir,
        output_size_reference_dir=args.output_size_reference_dir, load_model=False, device='cpu')
    images, _ = reader.get_image_list_and_labels(args.split)
    for image in images:
        gt, _ = find_ground_truth(image, reader.dataset_root, 'voc', args.split,
                                 reader.image_metadata[str(image)], args.gt_dir, args.gt_format)
        if gt is None:
            raise FileNotFoundError(f'Missing evaluation GT for {image.stem}')
    data = json.loads(args.imagelevel_json.read_text(encoding='utf-8'))
    shards = partition_records(data['images'])
    args.experiment_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', arguments=vars(args).copy(),
                    input_json_sha256=hashlib.sha256(args.imagelevel_json.read_bytes()).hexdigest(),
                    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
                    python=sys.version, torch_version=torch.__version__,
                    total=len(images), shard_counts=[len(s) for s in shards], workers=[])
    manifest_path = args.experiment_dir / 'two_gpu_run.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    processes, streams = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    previous_signal = signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, records in enumerate(shards):
            shard_json = args.experiment_dir / f'images_shard_{index}.json'
            shard_json.write_text(json.dumps(dict(data, images=records)), encoding='utf-8')
            command = [sys.executable, '-u', str(ROOT / 'tools/generate_pseudo_masks_sam3.py'), *argv,
                       '--imagelevel-json', str(shard_json), '--device', f'cuda:{index}',
                       '--experiment-dir', str(args.experiment_dir / f'worker_{index}'),
                       '--metadata-dir', str(args.output_dir / 'meta' / f'shard_{index}'), '--no-evaluate']
            log = args.experiment_dir / f'worker_{index}.log'
            stream = log.open('w', encoding='utf-8')
            streams.append(stream)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            manifest['workers'].append(dict(pid=process.pid, command=command, log=str(log)))
            save_manifest()
            print(f'Worker {index}: {len(records)} images, PID {process.pid}, log={log}', flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A worker failed; see worker logs. Full evaluation was not started.')
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('A worker failed; see worker logs')
        for image in images:
            if not (args.output_dir / f'{image.stem}.png').is_file():
                raise FileNotFoundError(f'Missing generated PNG: {image.stem}')
        command = [sys.executable, '-u', str(ROOT / 'tools/generate_pseudo_masks_sam3.py'), *argv,
                   '--device', 'cpu', '--eval-only', '--evaluate',
                   '--experiment-dir', str(args.experiment_dir / 'evaluation')]
        manifest['evaluation_command'] = command
        save_manifest()
        print('Both workers finished. Evaluating the full split...', flush=True)
        evaluator = subprocess.Popen(command)
        processes.append(evaluator)
        if evaluator.wait() != 0:
            raise RuntimeError('Full evaluation failed; inspect its report')
        manifest['status'] = 'complete'
        print(f'Done. Metrics: {args.experiment_dir / "evaluation" / ("miou_" + args.split + ".json")}', flush=True)
        return 0
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()
        save_manifest()
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
