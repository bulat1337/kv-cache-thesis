"""Fetch pinned public inputs and record their provenance (run inside the venv)."""
import importlib.metadata
import argparse
import inspect
import json
import os
import subprocess
import urllib.request
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import hf_hub_download, snapshot_download
from prepare_data import ROOT, digest


def prepare_evaluator():
    target = ROOT / 'sources/hotpot_evaluate_v1.py'
    expected = 'd35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/hotpotqa/hotpot/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py',
            target)
    if digest(target) != expected:
        raise ValueError('HotpotQA evaluator SHA256 mismatch')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator-only', action='store_true', help='Prepare the pinned answer scorer without downloading weights')
    args = parser.parse_args()
    evaluator = prepare_evaluator()
    if args.evaluator_only:
        print('HotpotQA evaluator ready; SHA256 verified')
        return
    snapshot_download('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                      local_dir=ROOT / 'weights', allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE', 'README.md'])
    for filename in ['distractor/validation-00000-of-00001.parquet', 'README.md']:
        hf_hub_download('hotpotqa/hotpot_qa', filename, repo_type='dataset',
                        revision='1908d6afbbead072334abe2965f91bd2709910ab', local_dir=ROOT / 'data/upstream')
    epic = ROOT / 'sources/epic'
    if not epic.exists():
        subprocess.run(['git', 'clone', 'https://github.com/DerekHJH/epic.git', str(epic)], check=True)
    subprocess.run(['git', '-C', str(epic), 'checkout', '3204410f1723ed0f39575b4eba8c17578a1ee1a1'], check=True)
    import mlx_lm.models.qwen3 as qwen
    license_files = []
    for package in ['mlx', 'mlx-lm']:
        dist = importlib.metadata.distribution(package)
        for path in dist.files:
            if str(path).endswith('/licenses/LICENSE'):
                target = ROOT / 'sources' / (package + '-LICENSE')
                target.write_text(dist.locate_file(path).read_text())
                license_files.append(target)
    code = Path(inspect.getfile(qwen))
    (ROOT / 'sources/mlx_lm_qwen3.py').write_text(code.read_text())
    files = [p for p in (ROOT / 'weights').iterdir() if p.is_file()]
    files += [ROOT / 'data/upstream/distractor/validation-00000-of-00001.parquet',
              ROOT / 'data/upstream/README.md', evaluator, ROOT / 'sources/mlx_lm_qwen3.py',
              epic / 'LICENSE', epic / 'README.md'] + license_files
    files += [p for p in [ROOT / 'sources/epic-paper.pdf', ROOT / 'sources/epic-paper.txt'] if p.exists()]
    manifest = {'date': '2026-11-07', 'timezone': 'Europe/Moscow',
                'model_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', 'model_license': 'Apache-2.0',
                'hotpot_revision': '1908d6afbbead072334abe2965f91bd2709910ab', 'hotpot_license': 'CC BY-SA 4.0',
                'epic_commit': '3204410f1723ed0f39575b4eba8c17578a1ee1a1', 'epic_license': 'Apache-2.0',
                'mlx_lm_version': '0.32.2', 'mlx_lm_tag_commit': '93e6d4a6447b3080861985947f0820e76fcb957d',
                'mlx_lm_license': 'MIT', 'mlx_version': '0.32.4', 'mlx_license': 'MIT',
                'mlx_tag_commit': '36969c094354086e15d2a327a9c50feb20b9cf11',
                'hotpot_evaluator_commit': '3635853403a8735609ee997664e1528f4480762a',
                'files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}}
    (ROOT / 'sources/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Pinned sources ready; manifest saved')


if __name__ == '__main__':
    main()
