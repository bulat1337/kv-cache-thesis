"""Independent result audit, not a second inference run."""
import importlib.util
import json
from pathlib import Path

from prepare_data import digest

ROOT = Path(__file__).resolve().parent


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    rows = load(ROOT / 'results/test/rows.jsonl')
    examples = load(ROOT / 'data/examples.jsonl')
    spec = importlib.util.spec_from_file_location('upstream_scorer', ROOT / 'sources/hotpot_evaluate_v1.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    manifest = json.loads((ROOT / 'results/test/run-manifest.json').read_text())
    findings = []
    for name, expected in manifest['files'].items():
        assert digest(ROOT / name) == expected, name
    assert manifest['config_sha256'] == digest(ROOT / 'config.json')
    assert manifest['examples_sha256'] == digest(ROOT / 'data/examples.jsonl')
    findings.append('Execution code, lock, config and dataset match hashes fixed at run start')
    calibration = json.loads((ROOT / 'results/calibration/run-manifest.json').read_text())
    assert digest(ROOT / 'results/calibration/source/run.py') == calibration['files']['run.py']
    assert digest(ROOT / 'results/calibration/source/config.json') == calibration['config_sha256']
    findings.append('Archived calibration source and configuration match original recorded hashes')
    groups = {s: {e['group'] for e in examples if e['split'] == s} for s in ['calibration', 'test']}
    assert not groups['calibration'] & groups['test']
    findings.append('Calibration/test groups disjoint; no filtering by model outputs')
    test = {e['id']: e for e in examples if e['split'] == 'test'}
    assert len(rows) == len(test) * 12 == 240
    for r in rows:
        assert r['gold'] == test[r['id']]['answer']
        assert r['quality']['em'] == float(scorer.exact_match_score(r['answer'], r['gold']))
        assert abs(r['quality']['f1'] - scorer.f1_score(r['answer'], r['gold'])[0]) < 1e-12
        assert len(r['output_ids']) <= 48
        for positions in r['computed_positions_per_layer']:
            assert positions == sorted(set(positions))
        assert len(r['timing_samples']) == 3
        assert r['computed_layer_tokens'] == sum(len(x) for x in r['computed_positions_per_layer'])
        assert r['selector_layer_tokens'] == sum(len(x) for x in r['selector_positions_per_layer'])
    findings.append('All 240 raw answers independently rescored by original HotpotQA evaluator; budgets and repetitions verified')
    full = {r['id']: r for r in rows if r['method'] == 'full'}
    for dataset, by_method in summary['summaries'].items():
        for method, aggregate in by_method.items():
            selected = [r for r in rows if r['method'] == method and (dataset == 'all' or r['dataset'] == dataset)]
            expected = sum(scorer.exact_match_score(r['answer'], r['gold']) for r in selected)
            assert aggregate['correct'] == expected
            lost = sum(full[r['id']]['quality']['em'] == 1 and r['quality']['em'] == 0 for r in selected)
            assert aggregate['lost'] == lost
            assert aggregate['full_correct'] == sum(full[r['id']]['quality']['em'] for r in selected)
    findings.append('All aggregate numerators and full-correct denominators reconcile independently to raw rows')
    checks = load(ROOT / 'results/test/checks.jsonl')
    assert len(checks) == 20
    assert all(c[x]['within_tolerance'] for c in checks for x in ['prefix', 'all_repair'])
    findings.append('Numerical controls present and within pre-test calibrated tolerance for all 20 test prompts')
    for r in rows:
        if r['method'] != 'no_documents':
            assert r['token_ids_sha256'] == full[r['id']]['token_ids_sha256']
    findings.append('Token identity holds in all compared cache modes; no-documents ablation is labeled separately')
    for e in examples:
        if e['dataset'] == 'hotpotqa':
            assert len(e['documents']) == 10 and len(e['prompt']['ids']) <= 2048
    findings.append('HotpotQA retains all 10 paragraphs and no token truncation; deterministic short-context selection disclosed')
    output = {'status': 'verified', 'date': '2026-11-07', 'checks': findings,
              'limits': ['6 independent HotpotQA test questions and 2 independent synthetic worlds',
                         'HotpotQA has zero examples correct with full context and wrong without documents; natural-data suitability unresolved',
                         'Qwen3-1.7B only, FP16 greedy, 48 output tokens',
                         'LegoLink and query-aware rule are explicit local adaptations',
                         'strict EM includes answer format changes; inspect F1 and raw answers',
                         'peak active memory includes shared harness pools; not an isolated server footprint',
                         'break-even is a measured-workload scenario, not a production guarantee']}
    (ROOT / 'results/validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
