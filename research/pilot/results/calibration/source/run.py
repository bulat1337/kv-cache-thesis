"""Run one pinned pilot cycle; retain all answers, phase times and selected rows."""
import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load

from backend import (Pool, assemble, cache_arrays, cache_pairs, finish, generate, key,
                     native, select, sparse)
from metrics import score
from prepare_data import digest, prompt_parts

ROOT = Path(__file__).resolve().parent


def logit_difference(a, b):
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return {"max_abs": float(np.abs(d).max()), "rms": float(np.sqrt((d * d).mean())),
            "argmax_equal": int(mx.argmax(a).item()) == int(mx.argmax(b).item())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['calibration', 'test'], required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    out = ROOT / 'results' / args.split
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'rows.jsonl').exists():
        raise RuntimeError(f'Results already exist: {out}; preserve or rename before rerunning')
    config = json.loads((ROOT / 'config.json').read_text())
    examples = [json.loads(s) for s in (ROOT / 'data/examples.jsonl').read_text().splitlines()]
    examples = [e for e in examples if e['split'] == args.split][:args.limit]
    begin = time.perf_counter()
    model, tokenizer = load(str(ROOT / 'weights'))
    model.update(tree_map(lambda x: x.astype(mx.float16), model.parameters()))
    model.eval()
    finish(*[v for _, v in tree_flatten(model.parameters())])
    load_s = time.perf_counter() - begin
    parameter_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    manifest = dict(date='2026-11-07', timezone='Europe/Moscow', python=platform.python_version(),
                    platform=platform.platform(), device=mx.device_info(), config=config,
                    config_sha256=digest(ROOT / 'config.json'), examples_sha256=digest(ROOT / 'data/examples.jsonl'),
                    load_s=load_s, parameter_bytes=parameter_bytes, split=args.split,
                    files={str(p.relative_to(ROOT)): digest(p) for p in
                           [ROOT / 'backend.py', ROOT / 'run.py', ROOT / 'metrics.py', ROOT / 'requirements.lock.txt']})
    (out / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pool, prefix_pool = Pool(model), Pool(model)
    prefix_history = []
    # Warm both code paths; discard preparation so it does not enter workload cost.
    warm = examples[0]['prompt']
    warm_pool = Pool(model)
    native(model, warm['ids'])
    sparse(model, warm['ids'], assemble(model, warm['segments'], warm_pool), select(warm['segments'], 'lego'))
    del warm_pool
    methods = ['full', 'no_documents', 'prefix', 'prefix_warm', 'repair_all', 'independent', 'lego4', 'lego16',
               'random4-s11', 'random4-s29', 'random4-s47', 'query4']
    try:
        for case_no, example in enumerate(examples):
            prompt = example['prompt']
            ids, segments = prompt['ids'], prompt['segments']
            assert ids == sum(segments, [])
            full_logits, full_cache = native(model, ids)
            del full_cache
            # Maximal exact token prefix from completed earlier requests.
            prefix_ids = segments[0]
            prefix_pairs = prefix_pool.get(prefix_ids, 'instruction')
            for old_ids, old_pairs in prefix_history:
                common = 0
                for a, b in zip(ids, old_ids):
                    if a != b:
                        break
                    common += 1
                if common > len(prefix_ids):
                    prefix_ids = ids[:common]
                    prefix_pairs = [(k[..., :common, :], v[..., :common, :]) for k, v in old_pairs]
            prefix_logits, prefix_cache = native(model, ids[len(prefix_ids):], prefix_pairs)
            document_end = len(ids) - len(segments[-1])
            warm_pairs = [(k[..., :document_end, :], v[..., :document_end, :])
                          for k, v in cache_pairs(prefix_cache)]
            finish(*(x for p in warm_pairs for x in p))
            del prefix_cache
            prefix_diff = logit_difference(full_logits, prefix_logits)
            pairs = assemble(model, segments, pool)
            repair_logits, repair_cache, _ = sparse(model, ids, pairs, select(segments, 'all'))
            del repair_cache
            repair_diff = logit_difference(full_logits, repair_logits)
            for label, diff in [('prefix', prefix_diff), ('all_repair', repair_diff)]:
                diff['within_tolerance'] = (diff['max_abs'] <= config['logit_absolute_tolerance']
                                            and diff['rms'] <= config['logit_rms_tolerance'])
                if args.split == 'test':
                    assert diff['within_tolerance'], (example['id'], label, diff)
            checks = dict(id=example['id'], token_ids_sha256=key(ids), tokens=len(ids),
                          document_bounds=prompt['document_bounds'], prefix=prefix_diff,
                          all_repair=repair_diff,
                          fragments=[dict(key=key(s), source_start=0, source_end=len(s),
                                          target_start=sum(map(len, segments[:i])),
                                          target_end=sum(map(len, segments[:i + 1])))
                                     for i, s in enumerate(segments[:-1])],
                          exact_available_prefix_tokens=len(prefix_ids),
                          warm_prefix_tokens=document_end,
                          mask='key_position <= absolute_query_position, including self')
            with open(out / 'checks.jsonl', 'a') as f:
                f.write(json.dumps(checks) + '\n')
            del pairs, repair_logits, prefix_logits
            # Rotate timing order across cases. Each method has three completed prefills.
            order = methods[case_no % len(methods):] + methods[:case_no % len(methods)]
            for method in order:
                samples, last = [], None
                for repetition in range(config['timing_repetitions']):
                    mx.synchronize()
                    active_before = mx.get_active_memory()
                    mx.reset_peak_memory()
                    t0 = time.perf_counter()
                    selection_s = assembly_s = 0.0
                    selection_positions = []
                    if method == 'full':
                        positions = list(range(len(ids)))
                        logits, cache = native(model, ids)
                    elif method == 'no_documents':
                        # Explicit ablation; token identity requirement applies to all other modes.
                        empty = dict(example, documents=[])
                        content = 'Use the documents to answer the question. Return only the short answer, without explanation.\n\nQuestion: ' + example['question'] + '\nAnswer:'
                        no_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                              tokenize=True, add_generation_prompt=True, enable_thinking=False)
                        positions = list(range(len(no_ids)))
                        logits, cache = native(model, no_ids)
                    elif method == 'prefix':
                        positions = list(range(len(prefix_ids), len(ids)))
                        logits, cache = native(model, ids[len(prefix_ids):], prefix_pairs)
                    elif method == 'prefix_warm':
                        positions = list(range(document_end, len(ids)))
                        logits, cache = native(model, ids[document_end:], warm_pairs)
                    else:
                        a0 = time.perf_counter()
                        pairs = assemble(model, segments, pool)
                        assembly_s = time.perf_counter() - a0
                        s0 = time.perf_counter()
                        if method == 'repair_all':
                            positions = select(segments, 'all')
                        elif method == 'query4':
                            selection_positions = select(segments, 'none')
                            _, selector_cache, scores = sparse(model, ids, pairs, selection_positions, collect_scores=True)
                            del selector_cache
                            positions = select(segments, 'query', k=4, scores=scores.tolist())
                        elif method.startswith('random'):
                            positions = select(segments, 'random', k=4, seed=int(method.split('-s')[1]))
                        elif method.startswith('lego'):
                            positions = select(segments, 'lego', k=int(method[4:]))
                        else:
                            positions = select(segments, 'none')
                        selection_s = time.perf_counter() - s0
                        logits, cache, _ = sparse(model, ids, pairs, positions)
                        del pairs
                    first_token = int(mx.argmax(logits).item())
                    mx.synchronize()
                    elapsed = time.perf_counter() - t0
                    samples.append(dict(ttft_s=elapsed, selection_s=selection_s, assembly_s=assembly_s,
                                        peak_active_bytes=mx.get_peak_memory(), active_before_bytes=active_before,
                                        peak_increment_bytes=max(0, mx.get_peak_memory() - active_before),
                                        allocator_cache_bytes=mx.get_cache_memory()))
                    if repetition < config['timing_repetitions'] - 1:
                        del logits, cache
                generated = generate(model, tokenizer, logits, cache, config['max_new_tokens'])
                quality = score(generated['answer'], example['answer'])
                row = dict(id=example['id'], group=example['group'], dataset=example['dataset'],
                           split=args.split, hops=example['hops'], layout=example['layout'],
                           method=method, gold=example['answer'], quality=quality,
                           prompt_tokens=len(no_ids) if method == 'no_documents' else len(ids),
                           token_ids_sha256=key(no_ids) if method == 'no_documents' else key(ids),
                           computed_positions_per_layer=[positions] * len(model.layers),
                           selector_positions_per_layer=[selection_positions] * len(model.layers),
                           computed_layer_tokens=len(positions) * len(model.layers),
                           selector_layer_tokens=len(selection_positions) * len(model.layers),
                           ttft_s=statistics.median(s['ttft_s'] for s in samples), timing_samples=samples,
                           online_decode_s=generated['decode_s'],
                           complete_s=statistics.median(s['ttft_s'] for s in samples) + generated['decode_s'],
                           decode_layer_tokens=len(generated['output_ids']) * len(model.layers),
                           decode_peak_active_bytes=mx.get_peak_memory(),
                           process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           pool_bytes=sum(r['bytes'] for r in pool.log),
                           exact_prefix_pool_bytes=sum(x.nbytes for _, pairs in prefix_history for p in pairs for x in p),
                           exact_available_prefix_tokens=len(prefix_ids), **generated)
                with open(out / 'rows.jsonl', 'a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"{args.split} {case_no + 1}/{len(examples)} {example['id']} {method}: "
                      f"EM={quality['em']:.0f} TTFT={row['ttft_s']:.3f}s {generated['answer']!r}", flush=True)
                del logits, cache
            prefix_history.append((ids[:document_end], warm_pairs))
            del full_logits
    finally:
        (out / 'preparation.json').write_text(json.dumps(dict(independent=pool.log, exact_prefix=prefix_pool.log), indent=2) + '\n')
    print('DONE', args.split, flush=True)


if __name__ == '__main__':
    main()
