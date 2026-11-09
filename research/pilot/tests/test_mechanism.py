"""Mechanism tests against stock MLX-LM on a deterministic tiny Qwen3."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3 import Model, ModelArgs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import Pool, assemble, cache_pairs, finish, native, select, shift_keys, sparse
from metrics import score


class Mechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(100)
        args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=3,
                         intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
                         vocab_size=80, num_key_value_heads=2, max_position_embeddings=512,
                         head_dim=8, tie_word_embeddings=True, rope_theta=10000)
        cls.model = Model(args)
        cls.segments = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12]]
        cls.ids = sum(cls.segments, [])
        finish(*[v for _, v in __import__('mlx').utils.tree_flatten(cls.model.parameters())])

    def test_prefix_and_full_sparse(self):
        model, ids = self.model, self.ids
        full, _ = native(model, ids)
        _, pc = native(model, ids[:7])
        prefix, _ = native(model, ids[7:], cache_pairs(pc))
        np.testing.assert_allclose(np.array(full), np.array(prefix), atol=2e-5, rtol=0)
        pool = Pool(model)
        pairs = assemble(model, self.segments, pool)
        snapshot = [np.array(k) for k, v in pairs]
        computed, cache, _ = sparse(model, ids, pairs, select(self.segments, "all"))
        np.testing.assert_allclose(np.array(full), np.array(computed), atol=2e-5, rtol=0)
        for before, (k, v) in zip(snapshot, pairs):
            np.testing.assert_array_equal(before, np.array(k))

    def test_causal_mask(self):
        pool = Pool(self.model)
        pairs = assemble(self.model, self.segments, pool)
        positions = select(self.segments, "all")
        # Compare a query at an interior position by ending both prompts there.
        for end in [8, 10, 12]:
            clipped_pairs = [(k[..., :end, :], v[..., :end, :]) for k, v in pairs]
            sparse_logits, _, _ = sparse(self.model, self.ids[:end], clipped_pairs,
                                         list(range(3, end)))
            full, _ = native(self.model, self.ids[:end])
            np.testing.assert_allclose(np.array(full), np.array(sparse_logits), atol=2e-5, rtol=0)

    def test_rope_shift(self):
        a = self.model.layers[0].self_attn
        x = mx.random.normal((1, 2, 5, 8))
        shifted = shift_keys(a, a.rope(x), 19)
        direct = a.rope(x, offset=19)
        finish(shifted, direct)
        np.testing.assert_allclose(np.array(shifted), np.array(direct), atol=2e-6, rtol=0)

    def test_first_layer_transfer_and_pool_immutability(self):
        pool = Pool(self.model)
        pairs = assemble(self.model, self.segments, pool)
        _, full_cache = native(self.model, self.ids)
        end = len(self.ids) - len(self.segments[-1])
        np.testing.assert_allclose(np.array(pairs[0][0][..., :end, :]),
                                   np.array(full_cache[0].keys[..., :end, :]), atol=2e-6, rtol=0)
        np.testing.assert_allclose(np.array(pairs[0][1][..., :end, :]),
                                   np.array(full_cache[0].values[..., :end, :]), atol=2e-6, rtol=0)
        before = [np.array(k) for p in pool.items.values() for k, v in p]
        sparse(self.model, self.ids, pairs, select(self.segments, 'lego', k=2))
        after = [np.array(k) for p in pool.items.values() for k, v in p]
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_actual_budget(self):
        selected = select(self.segments, "lego", k=2)
        random_selected = select(self.segments, "random", k=2)
        self.assertEqual(len(selected), len(random_selected))
        self.assertLess(len(selected), len(self.ids))
        self.assertEqual(selected[-2:], [10, 11])

    def test_metrics_against_upstream(self):
        path = ROOT / 'sources/hotpot_evaluate_v1.py'
        spec = importlib.util.spec_from_file_location('upstream_metrics', path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        for prediction, answer in [('The Oak-Ford.', 'OakFord'), ('no, thanks', 'no'),
                                    ('yes', 'yes'), ('red red blue', 'red blue blue'),
                                    ('', ''), ('T100', 'T100')]:
            actual = score(prediction, answer)
            self.assertEqual(actual['em'], float(upstream.exact_match_score(prediction, answer)))
            self.assertAlmostEqual(actual['f1'], upstream.f1_score(prediction, answer)[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
