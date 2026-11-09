"""Qwen3 sparse prefill on MLX. All omitted token rows skip QKV, MLP and attention queries."""
import hashlib
import json
import random
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache


def finish(*arrays):
    mx.eval(*arrays)
    mx.synchronize()


def cache_arrays(cache):
    return [v for c in cache for v in (c.keys, c.values) if v is not None]


def cache_from_pairs(pairs):
    out = []
    for k, v in pairs:
        c = KVCache()
        c.keys, c.values, c.offset = k, v, k.shape[2]
        out.append(c)
    return out


def cache_pairs(cache):
    return [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in cache]


def head(model, hidden):
    hidden = hidden[:, -1:, :]
    return (model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings
            else model.lm_head(hidden))[:, -1, :]


def native(model, ids, pairs=None):
    cache = cache_from_pairs(pairs) if pairs is not None else [KVCache() for _ in model.layers]
    hidden = model.model(mx.array([ids]), cache=cache)
    logits = head(model, hidden)
    finish(logits, *cache_arrays(cache))
    return logits, cache


def key(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def shift_keys(attention, keys, delta):
    # A positional transfer adds a CONSTANT angle to every row. A normal
    # sequence RoPE would add the local row index a second time.
    return attention.rope(keys.transpose(2, 1, 0, 3), offset=delta).transpose(2, 1, 0, 3)


class Pool:
    def __init__(self, model):
        self.model, self.items, self.log = model, {}, []

    def get(self, ids, kind="document"):
        tag = key(ids)
        if tag not in self.items:
            begin = time.perf_counter()
            _, cache = native(self.model, ids)
            pairs = cache_pairs(cache)
            finish(*(x for p in pairs for x in p))
            self.items[tag] = pairs
            self.log.append(dict(key=tag, kind=kind, tokens=len(ids),
                                 seconds=time.perf_counter() - begin,
                                 bytes=sum(x.nbytes for p in pairs for x in p),
                                 source_positions=[0, len(ids)], layers=len(pairs)))
        return self.items[tag]


def assemble(model, segments, pool):
    """Prefix and independent documents start at zero; shift cached post-RoPE K."""
    blocks, offset = [], 0
    for i, ids in enumerate(segments[:-1]):
        pairs = pool.get(ids, "instruction" if i == 0 else "document")
        shifted = []
        for layer, (k, v) in zip(model.layers, pairs):
            shifted.append((shift_keys(layer.self_attn, k, offset) if offset else k, v))
        blocks.append(shifted)
        offset += len(ids)
    suffix_length = len(segments[-1])
    pairs = []
    for layer_no in range(len(model.layers)):
        k0, v0 = blocks[0][layer_no]
        shape = (1, k0.shape[1], suffix_length, k0.shape[-1])
        k = mx.concatenate([b[layer_no][0] for b in blocks] + [mx.zeros(shape, k0.dtype)], axis=2)
        v = mx.concatenate([b[layer_no][1] for b in blocks] + [mx.zeros(shape, v0.dtype)], axis=2)
        pairs.append((k, v))
    finish(*(x for p in pairs for x in p))
    return pairs


def sparse(model, ids, pairs, positions, collect_scores=False):
    """Compute selected hidden rows, scatter their KV at each layer, causal attention."""
    assert positions == sorted(set(positions)) and positions[-1] == len(ids) - 1
    p = mx.array(positions)
    h = model.model.embed_tokens(mx.array([[ids[i] for i in positions]]))
    mask = mx.arange(len(ids))[None, :] <= p[:, None]
    result_pairs, scores = [], mx.zeros((len(ids),), mx.float32)
    for layer, (cached_k, cached_v) in zip(model.layers, pairs):
        a = layer.self_attn
        z = layer.input_layernorm(h)
        n = len(positions)
        q = a.q_norm(a.q_proj(z).reshape(1, n, a.n_heads, -1)).transpose(0, 2, 1, 3)
        k = a.k_norm(a.k_proj(z).reshape(1, n, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = a.v_proj(z).reshape(1, n, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
        # Each selected row becomes a length-one sequence with its own absolute offset.
        if positions == list(range(positions[0], positions[-1] + 1)):
            q, k = a.rope(q, offset=positions[0]), a.rope(k, offset=positions[0])
        else:
            q = a.rope(q.transpose(2, 1, 0, 3), offset=p).transpose(2, 1, 0, 3)
            k = a.rope(k.transpose(2, 1, 0, 3), offset=p).transpose(2, 1, 0, 3)
        # MLX updates have copy-on-write semantics; pool arrays remain immutable.
        new_k, new_v = mx.array(cached_k), mx.array(cached_v)
        new_k[:, :, p, :] = k
        new_v[:, :, p, :] = v
        output = mx.fast.scaled_dot_product_attention(q, new_k, new_v, scale=a.scale, mask=mask)
        h = h + a.o_proj(output.transpose(0, 2, 1, 3).reshape(1, n, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        if collect_scores:
            expanded_k = mx.repeat(new_k, a.n_heads // a.n_kv_heads, axis=1)
            weights = (q.astype(mx.float32) @ expanded_k.astype(mx.float32).swapaxes(-1, -2)) * a.scale
            weights = mx.where(mask, weights, -float("inf"))
            scores = scores + mx.softmax(weights, axis=-1).mean(axis=(0, 1, 2))
        result_pairs.append((new_k, new_v))
        # Bound lazy graphs and guarantee all KV rows and hidden rows were computed.
        mx.eval(h, new_k, new_v, scores)
    logits = head(model, model.model.norm(h))
    cache = cache_from_pairs(result_pairs)
    finish(logits, *cache_arrays(cache), scores)
    return logits, cache, scores


def select(segments, method, k=4, seed=11, scores=None):
    offset, selected = len(segments[0]), []
    rng = random.Random(seed)
    for doc in segments[1:-1]:
        size = min(k, len(doc))
        if method == "lego":
            local = list(range(size))
        elif method == "random":
            local = sorted(rng.sample(range(len(doc)), size))
        elif method == "query":
            # Deterministic tie breaking; only the current question's attention is used.
            local = sorted(sorted(range(len(doc)), key=lambda i: (-scores[offset + i], i))[:size])
        elif method == "all":
            local = list(range(len(doc)))
        elif method == "none":
            local = []
        else:
            raise ValueError(method)
        selected.extend(offset + i for i in local)
        offset += len(doc)
    selected.extend(range(offset, offset + len(segments[-1])))
    return selected


def generate(model, tokenizer, logits, cache, max_new_tokens=48):
    begin = time.perf_counter()
    output, stopped = [], False
    eos = set(tokenizer.eos_token_ids)
    for _ in range(max_new_tokens):
        token = int(mx.argmax(logits, axis=-1).item())
        if token in eos:
            stopped = True
            break
        output.append(token)
        hidden = model.model(mx.array([[token]]), cache=cache)
        logits = head(model, hidden)
        finish(logits, *cache_arrays(cache))
    return dict(answer=tokenizer.decode(output).strip(), output_ids=output,
                stopped=stopped, decode_s=time.perf_counter() - begin)
