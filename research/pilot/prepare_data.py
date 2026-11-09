"""Deterministic pilot selection; no model answers used for selection."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_parts(tokenizer, example):
    instruction = "Use the documents to answer the question. Return only the short answer, without explanation.\n\n"
    docs = ["<document>\n" + d["title"] + "\n" + d["text"] + "\n</document>\n\n" for d in example["documents"]]
    question = "Question: " + example["question"] + "\nAnswer:"
    content = instruction + "".join(docs) + question
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    bounds, char_at = [], text.index(instruction) + len(instruction)
    starts = {a: i for i, (a, b) in enumerate(offsets) if b > a}
    for doc in docs:
        assert text[char_at:char_at + len(doc)] == doc
        if char_at not in starts:
            raise ValueError(f"Non-token document boundary: {char_at}")
        bounds.append(starts[char_at])
        char_at += len(doc)
    if char_at not in starts:
        raise ValueError("Non-token question boundary")
    bounds.append(starts[char_at])
    segments = [ids[:bounds[0]]] + [ids[a:b] for a, b in zip(bounds, bounds[1:])] + [ids[bounds[-1]:]]
    assert [v for part in segments for v in part] == ids
    return {"text": text, "ids": ids, "segments": segments, "document_bounds": bounds,
            "question_char_start": char_at, "tokenization": "one full prompt, offset-verified cuts"}


def synthetic():
    examples = []
    cities = ["Riverton", "Oakford", "Pinehaven"]
    for world in range(3):
        base = world * 100
        # Four documents with identical facts for all questions in a world.
        facts = [f"Project P{base + j:03d} is managed by team T{base + j:03d}." for j in range(8)]
        teams = [f"Team T{base + j:03d} uses office O{base + j:03d}." for j in range(8)]
        offices = [f"Office O{base + j:03d} is in {cities[(world + j) % 3]}." for j in range(8)]
        noise = [f"Archive A{base + j:03d} has shelf S{base + j:03d}." for j in range(8)]
        docs = [{"title": name, "text": " ".join(lines)} for name, lines in
                zip(["Projects", "Teams", "Offices", "Archives"], [facts, teams, offices, noise])]
        split = "calibration" if world == 0 else "test"
        for hops, question, answer in [
            (1, f"Which team manages project P{base:03d}?", f"T{base:03d}"),
            (2, f"Which office does the team managing project P{base:03d} use?", f"O{base:03d}"),
            (3, f"In which city is the office used by the team managing project P{base:03d}?", cities[world])]:
            examples.append(dict(id=f"synth-{world}-{hops}-cross", dataset="synthetic", split=split,
                                 group=f"world-{world}", hops=hops, layout="cross", documents=docs,
                                 question=question, answer=answer))
        # Same three-hop question and facts, grouped across new boundaries.
        grouped = [{"title": f"Records {i}", "text": " ".join(
            [line for j in range(2 * i, 2 * i + 2) for line in [facts[j], teams[j], offices[j], noise[j]]])}
            for i in range(4)]
        examples.append(dict(id=f"synth-{world}-3-within", dataset="synthetic", split=split,
                             group=f"world-{world}", hops=3, layout="within", documents=grouped,
                             question=f"In which city is the office used by the team managing project P{base:03d}?",
                             answer=cities[world]))
    return examples


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights")
    parquet = ROOT / "data/upstream/distractor/validation-00000-of-00001.parquet"
    records = pq.read_table(parquet).to_pylist()
    records.sort(key=lambda r: hashlib.sha256(("pilot-v1:" + r["id"]).encode()).hexdigest())
    examples = synthetic()
    eligible, selected_ids = 0, []
    for r in records:
        context = r["context"]
        docs = [{"title": title, "text": "".join(sentences)} for title, sentences in
                zip(context["title"], context["sentences"])]
        e = dict(id=r["id"], dataset="hotpotqa", group=r["id"], hops=None, layout="original",
                 split="calibration" if len(selected_ids) < 2 else "test", documents=docs,
                 question=r["question"], answer=r["answer"], supporting_facts=r["supporting_facts"],
                 type=r["type"], level=r["level"])
        packed = prompt_parts(tokenizer, e)
        rotated = dict(e, id=r["id"] + "-rotate", documents=docs[1:] + docs[:1], layout="rotate")
        if max(len(packed["ids"]), len(prompt_parts(tokenizer, rotated)["ids"])) > 2048:
            continue
        eligible += 1
        if len(selected_ids) < 8:
            examples.extend([e, rotated])
            selected_ids.append(r["id"])
    for e in examples:
        e["prompt"] = prompt_parts(tokenizer, e)
    output = ROOT / "data/examples.jsonl"
    output.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    manifest = {"date": "2026-11-07", "dataset": "HotpotQA distractor validation",
                "homepage": "https://hotpotqa.github.io/", "mirror": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
                "revision": REVISION, "license": "CC BY-SA 4.0", "upstream_records": len(records),
                "parquet_sha256": digest(parquet), "examples_sha256": digest(output),
                "selection": "sort SHA256('pilot-v1:' + id), first 8 with original and rotated full prompts <=2048 tokens; first 2 calibration; retain 10 documents, no truncation",
                "eligible_records": eligible, "selected_ids": selected_ids,
                "unavailable_original": "curtis.ml.cmu.edu DNS lookup failed on 2026-11-07",
                "synthetic": "12 examples: 3 independent worlds x (1/2/3 hops cross + 3 hops within); world 0 calibration",
                "counts": {s: sum(e["split"] == s for e in examples) for s in ["calibration", "test"]},
                "lengths": {e["id"]: len(e["prompt"]["ids"]) for e in examples}}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
