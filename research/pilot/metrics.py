"""HotpotQA answer EM/F1 definitions; evaluated against upstream in tests."""
import re
import string
from collections import Counter


def normalize(value):
    value = "".join(c for c in value.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def score(prediction, answer):
    p, a = normalize(prediction), normalize(answer)
    em = float(p == a)
    if (p in ["yes", "no", "noanswer"] or a in ["yes", "no", "noanswer"]) and p != a:
        return {"em": em, "f1": 0.0}
    pt, at = p.split(), a.split()
    common = sum((Counter(pt) & Counter(at)).values())
    f1 = 2 * common / (len(pt) + len(at)) if common else 0.0
    return {"em": em, "f1": f1}
