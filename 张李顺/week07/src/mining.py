import random
from collections import defaultdict


def grams(text):
    text = "".join(text.split())
    return set(text) | {text[i:i + 2] for i in range(max(0, len(text) - 1))}


def overlap(a, b):
    x, y = grams(a), grams(b)
    return len(x & y) / max(1, len(x | y))


def mine_triplets(rows, size, seed=42):
    rng = random.Random(seed)
    positives = defaultdict(set)
    negatives = defaultdict(list)
    positive_rows = []
    for row in rows:
        q, d, label = row["sentence1"], row["sentence2"], int(row["label"])
        if label:
            positives[q].add(d)
            positive_rows.append(row)
        else:
            negatives[q].append(d)
    eligible = [row for row in positive_rows if negatives[row["sentence1"]]]
    selected = rng.sample(eligible, min(size, len(eligible)))
    triplets = []
    for row in selected:
        q, p = row["sentence1"], row["sentence2"]
        pool = [x for x in negatives[q] if x not in positives[q] and x != q]
        if pool:
            n = max(pool, key=lambda x: overlap(q, x))
            triplets.append((q, p, n))
    rng.shuffle(triplets)
    return triplets
