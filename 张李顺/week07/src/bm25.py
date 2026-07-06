from collections import Counter, defaultdict

import numpy as np


def terms(text):
    text = "".join(text.split())
    return list(text) + [text[i:i + 2] for i in range(max(0, len(text) - 1))]


class BM25:
    def __init__(self, docs, k1=1.5, b=.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.lengths = np.array([len(terms(x)) for x in docs], dtype=np.float32)
        self.avgdl = float(self.lengths.mean())
        postings = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(terms(doc)).items():
                postings[term].append((i, tf))
        self.postings = {term: (np.array([x[0] for x in values]), np.array([x[1] for x in values], dtype=np.float32)) for term, values in postings.items()}
        self.idf = {term: np.log(1 + (self.n - len(values) + .5) / (len(values) + .5)) for term, values in postings.items()}

    def scores(self, query):
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(terms(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avgdl)
            out[ids] += self.idf[term] * tf * (self.k1 + 1) / norm
        return out

    def pair_score(self, query, doc):
        tf = Counter(terms(doc))
        dl = sum(tf.values())
        score = 0
        for term in set(terms(query)):
            freq = tf.get(term, 0)
            if not freq:
                continue
            norm = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf.get(term, 0) * freq * (self.k1 + 1) / norm
        return score
