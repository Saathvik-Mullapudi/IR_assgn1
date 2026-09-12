"""
vsm_search.py: Vector Space Model (VSM) Ranked Retrieval using lnc.ltc Weighting.

Weighting Scheme (lnc.ltc):
- Document (lnc):
    - l (log-tf): wd,t = 1 + log10(tf) for tf > 0
    - n (no idf): no idf applied to documents
    - c (cosine): normalized by Euclidean length ||d|| = sqrt(sum(wd,t^2))
- Query (ltc):
    - l (log-tf): 1 + log10(tf) for tf > 0
    - t (idf): log10(N / df) where N = 100
    - c (cosine): normalized by Euclidean length ||q|| = sqrt(sum(wq,t^2))

Cosine Similarity:
    score(q, d) = (wq . wd) / (||q|| * ||d||)
Tie-breaking:
    Sorted by decreasing similarity; ties broken by increasing docID.
"""

import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.corpus import CorpusManager, Document
from src.preprocessor import Preprocessor
from src.inverted_index import InvertedIndex


class VSMSearchEngine:
    """
    Vector Space Model search engine implementing the exact lnc.ltc SMART weighting scheme.
    """

    def __init__(self, corpus: CorpusManager, preprocessor: Preprocessor, index: InvertedIndex):
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.index = index
        self.N = self.index.num_docs  # Total documents in corpus (100)

    def compute_query_weights(self, query: str) -> Tuple[Dict[str, float], float]:
        """
        Computes unnormalized query weights (ltc):
        wq,t = (1 + log10(tf_q,t)) * log10(N / df_t)
        Also computes the query Euclidean length ||q|| = sqrt(sum(wq,t^2)).
        """
        terms = self.preprocessor.process(query)
        if not terms:
            return {}, 0.0

        tf_q = Counter(terms)
        weights: Dict[str, float] = {}
        sum_sq = 0.0

        for term, tf in tf_q.items():
            df = self.index.get_df(term)
            if df > 0:
                # ltc: log-tf * idf
                log_tf = 1.0 + math.log10(tf)
                idf = math.log10(self.N / df)
                w = log_tf * idf
                if w > 0:
                    weights[term] = w
                    sum_sq += w * w

        query_len = math.sqrt(sum_sq)
        return weights, query_len

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs free-text ranked retrieval:
        1. Calculates query weights (ltc) and query length ||q||.
        2. Uses accumulators to compute dot product (wq . wd).
        3. Normalizes by ||q|| and pre-computed ||d|| to obtain cosine similarity.
        4. Sorts by decreasing similarity; breaks ties by increasing doc_id.
        5. Returns up to top_k matching documents.
        """
        query_weights, query_norm = self.compute_query_weights(query)

        # If query has no matching terms or zero norm
        if query_norm == 0.0 or not query_weights:
            return []

        # Accumulator: doc_id -> dot product sum(wq,t * wd,t)
        accumulators: Dict[int, float] = defaultdict(float)

        for term, w_qt in query_weights.items():
            postings = self.index.get_postings(term)
            for p in postings:
                # Document weight (lnc): wd,t = 1 + log10(tf)
                w_dt = 1.0 + math.log10(p.tf)
                accumulators[p.doc_id] += w_qt * w_dt

        # Compute cosine similarity and package results
        results = []
        for doc_id, dot_product in accumulators.items():
            doc_len = self.index.get_doc_length(doc_id)
            if doc_len > 0:
                score = dot_product / (query_norm * doc_len)
            else:
                score = 0.0

            doc = self.corpus.get_document(doc_id)
            if doc:
                results.append({
                    "doc_id": doc_id,
                    "doc_id_str": doc.doc_id_str,
                    "category": doc.category,
                    "title": doc.title,
                    "description": doc.description,
                    "score": round(score, 4),
                    "raw_score": score
                })

        # Tie-breaking: decreasing similarity score, then increasing doc_id
        results.sort(key=lambda r: (-r["raw_score"], r["doc_id"]))

        return results[:top_k]


if __name__ == "__main__":
    corpus = CorpusManager()
    preprocessor = Preprocessor()
    index = InvertedIndex()
    index.build_index(corpus, preprocessor)

    engine = VSMSearchEngine(corpus, preprocessor, index)

    sample_query = "breathable cotton fabric"
    print(f"=== VSM (lnc.ltc) Ranked Retrieval for: '{sample_query}' ===")
    hits = engine.search(sample_query, top_k=10)
    for rank, hit in enumerate(hits, start=1):
        print(f"Rank {rank:2d} | DocID: {hit['doc_id_str']} (id={hit['doc_id']:2d}) | Score: {hit['score']:.4f} | Category: {hit['category']:10s} | Title: {hit['title']}")
