"""
test_vsm_search.py: Unit tests for VSMSearchEngine with lnc.ltc weighting.
Verifies cosine similarity, score ordering, tie-breaking by docID, top-10 limit, and OOV handling.
"""

import unittest
from src.corpus import CorpusManager
from src.preprocessor import Preprocessor
from src.inverted_index import InvertedIndex
from src.vsm_search import VSMSearchEngine

class TestVSMSearchEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = CorpusManager()
        cls.prep = Preprocessor()
        cls.index = InvertedIndex()
        cls.index.build_index(cls.corpus, cls.prep)
        cls.engine = VSMSearchEngine(cls.corpus, cls.prep, cls.index)

    def test_query_weights_calculation(self):
        weights, norm = self.engine.compute_query_weights("cotton")
        # 'cotton' is in the corpus so weight and norm must be positive
        self.assertIn("cotton", weights)
        self.assertGreater(weights["cotton"], 0.0)
        self.assertGreater(norm, 0.0)

    def test_search_results_ranking(self):
        results = self.engine.search("cotton t-shirt", top_k=10)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)

        # Check descending score order
        scores = [r["raw_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Check cosine similarity is within [0.0, 1.0]
        for r in results:
            self.assertGreater(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_out_of_vocabulary_query(self):
        # Query with words not appearing anywhere in clothing corpus
        results = self.engine.search("supercalifragilistic astronaut spaceship", top_k=10)
        self.assertEqual(results, [])

    def test_tie_breaking_order(self):
        # Even if synthetic or natural ties occur, verify doc_id ascending order for identical scores
        results = self.engine.search("casual wear", top_k=10)
        for i in range(len(results) - 1):
            if results[i]["raw_score"] == results[i+1]["raw_score"]:
                self.assertLess(results[i]["doc_id"], results[i+1]["doc_id"])


if __name__ == "__main__":
    unittest.main()
