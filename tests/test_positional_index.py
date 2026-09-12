"""
test_positional_index.py: Unit tests for PositionalIndex.
Verifies positional postings, exact phrase search, ordered proximity search (WITHIN/k), and serialization.
"""

import os
import unittest
from src.corpus import CorpusManager
from src.preprocessor import Preprocessor
from src.positional_index import PositionalIndex

class TestPositionalIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = CorpusManager()
        cls.prep = Preprocessor()
        cls.index = PositionalIndex()
        cls.index.build_index(cls.corpus, cls.prep)

    def test_positional_postings_structure(self):
        self.assertEqual(self.index.num_docs, 100)
        # Check that postings contain non-empty sorted positions
        postings = self.index.get_postings("cotton")
        self.assertGreater(len(postings), 0)
        for p in postings:
            self.assertGreater(len(p.positions), 0)
            self.assertEqual(p.positions, sorted(p.positions))

    def test_exact_phrase_search(self):
        # "stretch denim" is present in multiple jeans documents
        results = self.index.exact_phrase_search("stretch denim", self.prep, self.corpus)
        self.assertGreater(len(results), 0)

        # Verify positional adjacency: pos2 == pos1 + 1
        for res in results:
            for span in res["matching_positions"]:
                self.assertEqual(len(span), 2)
                self.assertEqual(span[1], span[0] + 1)

    def test_ordered_proximity_search(self):
        # 'cotton WITHIN/3 shirt'
        results = self.index.proximity_search("cotton", "shirt", 3, self.prep, self.corpus)
        self.assertGreater(len(results), 0)

        for res in results:
            for p1, p2 in res["matching_positions"]:
                diff = p2 - p1
                self.assertGreaterEqual(diff, 1)
                self.assertLessEqual(diff, 3)

    def test_proximity_query_parser(self):
        parsed = self.index.parse_and_search_proximity("cotton WITHIN/3 shirt", self.prep, self.corpus)
        self.assertIsNotNone(parsed)
        self.assertGreater(len(parsed), 0)

        invalid = self.index.parse_and_search_proximity("cotton NEAR shirt", self.prep, self.corpus)
        self.assertIsNone(invalid)

    def test_json_export_and_load(self):
        temp_export = "data/temp_test_positional_index.json"
        self.index.export_to_json(temp_export)
        self.assertTrue(os.path.exists(temp_export))

        new_index = PositionalIndex()
        new_index.load_from_json(temp_export)
        self.assertEqual(new_index.num_docs, 100)
        self.assertEqual(len(new_index.postings), len(self.index.postings))

        if os.path.exists(temp_export):
            os.remove(temp_export)


if __name__ == "__main__":
    unittest.main()
