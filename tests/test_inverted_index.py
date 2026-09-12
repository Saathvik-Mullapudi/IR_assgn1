"""
test_inverted_index.py: Unit tests for InvertedIndex component.
Verifies dictionary, postings, term frequencies, document frequencies, and JSON serialization.
"""

import os
import unittest
from src.corpus import CorpusManager
from src.preprocessor import Preprocessor
from src.inverted_index import InvertedIndex

class TestInvertedIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corpus = CorpusManager()
        cls.prep = Preprocessor()
        cls.index = InvertedIndex()
        cls.index.build_index(cls.corpus, cls.prep)

    def test_total_indexed_documents(self):
        self.assertEqual(self.index.num_docs, 100)
        self.assertEqual(len(self.index.doc_lengths), 100)

    def test_vocabulary_and_frequencies(self):
        self.assertGreater(len(self.index.postings), 50)
        # Check standard clothing terms exist
        self.assertGreater(self.index.get_df("cotton"), 0)
        self.assertGreater(self.index.get_df("shirt"), 0)
        self.assertGreater(self.index.get_df("denim"), 0)

        # For every term, df must equal length of postings
        for term, plist in self.index.postings.items():
            self.assertEqual(self.index.df[term], len(plist))
            # Every posting must have tf >= 1
            for p in plist:
                self.assertGreaterEqual(p.tf, 1)

    def test_postings_are_sorted_by_doc_id(self):
        for term, plist in self.index.postings.items():
            doc_ids = [p.doc_id for p in plist]
            self.assertEqual(doc_ids, sorted(doc_ids))

    def test_doc_lengths_positive(self):
        for doc_id, length in self.index.doc_lengths.items():
            self.assertGreater(length, 0.0)

    def test_json_export_and_load(self):
        temp_export = "data/temp_test_inverted_index.json"
        self.index.export_to_json(temp_export)
        self.assertTrue(os.path.exists(temp_export))

        new_index = InvertedIndex()
        new_index.load_from_json(temp_export)
        self.assertEqual(new_index.num_docs, 100)
        self.assertEqual(len(new_index.postings), len(self.index.postings))
        self.assertEqual(new_index.get_df("cotton"), self.index.get_df("cotton"))

        # Clean up temporary test file
        if os.path.exists(temp_export):
            os.remove(temp_export)


if __name__ == "__main__":
    unittest.main()
