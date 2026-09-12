"""
test_preprocessor.py: Unit tests for Preprocessor component.
Verifies tokenization, case normalization, stop-word removal, Porter stemming, and position tracking.
"""

import unittest
from src.preprocessor import Preprocessor

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.prep = Preprocessor()

    def test_case_normalization(self):
        tokens = self.prep.process("COTTON DENIM SHIRT")
        self.assertEqual(tokens, ["cotton", "denim", "shirt"])

    def test_possessives_and_punctuation(self):
        tokens = self.prep.process("Men's & Women's Wear, Blue-Black.")
        # 'wear' is kept, 'blue', 'black' kept, possessives stripped
        self.assertIn("men", tokens)
        self.assertIn("women", tokens)
        self.assertIn("wear", tokens)
        self.assertIn("blue", tokens)
        self.assertIn("black", tokens)

    def test_domain_stopword_preservation(self):
        # 'wear' and 'fit' MUST be retained in clothing IR
        tokens = self.prep.process("regular fit festive wear")
        self.assertIn("regular", tokens)
        self.assertIn("fit", tokens)
        self.assertIn("festiv", tokens)
        self.assertIn("wear", tokens)

    def test_stopword_removal(self):
        # 'is', 'a', 'for', 'with', 'the' should be removed
        tokens = self.prep.process("this is a dress for summer with the best style")
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("for", tokens)
        self.assertNotIn("with", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("dress", tokens)
        self.assertIn("summer", tokens)

    def test_porter_stemming(self):
        # Morphological conflations
        self.assertEqual(self.prep.stem_term("shirts"), "shirt")
        self.assertEqual(self.prep.stem_term("jackets"), "jacket")
        self.assertEqual(self.prep.stem_term("breathable"), "breathabl")
        self.assertEqual(self.prep.stem_term("festive"), "festiv")
        self.assertEqual(self.prep.stem_term("leggings"), "leg")

    def test_position_tracking(self):
        text = "Cotton shirt festive wear"
        pos_tokens = self.prep.process_with_positions(text)
        self.assertEqual(pos_tokens[0], ("cotton", 1))
        self.assertEqual(pos_tokens[1], ("shirt", 2))
        self.assertEqual(pos_tokens[2], ("festiv", 3))
        self.assertEqual(pos_tokens[3], ("wear", 4))

    def test_stopword_justification_exists(self):
        justification = self.prep.get_stopword_justification()
        self.assertTrue(len(justification) > 50)
        self.assertIn("Stop-Word Policy", justification)


if __name__ == "__main__":
    unittest.main()
