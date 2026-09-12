"""
preprocessor.py: Text Pre-Processing Pipeline for Information Retrieval.
Implements Tokenization, Normalization, Stop-word Removal, Porter Stemming, and Positional Tracking.

IR Concepts Used:
- Tokenization: Dividing continuous text character streams into linguistic units (tokens).
- Case Normalization (Case Folding): Mapping all tokens to lowercase to prevent vocabulary inflation and mismatch.
- Punctuation Stripping & Possessive Handling: Normalizing contraction forms (e.g., "Men's" -> "men") and removing delimiters.
- Stop-word Removal: Removing non-discriminative, high-frequency grammatical noise while strictly preserving clothing domain keywords (e.g., 'wear', 'fit').
- Stemming (Porter Stemmer): Algorithmic suffix stripping to conflate morphological variants to canonical stems (e.g. 'shirts' -> 'shirt', 'breathable' -> 'breath').
- Positional Preservation: Tracking 1-indexed token offsets in the resulting term stream for phrase and proximity retrieval.
"""

import re
from typing import List, Tuple, Set, Optional
from nltk.stem import PorterStemmer


# Standard English Stop-Word List tailored and justified for Clothing IR:
# Retains domain-critical terms such as 'wear', 'fit', 'top', 'bottom'.
DEFAULT_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class Preprocessor:
    """
    Configurable text pre-processing pipeline for clothing search engine.
    Applies: Lowercasing -> Punctuation/Possessive Handling -> Tokenization -> Stop-word Filtering -> Porter Stemming.
    """

    def __init__(self, stop_words: Optional[Set[str]] = None, use_stemmer: bool = True):
        self.stop_words = DEFAULT_STOP_WORDS if stop_words is None else set(stop_words)
        self.use_stemmer = use_stemmer
        self.stemmer = PorterStemmer() if use_stemmer else None

    def tokenize_raw(self, text: str) -> List[str]:
        """
        Tokenizes text into words:
        1. Normalizes possessives (e.g. "Men's" -> "men", "Women's" -> "women").
        2. Replaces hyphens with spaces (e.g. "t-shirt" -> "t shirt", "skin-friendly" -> "skin friendly").
        3. Strips all punctuation and extracts alphanumeric word tokens.
        """
        if not text:
            return []

        # Case normalization (lowercasing)
        lowered = text.lower()

        # Remove possessive 's or s' (e.g., men's -> men)
        normalized = re.sub(r"['’]s\b", "", lowered)
        normalized = re.sub(r"s['’]\b", "s", normalized)

        # Treat hyphens, slashes, and underscores as word boundaries
        normalized = re.sub(r"[-_/]", " ", normalized)

        # Extract alphanumeric tokens (removes commas, dots, quotes, parentheses, etc.)
        tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
        return tokens

    def process(self, text: str) -> List[str]:
        """
        Complete pre-processing pipeline returning list of stemmed, filtered tokens.
        """
        tokens = self.tokenize_raw(text)
        filtered = [tok for tok in tokens if tok not in self.stop_words]
        if self.use_stemmer and self.stemmer:
            return [self.stemmer.stem(tok) for tok in filtered]
        return filtered

    def process_with_positions(self, text: str) -> List[Tuple[str, int]]:
        """
        Pre-processing pipeline returning list of (stemmed_term, 1_indexed_position) tuples.
        Positions reflect the 1-indexed order of indexed terms in the document stream.
        """
        tokens = self.tokenize_raw(text)
        result = []
        pos = 1
        for tok in tokens:
            if tok not in self.stop_words:
                stemmed = self.stemmer.stem(tok) if (self.use_stemmer and self.stemmer) else tok
                result.append((stemmed, pos))
                pos += 1
        return result

    def stem_term(self, term: str) -> str:
        """Helper to stem a single term."""
        lowered = term.lower().strip()
        if self.use_stemmer and self.stemmer:
            return self.stemmer.stem(lowered)
        return lowered

    @staticmethod
    def get_stopword_justification() -> str:
        """Returns documented justification of the stop-word policy."""
        return (
            "Stop-Word Policy Justification:\n"
            "1. Standard Noise Removal: High-frequency grammatical function words (articles, auxiliary verbs, "
            "pronouns, conjunctions) like 'the', 'is', 'at', 'and', 'from' are removed because they appear across "
            "nearly all documents, carrying virtually zero semantic discriminative value.\n"
            "2. Preservation of Clothing Domain Terms: Terms that might be pruned by overly aggressive generic "
            "stop-lists (such as 'wear', 'fit', 'top', 'suit') are explicitly retained because they represent crucial "
            "product categories and styling attributes in the clothing domain (e.g., 'festive wear', 'regular fit').\n"
            "3. Symmetric Consistency: The identical stop-word list and tokenization rules are applied symmetrically "
            "to both the document indexing phase and query processing to prevent vocabulary mismatch."
        )


if __name__ == "__main__":
    prep = Preprocessor()
    sample_text = "Men's Cotton Crew Neck T-Shirt - Black. Made from 100% cotton, this t-shirt is designed for everyday Indian wear."
    print("--- Original Text ---")
    print(sample_text)
    print("\n--- Tokens with Positions ---")
    tokens_with_pos = prep.process_with_positions(sample_text)
    for term, pos in tokens_with_pos:
        print(f"{pos:2d}: {term}")

    print("\n--- Stop-word Justification ---")
    print(prep.get_stopword_justification())
