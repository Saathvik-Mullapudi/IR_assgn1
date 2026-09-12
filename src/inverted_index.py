"""
inverted_index.py: Standard Inverted Index Construction.
Constructs dictionary (term -> df) and postings lists (docID -> tf) for the 100-document clothing corpus.

IR Concepts Used:
- Inverted Index Architecture: The core indexing data structure in Information Retrieval, inverting document-to-term mapping into term-to-document mapping.
- Dictionary (Vocabulary): Stores all unique vocabulary terms $t$ alongside their Document Frequency ($df_t$, the number of documents containing $t$).
- Postings List: A sorted sequence of posting entries $(docID, tf_{t,d})$ tracking the occurrences of term $t$ in each document $d$.
- Term Frequency ($tf_{t,d}$): Local frequency metric indicating term importance within a specific document.
- Document Vector Length (Cosine Normalization): Pre-computing Euclidean lengths $\|d\| = \sqrt{\sum (1 + \log_{10}(tf_{t,d}))^2}$ for efficient $O(1)$ cosine normalization during ranked retrieval.
- Serialization & Memory Efficiency: Storing and serializing the dictionary and postings into structured, human-readable JSON formats.
"""

import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.corpus import CorpusManager, Document
from src.preprocessor import Preprocessor


class Posting:
    """Represents an individual posting record (docID, tf)."""
    def __init__(self, doc_id: int, doc_id_str: str, tf: int):
        self.doc_id = int(doc_id)
        self.doc_id_str = doc_id_str
        self.tf = int(tf)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_id_str": self.doc_id_str,
            "tf": self.tf
        }

    def to_tuple(self) -> Tuple[int, int]:
        return (self.doc_id, self.tf)

    def __repr__(self) -> str:
        return f"Posting(doc_id={self.doc_id} ({self.doc_id_str}), tf={self.tf})"


class InvertedIndex:
    """
    Standard Inverted Index with Term Frequencies and Document Frequencies.
    Supports lnc.ltc pre-computation of document Euclidean lengths.
    """

    def __init__(self):
        # term -> list of Posting
        self.postings: Dict[str, List[Posting]] = {}
        # term -> document frequency (df)
        self.df: Dict[str, int] = {}
        # doc_id -> Euclidean vector length for lnc document weights: sqrt(sum((1 + log10(tf))^2))
        self.doc_lengths: Dict[int, float] = {}
        # doc_id -> canonical string id (e.g. 1 -> "D001")
        self.doc_id_map: Dict[int, str] = {}
        # Total number of indexed documents (N)
        self.num_docs: int = 0

    def build_index(self, corpus: CorpusManager, preprocessor: Preprocessor) -> None:
        """
        Builds inverted index across all corpus documents.
        Computes tf for every term in each document, records df, and computes document vector lengths.
        """
        self.postings.clear()
        self.df.clear()
        self.doc_lengths.clear()
        self.doc_id_map.clear()

        all_docs = corpus.get_all_documents()
        self.num_docs = len(all_docs)

        # Temporary store: term -> list of (doc_id, doc_id_str, tf)
        temp_postings: Dict[str, List[Posting]] = {}

        for doc in all_docs:
            self.doc_id_map[doc.doc_id] = doc.doc_id_str

            # Tokenize and stem full document text (title + description)
            terms = preprocessor.process(doc.full_text)
            term_counts = Counter(terms)

            # Compute document vector length for lnc weighting:
            # wd,t = 1 + log10(tf) for tf > 0
            # Length ||d|| = sqrt(sum(wd,t^2))
            squared_weight_sum = 0.0
            for term, tf in term_counts.items():
                if tf > 0:
                    wd_t = 1.0 + math.log10(tf)
                    squared_weight_sum += wd_t * wd_t

                if term not in temp_postings:
                    temp_postings[term] = []
                temp_postings[term].append(Posting(doc.doc_id, doc.doc_id_str, tf))

            self.doc_lengths[doc.doc_id] = math.sqrt(squared_weight_sum) if squared_weight_sum > 0 else 1.0

        # Sort terms alphabetically for clean dictionary representation
        for term in sorted(temp_postings.keys()):
            plist = temp_postings[term]
            # Ensure postings are sorted by doc_id
            plist.sort(key=lambda p: p.doc_id)
            self.postings[term] = plist
            self.df[term] = len(plist)

    def get_df(self, term: str) -> int:
        """Returns document frequency df_t for term."""
        return self.df.get(term.lower(), 0)

    def get_postings(self, term: str) -> List[Posting]:
        """Returns postings list for term."""
        return self.postings.get(term.lower(), [])

    def get_doc_length(self, doc_id: int) -> float:
        """Returns pre-computed Euclidean length ||d|| for doc_id."""
        return self.doc_lengths.get(doc_id, 1.0)

    def get_vocabulary(self) -> List[str]:
        """Returns sorted list of vocabulary terms."""
        return sorted(self.postings.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes inverted index to a structured dictionary."""
        dict_output = {}
        for term, plist in self.postings.items():
            dict_output[term] = {
                "df": self.df[term],
                "postings": [p.to_dict() for p in plist]
            }

        return {
            "total_documents": self.num_docs,
            "vocabulary_size": len(self.postings),
            "doc_lengths": {str(k): round(v, 6) for k, v in self.doc_lengths.items()},
            "dictionary": dict_output
        }

    def export_to_json(self, output_path: str) -> None:
        """Exports inverted index to JSON deliverable file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, input_path: str) -> None:
        """Loads inverted index from JSON deliverable file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.num_docs = data.get("total_documents", 0)
        self.doc_lengths = {int(k): float(v) for k, v in data.get("doc_lengths", {}).items()}
        self.postings.clear()
        self.df.clear()

        dictionary = data.get("dictionary", {})
        for term, term_data in dictionary.items():
            self.df[term] = term_data["df"]
            plist = []
            for p in term_data["postings"]:
                plist.append(Posting(p["doc_id"], p["doc_id_str"], p["tf"]))
            self.postings[term] = plist

    def summary(self) -> str:
        """Returns summary metrics of the inverted index."""
        lines = [
            "=== Inverted Index Summary ===",
            f"Total Documents Indexed (N): {self.num_docs}",
            f"Unique Vocabulary Terms: {len(self.postings)}",
            "Top 10 Most Frequent Terms by Document Frequency (df):"
        ]
        sorted_terms = sorted(self.df.items(), key=lambda x: x[1], reverse=True)[:10]
        for t, d in sorted_terms:
            lines.append(f"  - '{t}': df = {d}")
        return "\n".join(lines)


if __name__ == "__main__":
    corpus = CorpusManager()
    preprocessor = Preprocessor()
    index = InvertedIndex()
    index.build_index(corpus, preprocessor)
    print(index.summary())

    # Export index deliverables
    index.export_to_json("inverted_index.json")
    index.export_to_json("output/inverted_index.json")
    print("\nInverted index successfully exported to inverted_index.json and output/inverted_index.json")
