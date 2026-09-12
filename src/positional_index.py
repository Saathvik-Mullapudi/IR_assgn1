"""
positional_index.py: Positional Index Construction and Positional Intersect Algorithms.
Implements:
- Positional postings: term -> df -> [(docID, tf, [p1, p2, ...])]
- Exact Phrase Search using positional intersection (adjacent token matching)
- Ordered Proximity Search (term1 WITHIN/k term2: 1 <= pos2 - pos1 <= k)

IR Concepts Used:
- Positional Index: Extending inverted index postings lists with occurrence offsets [p1, p2, ...]
  to overcome the "bag of words" limitation of Boolean and Vector Space models.
- Positional Intersect: Linear-time two-pointer merge algorithm across document lists and
  embedded position arrays to detect exact phrase adjacency without document rescanning.
- k-Proximity Search: Searching for co-occurring concepts within a maximum word distance window (k)
  preserving word order (e.g. 'cotton' within 3 tokens of 'shirt').
- Elimination of False Positives: Positional filtering strictly eliminates documents where terms
  both appear but in reverse order or separated across unrelated sentences.
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.corpus import CorpusManager, Document
from src.preprocessor import Preprocessor


class PositionalPosting:
    """Represents a posting with term positions: (docID, tf, [p1, p2, ...])."""

    def __init__(self, doc_id: int, doc_id_str: str, positions: List[int]):
        self.doc_id = int(doc_id)
        self.doc_id_str = doc_id_str
        self.positions = sorted(positions)
        self.tf = len(self.positions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_id_str": self.doc_id_str,
            "tf": self.tf,
            "positions": self.positions
        }

    def __repr__(self) -> str:
        return f"PositionalPosting(doc={self.doc_id_str}, tf={self.tf}, pos={self.positions})"


class PositionalIndex:
    """
    Positional Inverted Index supporting exact phrase search and ordered proximity search.
    """

    def __init__(self):
        # term -> list of PositionalPosting
        self.postings: Dict[str, List[PositionalPosting]] = {}
        # term -> document frequency (df)
        self.df: Dict[str, int] = {}
        # doc_id -> canonical string ID
        self.doc_id_map: Dict[int, str] = {}
        # Total documents indexed
        self.num_docs: int = 0

    def build_index(self, corpus: CorpusManager, preprocessor: Preprocessor) -> None:
        """
        Constructs positional index from corpus using the preprocessor pipeline.
        Records 1-indexed token positions for each term.
        """
        self.postings.clear()
        self.df.clear()
        self.doc_id_map.clear()

        all_docs = corpus.get_all_documents()
        self.num_docs = len(all_docs)

        # Temporary accumulation: term -> doc_id -> list of positions
        temp_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        for doc in all_docs:
            self.doc_id_map[doc.doc_id] = doc.doc_id_str
            terms_with_pos = preprocessor.process_with_positions(doc.full_text)

            for term, pos in terms_with_pos:
                temp_index[term][doc.doc_id].append(pos)

        # Build sorted postings lists
        for term in sorted(temp_index.keys()):
            doc_dict = temp_index[term]
            plist = []
            for doc_id in sorted(doc_dict.keys()):
                doc_str = self.doc_id_map[doc_id]
                positions = doc_dict[doc_id]
                plist.append(PositionalPosting(doc_id, doc_str, positions))

            self.postings[term] = plist
            self.df[term] = len(plist)

    def get_df(self, term: str) -> int:
        return self.df.get(term.lower(), 0)

    def get_postings(self, term: str) -> List[PositionalPosting]:
        return self.postings.get(term.lower(), [])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes positional index to dictionary."""
        dict_output = {}
        for term, plist in self.postings.items():
            dict_output[term] = {
                "df": self.df[term],
                "postings": [p.to_dict() for p in plist]
            }

        return {
            "total_documents": self.num_docs,
            "vocabulary_size": len(self.postings),
            "dictionary": dict_output
        }

    def export_to_json(self, output_path: str) -> None:
        """Exports positional index to JSON deliverable file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, input_path: str) -> None:
        """Loads positional index from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.num_docs = data.get("total_documents", 0)
        self.postings.clear()
        self.df.clear()

        dictionary = data.get("dictionary", {})
        for term, term_data in dictionary.items():
            self.df[term] = term_data["df"]
            plist = []
            for p in term_data["postings"]:
                plist.append(PositionalPosting(p["doc_id"], p["doc_id_str"], p["positions"]))
            self.postings[term] = plist

    def exact_phrase_search(self, phrase: str, preprocessor: Preprocessor, corpus: CorpusManager) -> List[Dict[str, Any]]:
        """
        Executes exact phrase search via positional intersect.
        Matches documents where all phrase terms appear consecutively:
        pos(term_{i+1}) == pos(term_i) + 1.
        """
        terms = preprocessor.process(phrase)
        if not terms:
            return []

        # Single-term search
        if len(terms) == 1:
            term = terms[0]
            postings = self.get_postings(term)
            results = []
            for p in postings:
                doc = corpus.get_document(p.doc_id)
                if doc:
                    results.append({
                        "doc_id": p.doc_id,
                        "doc_id_str": p.doc_id_str,
                        "category": doc.category,
                        "title": doc.title,
                        "description": doc.description,
                        "matching_positions": [(pos,) for pos in p.positions],
                        "match_count": len(p.positions)
                    })
            results.sort(key=lambda x: x["doc_id"])
            return results

        # Multi-term phrase search using sequential positional intersection
        # First term candidates
        first_postings = self.get_postings(terms[0])
        if not first_postings:
            return []

        # candidates: doc_id -> list of starting positions
        candidates: Dict[int, List[int]] = {p.doc_id: list(p.positions) for p in first_postings}

        # Intersect with subsequent terms
        for offset, term in enumerate(terms[1:], start=1):
            next_postings = self.get_postings(term)
            if not next_postings:
                return []

            next_dict = {p.doc_id: set(p.positions) for p in next_postings}
            updated_candidates: Dict[int, List[int]] = {}

            for doc_id, start_positions in candidates.items():
                if doc_id in next_dict:
                    valid_starts = []
                    target_positions = next_dict[doc_id]
                    for start_pos in start_positions:
                        if (start_pos + offset) in target_positions:
                            valid_starts.append(start_pos)
                    if valid_starts:
                        updated_candidates[doc_id] = valid_starts

            candidates = updated_candidates
            if not candidates:
                break

        # Format output
        phrase_len = len(terms)
        results = []
        for doc_id in sorted(candidates.keys()):
            doc = corpus.get_document(doc_id)
            if doc:
                start_positions = candidates[doc_id]
                full_spans = [tuple(range(start, start + phrase_len)) for start in start_positions]
                results.append({
                    "doc_id": doc_id,
                    "doc_id_str": doc.doc_id_str,
                    "category": doc.category,
                    "title": doc.title,
                    "description": doc.description,
                    "matching_positions": full_spans,
                    "match_count": len(full_spans)
                })

        return results

    def proximity_search(self, term1: str, term2: str, k: int, preprocessor: Preprocessor, corpus: CorpusManager) -> List[Dict[str, Any]]:
        """
        Executes ordered proximity search: term1 WITHIN/k term2.
        Matches documents where term2 appears AFTER term1 within at most k token positions:
        1 <= pos2 - pos1 <= k.
        """
        stem1 = preprocessor.stem_term(term1)
        stem2 = preprocessor.stem_term(term2)

        plist1 = self.get_postings(stem1)
        plist2 = self.get_postings(stem2)

        if not plist1 or not plist2:
            return []

        dict2 = {p.doc_id: p.positions for p in plist2}
        results = []

        for p1 in plist1:
            doc_id = p1.doc_id
            if doc_id in dict2:
                positions2 = dict2[doc_id]
                matching_pairs = []
                for pos1 in p1.positions:
                    for pos2 in positions2:
                        diff = pos2 - pos1
                        if 1 <= diff <= k:
                            matching_pairs.append((pos1, pos2))

                if matching_pairs:
                    doc = corpus.get_document(doc_id)
                    if doc:
                        results.append({
                            "doc_id": doc_id,
                            "doc_id_str": doc.doc_id_str,
                            "category": doc.category,
                            "title": doc.title,
                            "description": doc.description,
                            "matching_positions": matching_pairs,
                            "match_count": len(matching_pairs)
                        })

        results.sort(key=lambda x: x["doc_id"])
        return results

    def parse_and_search_proximity(self, query_str: str, preprocessor: Preprocessor, corpus: CorpusManager) -> Optional[List[Dict[str, Any]]]:
        """
        Parses proximity query syntax like 'cotton WITHIN/3 shirt' or 'stretch WITHIN/4 denim'
        and executes proximity search.
        """
        pattern = r"^\s*([a-zA-Z0-9_-]+)\s+WITHIN/(\d+)\s+([a-zA-Z0-9_-]+)\s*$"
        match = re.match(pattern, query_str, re.IGNORECASE)
        if match:
            t1 = match.group(1)
            k = int(match.group(2))
            t2 = match.group(3)
            return self.proximity_search(t1, t2, k, preprocessor, corpus)
        return None


if __name__ == "__main__":
    corpus = CorpusManager()
    preprocessor = Preprocessor()
    pos_index = PositionalIndex()
    pos_index.build_index(corpus, preprocessor)

    # Export deliverable
    pos_index.export_to_json("positional_index.json")
    pos_index.export_to_json("output/positional_index.json")
    print("Positional Index built and exported to positional_index.json and output/positional_index.json")

    # Sample exact phrase query
    sample_phrase = "stretch denim"
    phrase_hits = pos_index.exact_phrase_search(sample_phrase, preprocessor, corpus)
    print(f"\n=== Exact Phrase Search for '{sample_phrase}' ({len(phrase_hits)} docs) ===")
    for h in phrase_hits[:5]:
        print(f"Doc {h['doc_id_str']} | Matching Positions: {h['matching_positions']} | Title: {h['title']}")

    # Sample proximity query
    prox_hits = pos_index.proximity_search("cotton", "shirt", 3, preprocessor, corpus)
    print(f"\n=== Proximity Search for 'cotton WITHIN/3 shirt' ({len(prox_hits)} docs) ===")
    for h in prox_hits[:5]:
        print(f"Doc {h['doc_id_str']} | Matching Positions: {h['matching_positions']} | Title: {h['title']}")
