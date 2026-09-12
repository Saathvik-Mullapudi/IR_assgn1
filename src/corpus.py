"""
corpus.py: Corpus Management and Validation Module.
Parses, structures, and validates the official 100-document clothing corpus.

IR Concepts Used:
- Document Representation: Transforming semi-structured document collections (SGML/XML tags like <DOC>, <DOCID>, <CATEGORY>, <TITLE>, <TEXT>) into clean, structured in-memory document objects.
- Fielded Information: Distinguishing between metadata (Category, Title) and body content (Text/Description) while supporting unified full-text indexing.
- Document Identification: Normalizing document identifiers (e.g., 'D001' -> int doc_id 1 and canonical string 'D001') to support tie-breaking by increasing document ID as required by the Vector Space Model specification.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional

EXPECTED_CATEGORIES = {
    "t-shirt", "shirt", "jeans", "kurta", "saree",
    "dress", "hoodie", "jacket", "leggings", "sweatshirt"
}

class Document:
    """Represents a single clothing product document."""
    def __init__(self, doc_id: int, doc_id_str: str, category: str, title: str, description: str):
        self.doc_id = int(doc_id)
        self.doc_id_str = doc_id_str.strip()
        self.category = category.strip()
        self.title = title.strip()
        self.description = description.strip()

    @property
    def full_text(self) -> str:
        """Combines title and description for comprehensive full-text indexing."""
        return f"{self.title} {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_id_str": self.doc_id_str,
            "category": self.category,
            "title": self.title,
            "description": self.description
        }

    def __repr__(self) -> str:
        return f"<Document {self.doc_id_str} (id={self.doc_id}) [{self.category}]: {self.title[:35]}...>"


class CorpusManager:
    """Manages parsing, schema validation, and retrieval of clothing documents."""

    def __init__(self, raw_path: str = "data/clothing_corpus_raw.txt", json_path: str = "data/clothing_corpus.json"):
        self.raw_path = raw_path
        self.json_path = json_path
        self.documents: Dict[int, Document] = {}
        self.load_or_parse()

    def load_or_parse(self) -> None:
        """Parses raw SGML/XML corpus if needed, saves JSON, and validates schema."""
        if os.path.exists(self.raw_path):
            self.parse_raw_sgml(self.raw_path)
            # Save parsed json for quick future loading
            self.save_json(self.json_path)
        elif os.path.exists(self.json_path):
            self.load_from_json(self.json_path)
        else:
            raise FileNotFoundError(f"Neither raw corpus ({self.raw_path}) nor json ({self.json_path}) found.")

        self.validate()

    def parse_raw_sgml(self, raw_path: str) -> None:
        """Parses <DOC> ... <DOCID>...</DOCID> ... </DOC> format."""
        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_blocks = re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL)
        if not doc_blocks:
            raise ValueError(f"No <DOC> blocks found in {raw_path}")

        self.documents.clear()
        for block in doc_blocks:
            docid_match = re.search(r"<DOCID>(.*?)</DOCID>", block, re.DOTALL)
            category_match = re.search(r"<CATEGORY>(.*?)</CATEGORY>", block, re.DOTALL)
            title_match = re.search(r"<TITLE>(.*?)</TITLE>", block, re.DOTALL)
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL)

            if not (docid_match and category_match and title_match and text_match):
                continue

            doc_id_str = docid_match.group(1).strip()
            # Extract numeric id (e.g. 'D001' -> 1)
            num_match = re.search(r"\d+", doc_id_str)
            numeric_id = int(num_match.group(0)) if num_match else len(self.documents) + 1

            category = category_match.group(1).strip()
            title = title_match.group(1).strip()
            description = text_match.group(1).strip()

            self.documents[numeric_id] = Document(
                doc_id=numeric_id,
                doc_id_str=doc_id_str,
                category=category,
                title=title,
                description=description
            )

    def load_from_json(self, json_path: str) -> None:
        """Loads corpus from structured JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.documents.clear()
        for item in items:
            doc = Document(
                doc_id=item["doc_id"],
                doc_id_str=item.get("doc_id_str", f"D{item['doc_id']:03d}"),
                category=item["category"],
                title=item["title"],
                description=item["description"]
            )
            self.documents[doc.doc_id] = doc

    def save_json(self, json_path: str) -> None:
        """Saves current corpus to JSON format."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        items = [doc.to_dict() for doc in self.get_all_documents()]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    def validate(self) -> None:
        """Validates that corpus satisfies all assignment specifications."""
        # 1. Total document count must be exactly 100 (N = 100)
        if len(self.documents) != 100:
            raise ValueError(f"Corpus must contain exactly 100 documents, found {len(self.documents)}")

        # 2. Check document IDs are unique and sequential 1..100
        expected_ids = set(range(1, 101))
        actual_ids = set(self.documents.keys())
        if expected_ids != actual_ids:
            missing = expected_ids - actual_ids
            raise ValueError(f"Corpus document IDs are missing: {missing}")

        # 3. Check categories cover all 10 expected categories
        categories_found = {doc.category.lower().replace("-", "") for doc in self.documents.values()}
        for cat in EXPECTED_CATEGORIES:
            normalized_cat = cat.lower().replace("-", "")
            if normalized_cat not in categories_found:
                raise ValueError(f"Missing required category: {cat}")

        # 4. Check that title and description are non-empty for all docs
        for doc in self.documents.values():
            if not doc.title:
                raise ValueError(f"Document {doc.doc_id_str} has empty title")
            if not doc.description:
                raise ValueError(f"Document {doc.doc_id_str} has empty description")

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Retrieve document by integer ID."""
        return self.documents.get(doc_id)

    def get_document_by_str_id(self, doc_id_str: str) -> Optional[Document]:
        """Retrieve document by canonical string ID (e.g. 'D001')."""
        clean_str = doc_id_str.strip().upper()
        for doc in self.documents.values():
            if doc.doc_id_str.upper() == clean_str:
                return doc
        return None

    def get_all_documents(self) -> List[Document]:
        """Returns all documents sorted by ascending doc_id."""
        return [self.documents[i] for i in sorted(self.documents.keys())]

    def get_category_distribution(self) -> Dict[str, int]:
        """Returns document count per category."""
        counts = {}
        for doc in self.documents.values():
            counts[doc.category] = counts.get(doc.category, 0) + 1
        return counts

    def summary(self) -> str:
        """Returns human-readable summary of the loaded corpus."""
        dist = self.get_category_distribution()
        lines = [
            f"=== Clothing Corpus Summary ===",
            f"Total Documents: {len(self.documents)} (N=100)",
            f"Categories ({len(dist)} distinct):"
        ]
        for cat, cnt in sorted(dist.items()):
            lines.append(f"  - {cat:15s}: {cnt} documents")
        return "\n".join(lines)


if __name__ == "__main__":
    corpus = CorpusManager()
    print(corpus.summary())
    print("\n[Sample Document D001]:")
    sample = corpus.get_document(1)
    print(f"ID: {sample.doc_id_str} (numeric={sample.doc_id})")
    print(f"Category: {sample.category}")
    print(f"Title: {sample.title}")
    print(f"Description: {sample.description[:120]}...")
    print("\nDataset validation PASSED successfully!")
