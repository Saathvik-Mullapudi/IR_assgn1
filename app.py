"""
app.py: Clothing Search Engine Web Application.
Provides interactive UI for:
1. Free-text Search using Vector Space Model (VSM lnc.ltc cosine ranking).
2. Exact Phrase & Proximity Search using Positional Inverted Index.
3. Side-by-Side Dual Comparison Mode (Novelty Feature +3 Marks) demonstrating false positive elimination.
"""

import os
import sys
import time
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.corpus import CorpusManager
from src.preprocessor import Preprocessor
from src.inverted_index import InvertedIndex
from src.vsm_search import VSMSearchEngine
from src.positional_index import PositionalIndex

app = Flask(__name__)

# Initialize IR pipeline components on startup
print("Initializing Clothing Search Engine...")
corpus = CorpusManager()
preprocessor = Preprocessor()

inverted_index = InvertedIndex()
inverted_index.build_index(corpus, preprocessor)

vsm_engine = VSMSearchEngine(corpus, preprocessor, inverted_index)

positional_index = PositionalIndex()
positional_index.build_index(corpus, preprocessor)

print("Search engine ready! Loaded 100 documents.")


@app.route("/")
def index():
    """Renders the main search engine interface."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST", "GET"])
def search_api():
    """
    Search API endpoint.
    Supports query modes: 'vsm', 'positional', and 'compare'.
    """
    if request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        mode = data.get("mode", "vsm")
    else:
        query = request.args.get("query", "").strip()
        mode = request.args.get("mode", "vsm")

    if not query:
        return jsonify({
            "query": "",
            "mode": mode,
            "vsm_results": [],
            "positional_results": [],
            "time_ms": 0.0
        })

    start_time = time.perf_counter()

    vsm_results: List[Dict[str, Any]] = []
    pos_results: List[Dict[str, Any]] = []

    # 1. Run VSM Ranked Retrieval if requested or in compare mode
    if mode in ["vsm", "compare"]:
        vsm_results = vsm_engine.search(query, top_k=10)

    # 2. Run Positional Retrieval if requested or in compare mode
    if mode in ["positional", "compare"]:
        # Check if query is an explicit WITHIN/k proximity query
        prox_hits = positional_index.parse_and_search_proximity(query, preprocessor, corpus)
        if prox_hits is not None:
            pos_results = prox_hits
        else:
            # Otherwise perform exact phrase search
            pos_results = positional_index.exact_phrase_search(query, preprocessor, corpus)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # In compare mode, compute false positives (documents ranked in VSM that fail positional adjacency)
    comparison_info = {}
    if mode == "compare":
        pos_doc_ids = {r["doc_id"] for r in pos_results}
        vsm_doc_ids = [r["doc_id"] for r in vsm_results]
        false_positives = [doc_id for doc_id in vsm_doc_ids if doc_id not in pos_doc_ids]
        comparison_info = {
            "total_vsm_hits": len(vsm_results),
            "total_positional_hits": len(pos_results),
            "vsm_false_positives": false_positives,
            "explanation": (
                f"VSM retrieved {len(vsm_results)} documents based on term frequency (bag-of-words), "
                f"while Positional Index matched {len(pos_results)} documents where terms satisfy exact word ordering/adjacency. "
                f"{len(false_positives)} VSM documents contained query words but lacked exact phrase adjacency."
            )
        }

    return jsonify({
        "query": query,
        "mode": mode,
        "vsm_results": vsm_results,
        "positional_results": pos_results,
        "comparison": comparison_info,
        "time_ms": elapsed_ms
    })


@app.route("/api/stats", methods=["GET"])
def stats_api():
    """Returns general collection metrics."""
    return jsonify({
        "total_documents": corpus.get_all_documents().__len__(),
        "categories": corpus.get_category_distribution(),
        "vsm_vocab_size": len(inverted_index.postings),
        "positional_vocab_size": len(positional_index.postings)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
