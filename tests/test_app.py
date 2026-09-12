"""
test_app.py: Integration tests for Flask Web Application.
Tests HTTP endpoints, search modes (VSM, positional, compare), and stats API.
"""

import unittest
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Clothing Search Engine", response.data)

    def test_api_vsm_search(self):
        response = self.client.post("/api/search", json={
            "query": "cotton crew neck",
            "mode": "vsm"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["mode"], "vsm")
        self.assertGreater(len(data["vsm_results"]), 0)
        self.assertIn("score", data["vsm_results"][0])

    def test_api_positional_search(self):
        response = self.client.post("/api/search", json={
            "query": "stretch denim",
            "mode": "positional"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["mode"], "positional")
        self.assertGreater(len(data["positional_results"]), 0)
        self.assertIn("matching_positions", data["positional_results"][0])

    def test_api_compare_search(self):
        response = self.client.post("/api/search", json={
            "query": "cotton shirt",
            "mode": "compare"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("comparison", data)
        self.assertIn("vsm_false_positives", data["comparison"])

    def test_api_stats(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["total_documents"], 100)


if __name__ == "__main__":
    unittest.main()
