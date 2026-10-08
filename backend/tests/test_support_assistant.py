import pytest
from fastapi.testclient import TestClient

from main import app
from support_assistant.tools import list_directory, read_document, search_documents_by_name
from support_assistant.vector_store import vector_store

client = TestClient(app)


def test_vector_store_loaded():
    """Verify that the vector store loaded all chunks and supports cosine search."""
    stats = vector_store.get_stats()
    assert stats["is_indexed"] is True
    assert stats["total_chunks"] > 0
    assert stats["total_documents"] > 0

    results = vector_store.search("asymptotic notation big o", top_k=5)
    assert len(results) == 5
    assert all("source" in r and "section" in r and "content" in r for r in results)
    assert any("Asymptotic" in r["source"] or "Notation" in r["source"] for r in results)


def test_filesystem_tools():
    """Test filesystem exploration tools within the knowledge base."""
    # 1. list_directory
    root_listing = list_directory.invoke({"subpath": ""})
    assert "error" not in root_listing
    assert "directories" in root_listing
    assert len(root_listing["directories"]) > 0

    # 2. search_documents_by_name
    search_res = search_documents_by_name.invoke({"pattern": "Asymptotic*"})
    assert search_res["total_matches"] > 0
    assert any("Asymptotic" in m for m in search_res["matches"])

    # 3. read_document
    doc_path = search_res["matches"][0]
    doc_content = read_document.invoke({"file_path": doc_path, "start_line": 1, "max_lines": 20})
    assert "error" not in doc_content
    assert doc_content["total_lines"] > 0
    assert len(doc_content["content"]) > 0


def test_stats_api_endpoint():
    """Verify GET /api/support-assistant/stats endpoint."""
    resp = client.get("/api/support-assistant/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_indexed"] is True
    assert data["total_chunks"] > 0
    assert data["total_documents"] > 0


def test_documents_api_endpoint():
    """Verify GET /api/support-assistant/documents endpoint."""
    resp = client.get("/api/support-assistant/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert data["total_documents"] > 0
