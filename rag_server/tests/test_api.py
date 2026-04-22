import pytest
import httpx
import time
import uuid
from typing import Optional

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


class TestConfig:
    COLLECTION_NAME = "test_collection"
    SECOND_COLLECTION = "test_collection_2"


@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
def cleanup_collections():
    collections_to_cleanup = [TestConfig.COLLECTION_NAME, TestConfig.SECOND_COLLECTION]
    yield
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for col in collections_to_cleanup:
            try:
                client.post("/embed/collections", data={"name": col})
                client.delete("/embed/collections", data={"name": col})
            except:
                pass


class TestHealthCheck:
    def test_health_endpoint(self, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "message" in data


class TestCollectionManagement:
    def test_create_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/collections",
            data={"name": TestConfig.COLLECTION_NAME}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_create_duplicate_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/collections",
            data={"name": TestConfig.COLLECTION_NAME}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "already exists" in data["message"]

    def test_list_collections(self, http_client, cleanup_collections):
        response = http_client.get("/embed/collections")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert isinstance(data["collections"], list)
        assert TestConfig.COLLECTION_NAME in data["collections"]

    def test_create_second_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/collections",
            data={"name": TestConfig.SECOND_COLLECTION}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_collection_info(self, http_client, cleanup_collections):
        response = http_client.get(f"/embed/info/{TestConfig.COLLECTION_NAME}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == TestConfig.COLLECTION_NAME
        assert "vectors_count" in data
        assert "points_count" in data
        assert "status" in data

    def test_delete_collection(self, http_client, cleanup_collections):
        response = http_client.delete(
            "/embed/collections",
            data={"name": TestConfig.SECOND_COLLECTION}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_nonexistent_collection(self, http_client, cleanup_collections):
        response = http_client.delete(
            "/embed/collections",
            data={"name": "nonexistent_collection"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"]


class TestEmbedText:
    DOC_ID = None
    SOURCE = "test_embed_text.txt"

    def test_embed_text_default_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/text",
            json={
                "text": "RAG là viết tắt của Retrieval-Augmented Generation, một kỹ thuật kết hợp tìm kiếm và sinh text.",
                "source": self.SOURCE,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"] == self.SOURCE
        assert data["chunks_created"] > 0
        TestEmbedText.DOC_ID = data["doc_id"]

    def test_embed_text_custom_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/text",
            json={
                "text": "Python là ngôn ngữ lập trình phổ biến cho AI và machine learning.",
                "source": "python_info.txt",
                "collection": TestConfig.COLLECTION_NAME,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] > 0

    def test_embed_text_with_metadata(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/text",
            json={
                "text": "FastAPI là framework web hiệu năng cao cho Python.",
                "source": "fastapi_info.txt",
                "metadata": {"author": "test", "version": "1.0"},
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_embed_empty_text(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/text",
            json={
                "text": "",
                "source": "empty.txt",
            }
        )
        assert response.status_code == 422

    def test_embed_text_missing_source(self, http_client, cleanup_collections):
        response = http_client.post(
            "/embed/text",
            json={
                "text": "Some text content",
            }
        )
        assert response.status_code == 422


class TestEmbedFile:
    def test_embed_small_text_file(self, http_client, cleanup_collections):
        content = b"Day la noi dung file text cho test.\nLine 2\nLine 3"
        files = {"file": ("small_test.txt", content, "text/plain")}
        data = {"collection": TestConfig.COLLECTION_NAME}
        response = http_client.post("/embed/file", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["source"] == "small_test.txt"


class TestQuery:
    def test_query_basic(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "RAG la gi",
                "top_k": 3,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_found" in data
        assert "query" in data

    def test_query_with_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "Python",
                "collection": TestConfig.COLLECTION_NAME,
                "top_k": 5,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] >= 0

    def test_query_with_score_threshold(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "machine learning",
                "score_threshold": 0.8,
                "top_k": 10,
            }
        )
        assert response.status_code == 200
        data = response.json()
        for r in data["results"]:
            assert r["score"] >= 0.8

    def test_query_with_source_filter(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "Python",
                "source_filter": "python_info.txt",
                "top_k": 5,
            }
        )
        assert response.status_code == 200
        data = response.json()
        for r in data["results"]:
            assert r["source"] == "python_info.txt"

    def test_query_empty_query(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "",
                "top_k": 5,
            }
        )
        assert response.status_code == 400

    def test_query_large_top_k(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "test",
                "top_k": 50,
            }
        )
        assert response.status_code == 200

    def test_query_nonexistent_collection(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "test query",
                "collection": "nonexistent_collection_xyz",
                "top_k": 5,
            }
        )
        assert response.status_code in [200, 404]


class TestDeleteDocuments:
    def test_delete_by_source(self, http_client, cleanup_collections):
        http_client.post(
            "/embed/text",
            json={
                "text": "Content to be deleted by source",
                "source": "delete_source_test.txt",
            }
        )
        time.sleep(0.5)
        response = http_client.delete(
            f"/embed/{TestConfig.COLLECTION_NAME}/source/delete_source_test.txt"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_by_doc_id(self, http_client, cleanup_collections):
        doc_id = str(uuid.uuid4())
        http_client.post(
            "/embed/text",
            json={
                "text": "Content to be deleted by doc_id",
                "source": "delete_docid_test.txt",
                "doc_id": doc_id,
            }
        )
        time.sleep(0.5)
        response = http_client.delete(
            f"/embed/{TestConfig.COLLECTION_NAME}/doc/{doc_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_nonexistent_doc(self, http_client, cleanup_collections):
        response = http_client.delete(
            f"/embed/{TestConfig.COLLECTION_NAME}/doc/nonexistent-uuid"
        )
        assert response.status_code == 200


class TestListDocuments:
    def test_list_documents(self, http_client, cleanup_collections):
        response = http_client.get(f"/embed/{TestConfig.COLLECTION_NAME}/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    def test_list_documents_structure(self, http_client, cleanup_collections):
        response = http_client.get(f"/embed/{TestConfig.COLLECTION_NAME}/documents")
        assert response.status_code == 200
        data = response.json()
        for doc in data["documents"]:
            assert "doc_id" in doc
            assert "source" in doc
            assert "chunks" in doc


class TestCollectionInfo:
    def test_default_collection_info(self, http_client):
        response = http_client.get("/embed/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "vectors_count" in data
        assert "points_count" in data
        assert "status" in data
        assert "vector_size" in data
        assert "distance" in data


class TestEdgeCases:
    def test_large_text_embed(self, http_client, cleanup_collections):
        large_text = "Line " + "\n".join([f"of content {i}" for i in range(1000)])
        response = http_client.post(
            "/embed/text",
            json={
                "text": large_text,
                "source": "large_text.txt",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] > 10

    def test_special_characters_in_text(self, http_client, cleanup_collections):
        special_text = "Test with special chars: @#$%^&*() Vietnamese: Tiếng Việt Chinese: 中文"
        response = http_client.post(
            "/embed/text",
            json={
                "text": special_text,
                "source": "special_chars.txt",
            }
        )
        assert response.status_code == 200

    def test_unicode_content(self, http_client, cleanup_collections):
        unicode_text = """
        Tiếng Việt: Chào buổi sáng! 🎉
        中文：你好世界
        日本語：こんにちは
        العربية：مرحبا
        Emoji: 😀 😂 👍 ❤️
        """
        response = http_client.post(
            "/embed/text",
            json={
                "text": unicode_text,
                "source": "unicode_test.txt",
            }
        )
        assert response.status_code == 200

    def test_query_special_characters(self, http_client, cleanup_collections):
        response = http_client.post(
            "/query/",
            json={
                "query": "test @#$% special",
                "top_k": 5,
            }
        )
        assert response.status_code == 200


class TestPerformance:
    def test_multiple_consecutive_queries(self, http_client, cleanup_collections):
        start = time.time()
        for i in range(10):
            response = http_client.post(
                "/query/",
                json={
                    "query": f"test query {i}",
                    "top_k": 5,
                }
            )
            assert response.status_code == 200
        duration = time.time() - start
        assert duration < 30

    def test_multiple_embeds_same_source(self, http_client, cleanup_collections):
        for i in range(3):
            response = http_client.post(
                "/embed/text",
                json={
                    "text": f"Batch content {i}",
                    "source": "batch_test.txt",
                }
            )
            assert response.status_code == 200


class TestConcurrent:
    def test_concurrent_queries(self, http_client, cleanup_collections):
        import concurrent.futures

        def query_task(query_id):
            response = http_client.post(
                "/query/",
                json={
                    "query": f"concurrent query {query_id}",
                    "top_k": 3,
                }
            )
            return response.status_code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_task, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)


class TestErrorHandling:
    def test_invalid_json(self, http_client):
        response = http_client.post(
            "/query/",
            content=b"not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_query_field(self, http_client):
        response = http_client.post(
            "/query/",
            json={"top_k": 5}
        )
        assert response.status_code == 422

    def test_invalid_score_threshold(self, http_client):
        response = http_client.post(
            "/query/",
            json={
                "query": "test",
                "score_threshold": 1.5,
            }
        )
        assert response.status_code == 422


class TestCleanup:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup_after_tests(self, http_client, cleanup_collections):
        yield
        try:
            http_client.delete("/embed/collections", data={"name": TestConfig.COLLECTION_NAME})
            http_client.delete("/embed/collections", data={"name": TestConfig.SECOND_COLLECTION})
        except:
            pass