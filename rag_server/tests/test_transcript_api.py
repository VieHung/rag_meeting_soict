"""Integration tests cho luồng transcript Phase 2 - Bản v2.

Theo phase2plan_v2.md:
- meeting_id suy ra từ collection (prefix meeting-)
- Lazy init - không cần endpoint init
- Endpoint gọp: context latest + by sequence_id -> /context?sequence_id=

Yêu cầu service đang chạy tại BASE_URL (qdrant + redis + rag_api).
Chạy:
    pytest tests/test_transcript_api.py -v
"""
import time
import uuid

import httpx
import pytest


BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0


def _new_meeting_id() -> str:
    return f"test_meeting_{uuid.uuid4().hex[:8]}"


def _collection_for_meeting(meeting_id: str) -> str:
    return f"meeting-{meeting_id}"


@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
def meeting_id(http_client):
    mid = _new_meeting_id()
    yield mid


# ---- Health ----------------------------------------------------------------


class TestHealth:
    def test_health(self, http_client):
        r = http_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


# ---- Embed transcript (lazy init) ------------------------------------------


class TestEmbedTranscript:
    def test_embed_assigns_sequential_ids(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        seqs = []
        for i in range(5):
            r = http_client.post(
                f"/transcript/{col}/embed",
                json={
                    "speaker": f"User_{i % 2}",
                    "text": f"Đây là câu thứ {i + 1} của cuộc họp.",
                },
            )
            assert r.status_code == 202, r.text
            body = r.json()
            seqs.append(body["sequence_id"])
            assert body["context_status"] in ("pending", "disabled")
            assert body["point_id"]
            assert body["meeting_id"] == meeting_id
        assert seqs == [1, 2, 3, 4, 5]

    def test_embed_empty_text_returns_422(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        r = http_client.post(
            f"/transcript/{col}/embed",
            json={
                "speaker": "A",
                "text": "",
            },
        )
        assert r.status_code == 422

    def test_embed_whitespace_only_returns_422(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        r = http_client.post(
            f"/transcript/{col}/embed",
            json={
                "speaker": "A",
                "text": "   ",
            },
        )
        assert r.status_code == 422

    def test_embed_wrong_collection_prefix_returns_400(self, http_client, meeting_id):
        r = http_client.post(
            "/transcript/docs-wrong/embed",
            json={"speaker": "A", "text": "Test"},
        )
        assert r.status_code == 400


# ---- Segments --------------------------------------------------------------


class TestSegments:
    def _seed(self, http_client, meeting_id, count=5):
        col = _collection_for_meeting(meeting_id)
        for i in range(count):
            http_client.post(
                f"/transcript/{col}/embed",
                json={
                    "speaker": f"User_{i % 2}",
                    "text": f"Nội dung câu {i + 1}.",
                },
            )
        time.sleep(1)

    def test_list_segments(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        col = _collection_for_meeting(meeting_id)
        r = http_client.get(
            f"/transcript/{col}/segments",
            params={"from_seq": 1, "to_seq": 5, "limit": 100},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["meeting_id"] == meeting_id
        assert body["count"] == 5
        seqs = [s["sequence_id"] for s in body["segments"]]
        assert seqs == [1, 2, 3, 4, 5]

    def test_segments_partial_range(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        col = _collection_for_meeting(meeting_id)
        r = http_client.get(
            f"/transcript/{col}/segments",
            params={"from_seq": 2, "to_seq": 4},
        )
        assert r.status_code == 200
        body = r.json()
        seqs = [s["sequence_id"] for s in body["segments"]]
        assert seqs == [2, 3, 4]


# ---- Context endpoint (merged) --------------------------------------------


class TestContext:
    def test_context_latest(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        http_client.post(
            f"/transcript/{col}/embed",
            json={"speaker": "A", "text": "Câu mở đầu."},
        )
        time.sleep(1.0)
        r = http_client.get(f"/transcript/{col}/context")
        assert r.status_code == 200
        body = r.json()
        assert body["meeting_id"] == meeting_id
        assert body["sequence_id"] == 1

    def test_context_at_sequence_id(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        for i in range(3):
            http_client.post(
                f"/transcript/{col}/embed",
                json={"speaker": "A", "text": f"Câu {i+1}"},
            )
        time.sleep(1.0)
        r = http_client.get(f"/transcript/{col}/context", params={"sequence_id": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["sequence_id"] == 2


# ---- Query transcript with window ------------------------------------------


class TestQueryTranscript:
    def _seed(self, http_client, meeting_id):
        col = _collection_for_meeting(meeting_id)
        utterances = [
            ("Mai Xuân Ngọc", "Chúng ta cần xem ngân sách quý 4 năm nay."),
            ("Đoàn Sỹ Nguyên", "Tôi nghĩ nên cắt giảm chi phí vận hành 15%."),
            ("Mai Xuân Ngọc", "Vậy chốt phương án cắt giảm 15% nhé?"),
            ("Đoàn Sỹ Nguyên", "Đồng ý. Triển khai từ tháng tới."),
            ("Mai Xuân Ngọc", "Tiếp theo bàn về kế hoạch tuyển dụng."),
        ]
        for speaker, text in utterances:
            http_client.post(
                f"/transcript/{col}/embed",
                json={"speaker": speaker, "text": text},
            )
        time.sleep(1.2)

    def test_query_returns_results_with_window(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        col = _collection_for_meeting(meeting_id)
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": col,
                "query": "ngân sách quý 4",
                "top_k": 3,
                "window_size": 1,
                "score_threshold": 0.0,
                "include_context": True,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] >= 1
        first = body["results"][0]
        assert "sequence_id" in first
        assert "speaker" in first
        assert "window" in first
        win = first["window"]
        assert isinstance(win["before"], list)
        assert isinstance(win["after"], list)

    def test_query_window_clamped(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        col = _collection_for_meeting(meeting_id)
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": col,
                "query": "cắt giảm",
                "top_k": 1,
                "window_size": 999,
            },
        )
        assert r.status_code == 200

    def test_query_with_speaker_filter(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        col = _collection_for_meeting(meeting_id)
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": col,
                "query": "cắt giảm",
                "speaker_filter": "Đoàn Sỹ Nguyên",
                "top_k": 5,
            },
        )
        assert r.status_code == 200
        body = r.json()
        for res in body["results"]:
            assert res["speaker"] == "Đoàn Sỹ Nguyên"

    def test_query_empty_query(self, http_client):
        col = _collection_for_meeting("test")
        r = http_client.post(
            "/query/transcript",
            json={"collection": col, "query": "", "top_k": 3},
        )
        assert r.status_code == 422

    def test_query_wrong_collection_prefix_returns_400(self, http_client):
        r = http_client.post(
            "/query/transcript",
            json={"collection": "docs-wrong", "query": "test", "top_k": 3},
        )
        assert r.status_code == 400


# ---- Backward compatibility (Phase 1 không bị phá) -------------------------


class TestPhase1Compat:
    def test_embed_collections_endpoint_still_works(self, http_client):
        r = http_client.get("/embed/collections")
        assert r.status_code == 200
        assert "collections" in r.json()

    def test_query_documents_endpoint_still_works(self, http_client):
        r = http_client.post("/query/", json={"query": "test", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert "total_found" in body
