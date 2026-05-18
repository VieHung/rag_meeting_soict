"""Integration tests cho luồng transcript Phase 2.

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
COLLECTION = "test_transcripts"


def _new_meeting_id() -> str:
    return f"test_meeting_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture
def meeting_id(http_client):
    mid = _new_meeting_id()
    # init
    r = http_client.post(
        f"/transcript/{COLLECTION}/meeting/init",
        json={"meeting_id": mid},
    )
    assert r.status_code == 200, r.text
    yield mid
    # cleanup
    try:
        http_client.delete(f"/transcript/{COLLECTION}/meeting/{mid}")
    except Exception:
        pass


# ---- Health ----------------------------------------------------------------


class TestHealth:
    def test_health(self, http_client):
        r = http_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


# ---- Meeting lifecycle -----------------------------------------------------


class TestMeetingLifecycle:
    def test_init_meeting(self, http_client):
        mid = _new_meeting_id()
        try:
            r = http_client.post(
                f"/transcript/{COLLECTION}/meeting/init",
                json={"meeting_id": mid},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["meeting_id"] == mid
            assert body["status"] == "initialized"
        finally:
            http_client.delete(f"/transcript/{COLLECTION}/meeting/{mid}")

    def test_init_duplicate_meeting(self, http_client, meeting_id):
        r = http_client.post(
            f"/transcript/{COLLECTION}/meeting/init",
            json={"meeting_id": meeting_id},
        )
        assert r.status_code == 409

    def test_init_force_reset(self, http_client, meeting_id):
        # embed 1 câu rồi force_reset
        http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={
                "meeting_id": meeting_id,
                "speaker": "A",
                "text": "Câu sẽ bị xoá",
            },
        )
        time.sleep(0.5)
        r = http_client.post(
            f"/transcript/{COLLECTION}/meeting/init",
            json={"meeting_id": meeting_id, "force_reset": True},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "reset"

    def test_delete_meeting(self, http_client):
        mid = _new_meeting_id()
        http_client.post(
            f"/transcript/{COLLECTION}/meeting/init",
            json={"meeting_id": mid},
        )
        r = http_client.delete(f"/transcript/{COLLECTION}/meeting/{mid}")
        assert r.status_code == 200
        body = r.json()
        assert body["meeting_id"] == mid
        assert body["success"] in (True, False)


# ---- Embed transcript ------------------------------------------------------


class TestEmbedTranscript:
    def test_embed_assigns_sequential_ids(self, http_client, meeting_id):
        seqs = []
        for i in range(5):
            r = http_client.post(
                f"/transcript/{COLLECTION}/embed",
                json={
                    "meeting_id": meeting_id,
                    "speaker": f"User_{i % 2}",
                    "text": f"Đây là câu thứ {i + 1} của cuộc họp.",
                },
            )
            assert r.status_code == 202, r.text
            body = r.json()
            seqs.append(body["sequence_id"])
            assert body["context_status"] == "pending"
            assert body["point_id"]
        assert seqs == [1, 2, 3, 4, 5]

    def test_embed_without_init_returns_404(self, http_client):
        r = http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={
                "meeting_id": _new_meeting_id(),
                "speaker": "Ghost",
                "text": "Câu của meeting chưa init",
            },
        )
        assert r.status_code == 404

    def test_embed_empty_text_returns_422(self, http_client, meeting_id):
        r = http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={
                "meeting_id": meeting_id,
                "speaker": "A",
                "text": "",
            },
        )
        assert r.status_code == 422

    def test_embed_whitespace_only_returns_422(self, http_client, meeting_id):
        r = http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={
                "meeting_id": meeting_id,
                "speaker": "A",
                "text": "   ",
            },
        )
        assert r.status_code == 422


# ---- Segments --------------------------------------------------------------


class TestSegments:
    def _seed(self, http_client, meeting_id, count=5):
        for i in range(count):
            http_client.post(
                f"/transcript/{COLLECTION}/embed",
                json={
                    "meeting_id": meeting_id,
                    "speaker": f"User_{i % 2}",
                    "text": f"Nội dung câu {i + 1}.",
                },
            )
        time.sleep(1)  # đợi qdrant index

    def test_list_segments(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        r = http_client.get(
            f"/transcript/{COLLECTION}/meeting/{meeting_id}/segments",
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
        r = http_client.get(
            f"/transcript/{COLLECTION}/meeting/{meeting_id}/segments",
            params={"from_seq": 2, "to_seq": 4},
        )
        assert r.status_code == 200
        body = r.json()
        seqs = [s["sequence_id"] for s in body["segments"]]
        assert seqs == [2, 3, 4]


# ---- Query transcript with window ------------------------------------------


class TestQueryTranscript:
    def _seed(self, http_client, meeting_id):
        utterances = [
            ("Mai Xuân Ngọc", "Chúng ta cần xem ngân sách quý 4 năm nay."),
            ("Đoàn Sỹ Nguyên", "Tôi nghĩ nên cắt giảm chi phí vận hành 15%."),
            ("Mai Xuân Ngọc", "Vậy chốt phương án cắt giảm 15% nhé?"),
            ("Đoàn Sỹ Nguyên", "Đồng ý. Triển khai từ tháng tới."),
            ("Mai Xuân Ngọc", "Tiếp theo bàn về kế hoạch tuyển dụng."),
        ]
        for speaker, text in utterances:
            http_client.post(
                f"/transcript/{COLLECTION}/embed",
                json={"meeting_id": meeting_id, "speaker": speaker, "text": text},
            )
        time.sleep(1.2)

    def test_query_returns_results_with_window(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": COLLECTION,
                "query": "ngân sách quý 4",
                "meeting_id": meeting_id,
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
        # window_size lớn — sẽ bị clamp theo TRANSCRIPT_MAX_WINDOW_SIZE
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": COLLECTION,
                "query": "cắt giảm",
                "meeting_id": meeting_id,
                "top_k": 1,
                "window_size": 999,
            },
        )
        assert r.status_code == 200

    def test_query_with_speaker_filter(self, http_client, meeting_id):
        self._seed(http_client, meeting_id)
        r = http_client.post(
            "/query/transcript",
            json={
                "collection": COLLECTION,
                "query": "cắt giảm",
                "meeting_id": meeting_id,
                "speaker_filter": "Đoàn Sỹ Nguyên",
                "top_k": 5,
            },
        )
        assert r.status_code == 200
        body = r.json()
        for res in body["results"]:
            assert res["speaker"] == "Đoàn Sỹ Nguyên"

    def test_query_empty_query(self, http_client):
        r = http_client.post(
            "/query/transcript",
            json={"collection": COLLECTION, "query": "", "top_k": 3},
        )
        # Pydantic validation -> 422
        assert r.status_code == 422


# ---- Context endpoints -----------------------------------------------------


class TestContext:
    def test_first_utterance_has_empty_context(self, http_client, meeting_id):
        # câu đầu tiên (seq=1) phải có context = "" và status ready (LLM_PROVIDER=none cũng ready)
        http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={"meeting_id": meeting_id, "speaker": "A", "text": "Câu mở đầu."},
        )
        # đợi background task
        time.sleep(2.0)
        r = http_client.get(
            f"/transcript/{COLLECTION}/context/latest",
            params={"meeting_id": meeting_id},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["sequence_id"] == 1
        assert body["context"] == ""
        assert body["context_status"] in ("ready", "pending", "processing")

    def test_patch_context_manually(self, http_client, meeting_id):
        http_client.post(
            f"/transcript/{COLLECTION}/embed",
            json={"meeting_id": meeting_id, "speaker": "A", "text": "Câu một."},
        )
        time.sleep(1.0)
        r = http_client.patch(
            f"/transcript/{COLLECTION}/context/1",
            json={
                "meeting_id": meeting_id,
                "context": "Tóm tắt thủ công.",
                "context_status": "ready",
                "context_seq_base": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["context"] == "Tóm tắt thủ công."
        assert body["context_status"] == "ready"

        # GET lại để xác nhận persist
        r2 = http_client.get(
            f"/transcript/{COLLECTION}/context/1",
            params={"meeting_id": meeting_id},
        )
        assert r2.status_code == 200
        assert r2.json()["context"] == "Tóm tắt thủ công."


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
