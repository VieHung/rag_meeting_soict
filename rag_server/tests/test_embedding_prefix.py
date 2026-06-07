"""Unit test cho prefix E5 trong EmbeddingService — dùng backend giả, không tải model.

Mục tiêu:
- embed_texts (passage) gắn prefix EMBEDDING_PASSAGE_PREFIX.
- embed_query gắn prefix EMBEDDING_QUERY_PREFIX.
- dim lấy từ backend.
"""
import importlib

import pytest


class FakeBackend:
    """Backend ghi lại text nhận được để kiểm tra prefix."""

    def __init__(self):
        self.seen = []

    @property
    def dim(self):
        return 768

    def encode(self, texts, batch_size=64):
        self.seen.extend(texts)
        return [[0.0] * self.dim for _ in texts]


@pytest.fixture
def service(monkeypatch):
    import app.services.embedding_backends as backends
    import app.services.embedding as embedding

    fake = FakeBackend()
    monkeypatch.setattr(backends, "create_backend", lambda: fake)
    monkeypatch.setattr(embedding, "create_backend", lambda: fake)
    # reset singleton
    embedding.EmbeddingService._instance = None
    svc = embedding.EmbeddingService()
    yield svc, fake
    embedding.EmbeddingService._instance = None


def test_passage_prefix_applied(service):
    svc, fake = service
    svc.embed_texts(["xin chào", "hello"])
    assert fake.seen == ["passage: xin chào", "passage: hello"]


def test_query_prefix_applied(service):
    svc, fake = service
    svc.embed_query("ngân sách Q4")
    assert fake.seen == ["query: ngân sách Q4"]


def test_dim_from_backend(service):
    svc, _ = service
    assert svc.dim == 768
