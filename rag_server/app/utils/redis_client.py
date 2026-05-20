"""Singleton async Redis client.

Dùng cho:
- atomic sequence counter per meeting (INCR)
- cache `latest context` để build context tích lũy
"""
from __future__ import annotations

from typing import Optional

import redis.asyncio as redis

from app.config import settings


class RedisClient:
    """Wrapper singleton quanh `redis.asyncio.Redis`.

    Connection được tạo lazy, dùng connection pool mặc định.
    Cần gọi `close()` lúc shutdown để giải phóng pool.
    """

    _instance: Optional["RedisClient"] = None
    _client: Optional[redis.Redis] = None

    def __new__(cls) -> "RedisClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> redis.Redis:
        if RedisClient._client is None:
            RedisClient._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                encoding="utf-8",
            )
        return RedisClient._client

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception as e:
            print(f"[RedisClient] ping failed: {e}")
            return False

    async def close(self) -> None:
        if RedisClient._client is not None:
            try:
                await RedisClient._client.aclose()
            except Exception:
                pass
            RedisClient._client = None


def get_redis() -> RedisClient:
    """FastAPI dependency / helper."""
    return RedisClient()
