"""Blue Orchid 电商平台内部 HTTP 客户端。

Echomind 不直接访问 Blue Orchid 数据库，而是通过同源网关暴露的内部接口读取
真实订单、购物车、收藏与商品目录数据。所有请求都携带 ``X-Internal-Token``，
且只做只读查询，不执行退款、取消订单、修改地址等写操作。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


_client: Optional["BlueOrchidClient"] = None


class BlueOrchidClient:
    """Blue Orchid 内部接口客户端。

    环境变量：
      - ``BLUE_ORCHID_BASE_URL``：Blue Orchid 网关地址，例如 http://localhost:3010
      - ``BLUE_ORCHID_INTERNAL_TOKEN``：必须与 Blue Orchid 的 ``ECHOMIND_INTERNAL_TOKEN`` 一致
      - ``BLUE_ORCHID_TIMEOUT_MS``：请求超时时间，默认 10000
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("BLUE_ORCHID_BASE_URL", "")).strip().rstrip("/")
        self.token = token if token is not None else os.getenv("BLUE_ORCHID_INTERNAL_TOKEN", "").strip()
        self.timeout_s = float(os.getenv("BLUE_ORCHID_TIMEOUT_MS", "10000")) / 1000.0
        # 复用连接池，避免每次工具调用都创建新的 AsyncClient。
        self._client = httpx.AsyncClient(timeout=self.timeout_s)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError(
                "Blue Orchid 内部接口未配置：缺少 BLUE_ORCHID_BASE_URL 或 BLUE_ORCHID_INTERNAL_TOKEN"
            )
        response = await self._client.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers={"X-Internal-Token": self.token},
        )
        response.raise_for_status()
        return response.json()

    async def orders(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        return await self._get(
            "/api/internal/echomind/orders",
            {"userId": user_id, "limit": limit},
        )

    async def store_state(self, user_id: str) -> Dict[str, Any]:
        return await self._get(
            "/api/internal/echomind/store-state",
            {"userId": user_id},
        )

    async def search_products(
        self,
        query: str,
        limit: int = 10,
        category: str = "",
        in_stock: bool = False,
    ) -> Dict[str, Any]:
        return await self._get(
            "/api/internal/echomind/products",
            {
                "q": query,
                "limit": limit,
                "category": category or "",
                "inStock": str(in_stock).lower(),
            },
        )

    async def product(self, product_id: int) -> Dict[str, Any]:
        return await self._get(f"/api/internal/echomind/products/{product_id}")


def get_blue_orchid_client() -> BlueOrchidClient:
    """返回进程内复用的 Blue Orchid 客户端单例。"""
    global _client
    if _client is None:
        _client = BlueOrchidClient()
    return _client
