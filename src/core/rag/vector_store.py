from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Union
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


@dataclass
class VectorMatch:
    """Single search hit returned from the vector store."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class QdrantVectorStore:
    """Thin async wrapper over Qdrant search functionality."""

    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout: float | None = None,
        score_threshold: float | None = None,
    ) -> None:
        if not url and not host:
            raise ValueError("Qdrant configuration requires either url or host")

        self.collection = collection
        self.score_threshold = score_threshold
        insecure = (url or "").startswith("http://") or (host in {"localhost", "127.0.0.1"})
        safe_api_key = None if insecure else api_key
        self._client = AsyncQdrantClient(
            url=url,
            host=host,
            port=port,
            api_key=safe_api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: qmodels.Filter | None = None,
        with_payload: bool = True,
    ) -> list[VectorMatch]:
        try:
            points = await self._client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                limit=limit,
                score_threshold=self.score_threshold,
                query_filter=filters,
                with_payload=with_payload,
            )
        except UnexpectedResponse as exc:
            if "doesn't exist" in str(exc).lower():
                await self.ensure_collection(len(vector))
                points = await self._client.search(
                    collection_name=self.collection,
                    query_vector=list(vector),
                    limit=limit,
                    score_threshold=self.score_threshold,
                    query_filter=filters,
                    with_payload=with_payload,
                )
            else:
                logger.error("Qdrant search failed: %s", exc)
                raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Qdrant search failed: %s", exc)
            raise

        matches: list[VectorMatch] = []
        for point in points or []:
            payload = dict(getattr(point, "payload", {}) or {})
            text = payload.get("text") or payload.get("content") or ""
            match = VectorMatch(
                id=str(getattr(point, "id", "")),
                score=float(getattr(point, "score", 0.0) or 0.0),
                text=str(text),
                metadata=payload,
            )
            matches.append(match)
        return matches

    async def upsert(
        self,
        vectors: Iterable[Sequence[float]],
        payloads: Iterable[dict[str, Any]],
        *,
        ids: Iterable[Union[int, str]] | None = None,
        batch_size: int = 64,
    ) -> None:
        """Utility helper for bulk upserts (used by background ingestion scripts)."""

        vector_list = [list(vector) for vector in vectors]
        payload_list = [dict(payload) for payload in payloads]
        id_list = list(ids) if ids is not None else None

        if len(vector_list) != len(payload_list):
            raise ValueError("Vectors and payloads length mismatch")

        if id_list is not None and len(id_list) != len(vector_list):
            raise ValueError("IDs count must match vectors count")

        all_points = []
        for idx, vector in enumerate(vector_list):
            payload = payload_list[idx]
            point_id = id_list[idx] if id_list is not None else None
            if isinstance(point_id, str):
                try:
                    uuid.UUID(point_id)
                except Exception:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))
            point = qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            all_points.append(point)

        for start in range(0, len(all_points), batch_size):
            chunk = all_points[start : start + batch_size]
            await self._client.upsert(collection_name=self.collection, points=chunk)

    async def ensure_collection(
        self,
        vector_size: int,
        *,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
        shard_number: int | None = None,
    ) -> None:
        """Create the collection if it does not exist yet."""

        try:
            await self._client.get_collection(self.collection)
            return
        except Exception:  # noqa: BLE001
            logger.info("Creating Qdrant collection '%s'", self.collection)

        params = qmodels.VectorParams(size=vector_size, distance=distance)
        await self._client.create_collection(
            collection_name=self.collection,
            vectors_config=params,
            shard_number=shard_number,
        )

    async def close(self) -> None:
        await self._client.close()
