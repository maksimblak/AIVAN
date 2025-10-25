from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

logger = logging.getLogger(__name__)


class GarantAPIError(RuntimeError):
    """Raised when interaction with the Garant API fails."""


@dataclass(slots=True)
class GarantAncestor:
    entry: int
    title: str = ""


@dataclass(slots=True)
class GarantSnippet:
    entry: int
    relevance: float | None = None
    ancestors: list[GarantAncestor] = field(default_factory=list)

    def formatted_path(self, *, max_length: int = 140) -> str:
        titles = [ancestor.title.strip() for ancestor in self.ancestors if ancestor.title]
        path = " > ".join(titles)
        if len(path) > max_length:
            return path[: max_length - 3].rstrip() + "..."
        return path


@dataclass(slots=True)
class GarantDocument:
    topic: int
    name: str
    url: str | None = None

    def absolute_url(self, document_base_url: str | None) -> str | None:
        raw = (self.url or "").strip()
        if not raw:
            return None
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        base = (document_base_url or "").strip().rstrip("/")
        if not base:
            return raw
        if raw.startswith("/"):
            return f"{base}{raw}"
        return f"{base}/{raw}"


@dataclass(slots=True)
class GarantSearchResult:
    document: GarantDocument
    snippets: list[GarantSnippet] = field(default_factory=list)


@dataclass(slots=True)
class GarantReference:
    topic: int | None
    name: str
    url: str | None = None

    def absolute_url(self, document_base_url: str | None) -> str | None:
        doc = GarantDocument(topic=self.topic or 0, name=self.name, url=self.url)
        return doc.absolute_url(document_base_url)


@dataclass(slots=True)
class GarantSutyazhnikResult:
    kind: str
    norms: list[GarantReference] = field(default_factory=list)
    courts: list[GarantReference] = field(default_factory=list)


class GarantAPIClient:
    """Thin async wrapper around the Garant public API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout: float = 15.0,
        default_env: str | None = None,
        result_limit: int = 3,
        snippet_limit: int = 2,
        document_base_url: str | None = None,
        use_query_language: bool = True,
        verify_ssl: bool = True,
        sutyazhnik_enabled: bool = True,
        sutyazhnik_kinds: Sequence[str] | None = None,
        sutyazhnik_count: int = 5,
        log_debug: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip() if token else None
        env = (default_env or "").strip().lower()
        self._default_env = env if env in {"internet", "arbitr"} else "internet"
        self._result_limit = max(1, result_limit)
        self._snippet_limit = max(0, snippet_limit)
        self._document_base_url = document_base_url
        self._use_query_language = use_query_language
        self._sutyazhnik_enabled = sutyazhnik_enabled
        self._sutyazhnik_kinds = [kind for kind in (sutyazhnik_kinds or []) if kind]
        self._sutyazhnik_count = max(1, sutyazhnik_count)
        self._timeout = httpx.Timeout(timeout, connect=timeout, read=timeout)
        self._log_debug = bool(log_debug)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=headers,
            verify=verify_ssl,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    @property
    def document_base_url(self) -> str | None:
        return self._document_base_url

    @property
    def sutyazhnik_enabled(self) -> bool:
        return self._sutyazhnik_enabled and self.enabled

    async def close(self) -> None:
        await self._client.aclose()

    async def search_documents(
        self,
        query: str,
        *,
        env: str | None = None,
        count: int | None = None,
        page: int | None = None,
        is_query: bool | None = None,
        sort: int = 0,
        sort_order: int = 0,
    ) -> list[GarantDocument]:
        if not self.enabled:
            return []
        text = (query or "").strip()
        if not text:
            return []

        requested_env = (env or self._default_env or "internet").strip().lower()
        if requested_env not in {"internet", "arbitr"}:
            requested_env = "internet"
        page_number = page if isinstance(page, int) and page > 0 else 1
        desired_count = count if isinstance(count, int) and count >= 0 else self._result_limit
        if desired_count == 0:
            return []
        desired_count = min(desired_count, 50)

        payload: dict[str, Any] = {
            "text": text,
            "env": requested_env,
            "page": page_number,
            "sort": sort,
            "sortOrder": sort_order,
        }
        if is_query if is_query is not None else self._use_query_language:
            payload["isQuery"] = True

        try:
            response = await self._client.post("/v2/search", json=payload)
            response.raise_for_status()
            data = response.json()
            if self._log_debug:
                try:
                    logger.debug(
                        "[GARANT] /v2/search request: %s", json.dumps(payload, ensure_ascii=False)
                    )
                    logger.debug(
                        "[GARANT] /v2/search response: %s", json.dumps(data, ensure_ascii=False)
                    )
                except Exception:
                    logger.debug("[GARANT] /v2/search response parsed (non-serializable)")
        except httpx.TimeoutException as exc:  # noqa: PERF203
            raise GarantAPIError(f"Garant search request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise GarantAPIError(
                f"Garant search failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise GarantAPIError(f"Garant search request failed: {exc}") from exc
        except ValueError as exc:
            raise GarantAPIError(f"Invalid JSON from Garant search: {exc}") from exc

        documents_raw = data.get("documents") if isinstance(data, dict) else None
        documents: list[GarantDocument] = []
        if isinstance(documents_raw, Iterable):
            for item in documents_raw:
                doc = self._parse_document(item)
                if doc:
                    documents.append(doc)
        return documents[:desired_count]

    async def get_snippets(
        self,
        *,
        topic: int | str,
        text: str | None = None,
        correspondent: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[GarantSnippet]:
        if not self.enabled:
            return []
        payload: dict[str, Any] = {"topic": int(topic)}
        if text:
            payload["text"] = text.strip()
        if correspondent:
            payload["correspondent"] = correspondent

        if "text" in payload and "correspondent" in payload:
            payload.pop("correspondent")

        try:
            response = await self._client.post("/v2/snippets", json=payload)
            response.raise_for_status()
            data = response.json()
            if self._log_debug:
                try:
                    logger.debug(
                        "[GARANT] /v2/snippets request: %s", json.dumps(payload, ensure_ascii=False)
                    )
                    logger.debug(
                        "[GARANT] /v2/snippets response: %s", json.dumps(data, ensure_ascii=False)
                    )
                except Exception:
                    logger.debug("[GARANT] /v2/snippets response parsed (non-serializable)")
        except httpx.TimeoutException as exc:  # noqa: PERF203
            raise GarantAPIError(f"Garant snippets request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise GarantAPIError(
                f"Garant snippets failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise GarantAPIError(f"Garant snippets request failed: {exc}") from exc
        except ValueError as exc:
            raise GarantAPIError(f"Invalid JSON from Garant snippets: {exc}") from exc

        snippets_raw = data.get("snippets") if isinstance(data, dict) else None
        snippets: list[GarantSnippet] = []
        if isinstance(snippets_raw, Iterable):
            for item in snippets_raw:
                snippet = self._parse_snippet(item)
                if snippet:
                    snippets.append(snippet)

        trimmed_limit = limit if limit is not None else self._snippet_limit
        if trimmed_limit:
            return snippets[:trimmed_limit]
        return snippets

    async def search_with_snippets(
        self,
        query: str,
        *,
        env: str | None = None,
        count: int | None = None,
        snippet_limit: int | None = None,
    ) -> list[GarantSearchResult]:
        documents = await self.search_documents(query, env=env, count=count)
        if not documents:
            return []

        results: list[GarantSearchResult] = []
        limit = snippet_limit if snippet_limit is not None else self._snippet_limit

        for document in documents:
            try:
                snippets = await self.get_snippets(topic=document.topic, text=query, limit=limit)
            except GarantAPIError as exc:
                logger.warning("Failed to load snippets for topic %s: %s", document.topic, exc)
                snippets = []
            results.append(GarantSearchResult(document=document, snippets=snippets))
            await asyncio.sleep(0)  # allow event loop to switch tasks
        return results

    def format_results(self, results: Sequence[GarantSearchResult]) -> str:
        return format_search_results(results, document_base_url=self._document_base_url)

    @dataclass(slots=True)
    class LimitInfo:
        title: str
        value: int
        names: list[str] = field(default_factory=list)

    async def get_limits(self) -> list["GarantAPIClient.LimitInfo"]:
        """Fetch monthly API usage limits/remaining quotas from Garant.

        Returns a list of LimitInfo(title, value, names) on success, or an empty list on error.
        """
        try:
            response = await self._client.get("/v2/limits")
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:  # noqa: PERF203
            logger.warning("Garant limits request timed out: %s", exc)
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Garant limits failed with HTTP %s: %s",
                getattr(exc.response, "status_code", "?"),
                getattr(exc.response, "text", str(exc)),
            )
            return []
        except httpx.RequestError as exc:
            logger.warning("Garant limits request failed: %s", exc)
            return []
        except ValueError as exc:
            logger.warning("Invalid JSON from Garant limits: %s", exc)
            return []

        results: list[GarantAPIClient.LimitInfo] = []
        if isinstance(data, Iterable):
            for item in data:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                value_raw = item.get("value")
                try:
                    value = int(value_raw)
                except (TypeError, ValueError):
                    value = 0
                names_raw = item.get("names") or []
                names: list[str] = []
                if isinstance(names_raw, Iterable):
                    for nm in names_raw:
                        if isinstance(nm, str) and nm:
                            names.append(nm)
                if title:
                    results.append(GarantAPIClient.LimitInfo(title=title, value=value, names=names))
        return results

    def format_limits(
        self,
        limits: Sequence["GarantAPIClient.LimitInfo"],
        *,
        max_items: int = 6,
        warn_threshold: int = 20,
    ) -> str:
        """Красиво отформатировать лимиты для Telegram UI.

        Показывает заголовок, разделитель и список с индикаторами статуса:
        🔴 = 0, 🟡 = ≤ warn_threshold, 🟢 = остальное.
        """
        if not limits:
            return ""

        warn_threshold = max(0, int(warn_threshold))

        lines: list[str] = []
        lines.append("⚖️ <b>ГАРАНТ • Лимиты API</b>")
        lines.append("<code>────────────────────</code>")
        lines.append("")

        count = 0
        for item in limits:
            val = int(item.value)
            if val <= 0:
                badge = "🔴"
            elif val <= warn_threshold:
                badge = "🟡"
            else:
                badge = "🟢"
            title = (item.title or "").strip()
            lines.append(f"{badge} {title}: <b>{val}</b>")
            count += 1
            if max_items and count >= max_items:
                break

        return "\n".join(lines)

    async def sutyazhnik_search(
        self,
        text: str,
        *,
        kinds: Sequence[str] | None = None,
        count: int | None = None,
    ) -> list[GarantSutyazhnikResult]:
        if not self.sutyazhnik_enabled:
            return []
        query = (text or "").strip()
        if not query:
            return []

        payload: dict[str, Any] = {
            "text": query,
            "count": max(1, min(count or self._sutyazhnik_count, 1000)),
            "kind": list(kinds or self._sutyazhnik_kinds) or ["301", "302", "303"],
        }

        try:
            response = await self._client.post("/v2/sutyazhnik-search", json=payload)
            response.raise_for_status()
            data = response.json()
            if self._log_debug:
                try:
                    logger.debug(
                        "[GARANT] /v2/sutyazhnik-search request: %s",
                        json.dumps(payload, ensure_ascii=False),
                    )
                    logger.debug(
                        "[GARANT] /v2/sutyazhnik-search response: %s",
                        json.dumps(data, ensure_ascii=False),
                    )
                except Exception:
                    logger.debug(
                        "[GARANT] /v2/sutyazhnik-search response parsed (non-serializable)"
                    )
        except httpx.TimeoutException as exc:  # noqa: PERF203
            raise GarantAPIError(f"Garant sutyazhnik request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise GarantAPIError(
                f"Garant sutyazhnik failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise GarantAPIError(f"Garant sutyazhnik request failed: {exc}") from exc
        except ValueError as exc:
            raise GarantAPIError(f"Invalid JSON from Garant sutyazhnik: {exc}") from exc

        documents_raw = data.get("documents") if isinstance(data, dict) else None
        results: list[GarantSutyazhnikResult] = []
        if isinstance(documents_raw, Iterable):
            for item in documents_raw:
                parsed = self._parse_sutyazhnik_result(item)
                if parsed:
                    results.append(parsed)
        return results

    def format_sutyazhnik_results(
        self,
        results: Sequence[GarantSutyazhnikResult],
    ) -> str:
        return format_sutyazhnik_results(results, document_base_url=self._document_base_url)

    def _parse_document(self, item: Any) -> GarantDocument | None:
        if not isinstance(item, dict):
            return None
        topic_raw = item.get("topic")
        name = str(item.get("name") or "").strip()
        url = item.get("url")
        try:
            topic = int(topic_raw)
        except (TypeError, ValueError):
            return None
        if not name:
            name = f"Документ {topic}"
        return GarantDocument(topic=topic, name=name, url=url)

    def _parse_snippet(self, item: Any) -> GarantSnippet | None:
        if not isinstance(item, dict):
            return None
        entry_raw = item.get("entry")
        try:
            entry = int(entry_raw)
        except (TypeError, ValueError):
            return None
        relevance_raw = item.get("relevance")
        relevance: float | None = None
        if relevance_raw is not None:
            try:
                relevance = float(relevance_raw)
            except (TypeError, ValueError):
                relevance = None

        ancestors_raw = item.get("ancestors") or []
        ancestors: list[GarantAncestor] = []
        if isinstance(ancestors_raw, Iterable):
            for ancestor in ancestors_raw:
                if not isinstance(ancestor, dict):
                    continue
                ancestor_entry = ancestor.get("entry")
                title = str(ancestor.get("title") or "").strip()
                try:
                    ancestor_idx = int(ancestor_entry)
                except (TypeError, ValueError):
                    continue
                ancestors.append(GarantAncestor(entry=ancestor_idx, title=title))

        return GarantSnippet(entry=entry, relevance=relevance, ancestors=ancestors)

    def _parse_reference(self, item: Any) -> GarantReference | None:
        if not isinstance(item, dict):
            return None
        name = str(item.get("name") or "").strip()
        if not name:
            return None
        topic_raw = item.get("topic")
        try:
            topic = int(topic_raw) if topic_raw is not None else None
        except (TypeError, ValueError):
            topic = None
        url = item.get("url")
        return GarantReference(topic=topic, name=name, url=url)

    def _parse_sutyazhnik_result(self, item: Any) -> GarantSutyazhnikResult | None:
        if not isinstance(item, dict):
            return None
        kind = str(item.get("kind") or "").strip()
        norms_raw = item.get("norms") or []
        courts_raw = item.get("courts") or []

        norms: list[GarantReference] = []
        if isinstance(norms_raw, Iterable):
            for norm in norms_raw:
                parsed = self._parse_reference(norm)
                if parsed:
                    norms.append(parsed)

        courts: list[GarantReference] = []
        if isinstance(courts_raw, Iterable):
            for court in courts_raw:
                parsed = self._parse_reference(court)
                if parsed:
                    courts.append(parsed)

        if not kind and not norms and not courts:
            return None
        return GarantSutyazhnikResult(kind=kind, norms=norms, courts=courts)


def format_search_results(
    results: Sequence[GarantSearchResult],
    *,
    document_base_url: str | None = None,
) -> str:
    if not results:
        return ""

    lines: list[str] = ["[ГАРАНТ] Результаты поиска судебной практики:"]

    for idx, result in enumerate(results, start=1):
        doc = result.document
        title = doc.name.strip() or f"Документ {doc.topic}"
        lines.append(f"{idx}. {title}")
        url = doc.absolute_url(document_base_url)
        if url:
            lines.append(f"    URL: {url}")
        lines.append(f"    topic: {doc.topic}")

        if result.snippets:
            lines.append("    Вхождения:")
            for snippet in result.snippets:
                path = snippet.formatted_path()
                relevance = (
                    f"{snippet.relevance:.2f}" if isinstance(snippet.relevance, float) else None
                )
                parts = [f"entry {snippet.entry}"]
                if relevance:
                    parts.append(f"rlv {relevance}")
                if path:
                    parts.append(path)
                lines.append("      - " + " | ".join(parts))
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def format_sutyazhnik_results(
    results: Sequence[GarantSutyazhnikResult],
    *,
    document_base_url: str | None = None,
) -> str:
    if not results:
        return ""

    kind_titles = {
        "301": "Суды общей юрисдикции",
        "302": "Арбитражные суды",
        "303": "Суды по уголовным делам",
    }

    lines: list[str] = ["[ГАРАНТ Сутяжник] Анализ текста и релевантная практика:"]

    for idx, item in enumerate(results, start=1):
        kind_label = kind_titles.get(item.kind, item.kind or "Неизвестный тип")
        lines.append(f"{idx}. {kind_label}")

        if item.norms:
            lines.append("    Нормативные акты:")
            for ref in item.norms:
                url = ref.absolute_url(document_base_url)
                suffix = f" — {url}" if url else ""
                lines.append(f"      • {ref.name}{suffix}")

        if item.courts:
            lines.append("    Судебная практика:")
            for ref in item.courts:
                url = ref.absolute_url(document_base_url)
                suffix = f" — {url}" if url else ""
                lines.append(f"      • {ref.name}{suffix}")

        lines.append("")

    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
