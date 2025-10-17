from __future__ import annotations

from src.core.garant_api import (
    GarantAncestor,
    GarantDocument,
    GarantSearchResult,
    GarantSnippet,
    format_search_results,
)


def test_format_search_results_builds_human_readable_context() -> None:
    document = GarantDocument(topic=123456, name="Постановление ВС РФ", url="/#/document/123456")
    snippet = GarantSnippet(
        entry=1000,
        relevance=0.87,
        ancestors=[
            GarantAncestor(entry=1, title="Часть первая"),
            GarantAncestor(entry=10, title="Раздел I. Общие положения"),
            GarantAncestor(entry=100, title="Глава 1. Основные начала"),
            GarantAncestor(entry=1000, title="Статья 2. Основные принципы"),
        ],
    )

    formatted = format_search_results(
        [GarantSearchResult(document=document, snippets=[snippet])],
        document_base_url="https://d.garant.ru",
    )

    assert "ГАРАНТ" in formatted
    assert "Постановление ВС РФ" in formatted
    assert "entry 1000" in formatted
    assert "rlv 0.87" in formatted
    assert "https://d.garant.ru/#/document/123456" in formatted


def test_format_search_results_handles_missing_snippets() -> None:
    doc = GarantDocument(topic=77, name="Документ без вхождений", url=None)
    formatted = format_search_results([GarantSearchResult(document=doc, snippets=[])])

    assert "Документ без вхождений" in formatted
    assert "topic: 77" in formatted


def test_document_absolute_url_builds_from_relative_path() -> None:
    doc = GarantDocument(topic=1, name="Тест", url="/#/document/1")
    assert (
        doc.absolute_url("https://d.garant.ru")
        == "https://d.garant.ru/#/document/1"
    )

    doc_direct = GarantDocument(topic=1, name="Тест", url="https://example.org/path")
    assert doc_direct.absolute_url("https://d.garant.ru") == "https://example.org/path"
