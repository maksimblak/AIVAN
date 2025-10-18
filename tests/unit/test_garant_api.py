from __future__ import annotations

from src.core.garant_api import (
    GarantAncestor,
    GarantDocument,
    GarantReference,
    GarantSearchResult,
    GarantSnippet,
    GarantSutyazhnikResult,
    format_search_results,
    format_sutyazhnik_results,
)


def test_format_search_results_builds_human_readable_context() -> None:
    document = GarantDocument(topic=123456, name="Test Document", url="/#/document/123456")
    snippet = GarantSnippet(
        entry=1000,
        relevance=0.87,
        ancestors=[
            GarantAncestor(entry=1, title="Part I"),
            GarantAncestor(entry=10, title="Section A"),
            GarantAncestor(entry=100, title="Chapter 1"),
            GarantAncestor(entry=1000, title="Article 2"),
        ],
    )

    formatted = format_search_results(
        [GarantSearchResult(document=document, snippets=[snippet])],
        document_base_url="https://d.garant.ru",
    )

    assert "[\u0413\u0410\u0420\u0410\u041d\u0422" in formatted
    assert "Test Document" in formatted
    assert "entry 1000" in formatted
    assert "rlv 0.87" in formatted
    assert "https://d.garant.ru/#/document/123456" in formatted


def test_format_search_results_handles_missing_snippets() -> None:
    doc = GarantDocument(topic=77, name="Document without snippets", url=None)
    formatted = format_search_results([GarantSearchResult(document=doc, snippets=[])])

    assert "Document without snippets" in formatted
    assert "topic: 77" in formatted


def test_document_absolute_url_builds_from_relative_path() -> None:
    doc = GarantDocument(topic=1, name="Test", url="/#/document/1")
    assert (
        doc.absolute_url("https://d.garant.ru")
        == "https://d.garant.ru/#/document/1"
    )

    doc_direct = GarantDocument(topic=1, name="Test", url="https://example.org/path")
    assert doc_direct.absolute_url("https://d.garant.ru") == "https://example.org/path"


def test_format_sutyazhnik_results_includes_links() -> None:
    results = [
        GarantSutyazhnikResult(
            kind="301",
            norms=[GarantReference(topic=123, name="Law reference", url="/#/document/123")],
            courts=[GarantReference(topic=456, name="Court case", url="/#/document/456/paragraph/7")],
        )
    ]

    formatted = format_sutyazhnik_results(results, document_base_url="https://d.garant.ru")

    assert "[\u0413\u0410\u0420\u0410\u041d\u0422 \u0421\u0443\u0442\u044f\u0436\u043d\u0438\u043a" in formatted
    assert "\u2022 Law reference \u2014 https://d.garant.ru/#/document/123" in formatted
    assert "\u2022 Court case \u2014 https://d.garant.ru/#/document/456/paragraph/7" in formatted


def test_format_sutyazhnik_results_handles_unknown_kind() -> None:
    formatted = format_sutyazhnik_results(
        [GarantSutyazhnikResult(kind="399", norms=[], courts=[])],
        document_base_url=None,
    )

    assert "399" in formatted
