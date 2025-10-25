# RAG Setup Guide

This guide explains how Retrieval-Augmented Generation is wired inside AIVAN and how to operate it
in production. It complements the quick checklist in `docs/QUICKSTART_RAG.md`.

## Components
- `scripts/load_judicial_practice.py` - ingests JSONL cases, creates embeddings, and upserts them
  into Qdrant.
- `src/core/rag/embedding_service.py` - thin wrapper around OpenAI embeddings.
- `src/core/rag/vector_store.py` - Qdrant client (HTTP or gRPC) with helper methods for search,
  payload upserts, and collection management.
- `src/core/rag/judicial_rag.py` - orchestrates question embeddings, vector search, snippet
  formatting, and prompt truncation.
- `src/core/bot_app/openai_gateway.py` and `src/core/openai_service.py` - inject the concatenated
  context into the legal prompt if `JudicialPracticeRAG.enabled` is true.

```
question → EmbeddingService → Qdrant search → PracticeFragment list
         ↘ formatted “[case N] …” blocks ↘ OpenAI prompt builder
```

## Configuration matrix
| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_ENABLED` | Global on/off flag. | `false` |
| `RAG_QDRANT_URL` / `RAG_QDRANT_HOST`+`RAG_QDRANT_PORT` | Where to reach Qdrant. | `None` |
| `RAG_QDRANT_API_KEY` | Optional API key for managed clusters. | `None` |
| `RAG_QDRANT_GRPC` | Use gRPC instead of HTTP (`true`/`false`). | `false` |
| `RAG_COLLECTION` | Collection name to read/write. | `judicial_practice` |
| `RAG_TOP_K` | Max number of matches to request from Qdrant. | `6` |
| `RAG_SCORE_THRESHOLD` | Drop matches below this similarity score. | `None` |
| `RAG_SNIPPET_CHAR_LIMIT` | Trim each fragment to N characters. | `1200` |
| `RAG_CONTEXT_CHAR_LIMIT` | Hard cap on the combined context size (0 disables). | `6500` |
| `RAG_EMBEDDING_MODEL` | OpenAI embedding model used by both loader and runtime. | `text-embedding-3-large` |
| `RAG_QDRANT_TIMEOUT` | Seconds before requests time out. | `None` |

Qdrant credentials are read directly from `.env` via `AppSettings`, so you can swap clusters without
code changes.

## Dataset schema
The loader expects JSONL with a mandatory `text` field. Everything else becomes Qdrant payload
metadata and can be used for filtering later.

```jsonl
{
  "text": "Резолютивная часть ...",
  "case_number": "A40-12345/2024",
  "court": "АС г. Москвы",
  "date": "2024-05-10",
  "title": "Спор с контрагентом",
  "url": "https://...",
  "region": "Москва",
  "law_articles": ["ГК РФ 309", "ГК РФ 330"]
}
```

Inside `scripts/load_judicial_practice.py` you can extend the payload dictionary to include custom
fields, or derive IDs from your own primary keys instead of `case_number`.

## Operating procedures
1. **Initial load** - run the loader locally or inside CI with the same `.env` to populate the
   target cluster. Monitor stdout logs for embedding progress and upsert statistics.
2. **Delta updates** - rerun the loader with a JSONL file that only contains new/changed cases.
   Qdrant upserts reuse IDs, so existing records are replaced.
3. **Testing** - `scripts/test_rag.py --question "..."` prints the selected fragments and scores.
4. **Monitoring** - enable Prometheus metrics and alert on:
   - `rag_queries_total` vs `rag_fallback_total` (implemented in `src/core/openai_service.py`).
   - `Judicial RAG search failed` log entries.
   - Latency spikes from Qdrant (wrap the HTTP endpoint with your APM if needed).
5. **Cleanup** - use Qdrant’s native tools to snapshot or prune data. The project does not delete
   vectors automatically.

## Performance tips
- Lower `RAG_TOP_K` once you confirm the dataset quality; fewer fragments reduce prompt size and
  latency.
- Use `RAG_SCORE_THRESHOLD` (0.5-0.7) to suppress irrelevant matches for ambiguous questions.
- Set `RAG_CONTEXT_CHAR_LIMIT` to stay under the OpenAI token limit when documents are verbose.
- When texts are extremely long, consider pre-splitting them before running the loader so each
  fragment describes one logical paragraph.
- If Qdrant runs remotely, enable gRPC (`RAG_QDRANT_GRPC=true`) to reduce latency and bandwidth.

## Troubleshooting
- **No fragments returned** - verify embeddings were created (look for `vector_size` log), ensure the
  collection exists (`curl /collections/<name>`), and drop the score threshold temperature.
- **Gateway TypeError** - `JudicialPracticeRAG` gracefully disables itself when the DI container
  cannot provide Qdrant credentials, but you still need to remove `RAG_ENABLED=true` from `.env`.
- **Token limit errors** - reduce `RAG_TOP_K` or `RAG_CONTEXT_CHAR_LIMIT`; you can also shrink
  `RAG_SNIPPET_CHAR_LIMIT` if each case excerpt is long.
- **429 / quota issues** - the loader uses OpenAI embeddings for every record. Consider batching
  (`--batch-size`) or caching embeddings externally when refreshing large corpora.

With the above in place the bot automatically threads `[case N] …` excerpts into prompts and shows
links in its final response, giving users transparent citations for each answer.
