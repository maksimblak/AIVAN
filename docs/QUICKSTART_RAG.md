# RAG Quickstart

Follow this checklist to enable Retrieval-Augmented Generation (RAG) for judicial practice snippets.
It assumes you are running Qdrant locally; see `docs/RAG_SETUP.md` for advanced tuning.

## 1. Start Qdrant locally
- **If you already use `docker compose` for the bot** (recommended):
  ```bash
  docker compose up -d qdrant
  ```
  Data is persisted in the named volume `qdrant_data` (host path `./qdrant_data` by default).
- **Standalone Docker run**:
  ```bash
  docker run -d \
    --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest
  ```
Qdrant Cloud also works—copy the HTTPS URL and API key.

## 2. Prepare a dataset
Create a JSONL file with at least the `text` field (see `data/judicial_cases_example.jsonl`):
```jsonl
{"text": "Суть дела ...", "case_number": "A40-12345/2024", "court": "АС г. Москвы", "date": "2024-05-10", "url": "https://..."}
```
Optional metadata such as `title`, `region`, `tags`, or `law_articles` will be stored inside Qdrant.

## 3. Configure `.env`
```env
RAG_ENABLED=true
RAG_QDRANT_URL=http://localhost:6333
RAG_COLLECTION=judicial_practice
RAG_TOP_K=6
RAG_SCORE_THRESHOLD=0.65
RAG_CONTEXT_CHAR_LIMIT=8000
RAG_EMBEDDING_MODEL=text-embedding-3-large
```
If you use Qdrant Cloud, set `RAG_QDRANT_API_KEY` and switch to HTTPS.

## 4. Load documents
```bash
poetry run python scripts/load_judicial_practice.py --input data/judicial_cases_example.jsonl
```
The loader embeds texts with OpenAI, ensures the collection exists, and upserts vectors in batches.
You can re-run the script; upserts are idempotent when `case_number` stays stable.

## 5. Verify
- Run `poetry run telegram-legal-bot` and ask a question that references your dataset. The bot will
  mention “[case N] ...” blocks if RAG context was attached.
- Tail the logs for `Judicial RAG search failed` or `Cache HIT for ask_legal` entries.
- Optionally call `poetry run python scripts/test_rag.py --question "..."` to exercise the pipeline
  without Telegram.

## 6. Troubleshooting
- Ensure `OPENAI_API_KEY` is present; the loader and runtime both create embeddings.
- Inspect Qdrant collections via `curl http://localhost:6333/collections`.
- Lower `RAG_SCORE_THRESHOLD` if no snippets are returned; reduce `RAG_TOP_K` if prompts become too
  long.
- Use `RAG_CONTEXT_CHAR_LIMIT` to cap the final prompt size; when set to `0` the limit is disabled.

Once the quickstart works, move on to `docs/RAG_SETUP.md` for schema design, filtering, and
production hardening tips.
