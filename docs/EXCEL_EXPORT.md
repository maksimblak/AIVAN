# Judicial Practice Excel Export

`src/core/excel_export.py` generates the downloadable XLSX report that accompanies RAG answers and
case digests. The helper is reusable, so this document explains its inputs and how to integrate it
into other flows.

## Public API
```python
from src.core import excel_export

path = excel_export.build_practice_excel(
    summary_html=html_summary_from_model,
    fragments=rag_fragments,            # Sequence[PracticeFragment] or similar objects
    structured=structured_payload,      # Optional dict with "cases", "summary", etc.
    file_stub="rag_export_2024_10_25",  # Optional filename prefix
)
```

Returns a `pathlib.Path` pointing to a temp file (created via `_create_temp_path`). You are
responsible for sending it to the user (for example, via `bot.send_document`) and deleting it when
the transfer completes.

## Input formats
- `summary_html`: rich-text block that will be converted to plain text via `_html_to_plain`. Pass the
  assistant’s Markdown or HTML response here.
- `fragments`: typically `JudicialPracticeRAG.PracticeFragment` objects. The exporter looks at
  `fragment.header`, `fragment.excerpt`, and `fragment.match.metadata`. Plain dicts with the same
  keys also work.
- `structured`: optional mapping that mirrors the JSON schema produced by the RAG prompt. Supported
  keys:
  - `cases`: list of dicts with `title`, `case_number`, `url`, `facts`, `holding`, `norms`,
    `applicability`.
  - `summary`, `analysis`, `legal_basis`, `risks`, `disclaimer`.
- `file_stub`: slug for the output file. Unsafe characters are replaced with `_`.

## Sheet layout
The workbook contains two sheets:
1. **"��������"** (cases) — a table with five columns:
   - Name / case number / link (hyperlinked when a URL is available)
   - Facts or summary
   - Holding / decision outcome
   - Applicable norms
   - Applicability notes
   The exporter prefers the `structured["cases"]` payload. When fields are missing it falls back to
   fragment metadata automatically.
2. **"�����"** (overview) — a key-value table containing the executive summary, optional analysis,
   list of legal bases (combined into a newline-separated block), known risks, and disclaimers.

All columns are auto-sized, text wrapping is enabled, and bold headers are applied. You can adjust
widths or localization by editing `ws_cases` and `ws_overview` in `src/core/excel_export.py`.

## Integration points
- Document handlers in `src/core/bot_app/documents.py` call `build_practice_excel` when a user asks
  for supporting files. Reuse the same helper inside admin exports or background digests to keep file
  formats consistent.
- If you generate your own structured payload, make sure `structured["cases"]` is a list of mappings.
  Non-mapping entries are ignored silently.
- Use the returned `Path` directly with aiogram:
  ```python
  from aiogram.types import FSInputFile
  file = FSInputFile(path)
  await message.answer_document(file, caption="Материалы по делу")
  path.unlink(missing_ok=True)
  ```

## Troubleshooting
- `RuntimeError("библиотека openpyxl не установлена...")` — include `openpyxl` in your Poetry group
  or switch to the Docker image that already ships with it.
- Empty sheet — confirm that either `structured["cases"]` or `fragments` contains mappings with at
  least a title or URL. The exporter drops entries without identifiers to keep the sheet clean.
- Wrong language — the labels are localized in Russian by default. Update the header arrays in
  `build_practice_excel` to switch languages project-wide.
