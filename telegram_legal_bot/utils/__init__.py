# Ничего не импортируем на инициализации пакета, чтобы избежать ранних ошибок.
# При желании можно реэкспортнуть функции форматтера (они лёгкие).

from .message_formatter import build_legal_reply, chunk_markdown_v2, format_laws

__all__ = ["build_legal_reply", "chunk_markdown_v2", "format_laws"]
