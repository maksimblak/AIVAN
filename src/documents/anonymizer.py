"""
Модуль обезличивания (анонимизации) документов
Удаление персональных данных из документов для безопасного обмена.

Особенности:
- Один проход по тексту с единым комбинированным регэкспом (без «двойных замен»).
- Поддержка: ФИО, телефоны (междунар.), email (в т.ч. unicode \w), адреса (RU-маркеры),
  документы РФ (паспорт 4+6, СНИЛС с чексуммой, ИНН 10/12 с чексуммой), банковские
  реквизиты (р/с, БИК), карты (Luhn), IBAN (mod-97), даты рождения dd.mm.yyyy.
- Режимы: replace / mask (сохранение формата для телефонов/email) / remove / pseudonym.
- Псевдонимы детерминированы (HMAC-SHA256) и стабильны между запусками при ANON_SECRET.
- Отчёт: статистика по типам, список обработанных фрагментов со сниппетами и координатами.
"""

from __future__ import annotations

import base64 as _b64
import hmac
import hashlib as _hash
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)


# ---------------------------- ВАЛИДАЦИИ / УТИЛИТЫ ----------------------------

def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _luhn_ok(digits_only: str) -> bool:
    """Проверка Luhn для банковских карт (13–19 цифр)."""
    if not digits_only or not digits_only.isdigit():
        return False
    total = 0
    rev = digits_only[::-1]
    for i, ch in enumerate(rev):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _snils_ok(snils: str) -> bool:
    """СНИЛС: XXX-XXX-XXX YY (YY — контрольная сумма)."""
    d = _digits(snils)
    if len(d) != 11:
        return False
    num, cs = d[:9], int(d[9:])
    s = sum((9 - i) * int(num[i]) for i in range(9))
    if s < 100:
        chk = s
    elif s in (100, 101):
        chk = 0
    else:
        chk = s % 101
        if chk == 100:
            chk = 0
    return chk == cs


def _inn_ok(inn: str) -> bool:
    """ИНН РФ: 10 (юр) или 12 (физ), с контрольными цифрами."""
    d = _digits(inn)
    if len(d) == 10:
        k = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        chk = sum(int(d[i]) * k[i] for i in range(9)) % 11 % 10
        return chk == int(d[9])
    if len(d) == 12:
        k1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        k2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        c1 = sum(int(d[i]) * k1[i] for i in range(11)) % 11 % 10
        c2 = sum(int(d[i]) * k2[i] for i in range(12)) % 11 % 10
        return c1 == int(d[10]) and c2 == int(d[11])
    return False


def _iban_ok(iban: str) -> bool:
    """IBAN mod-97 (упрощённая проверка формата + контроль)."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    s = s[4:] + s[:4]
    num = []
    for ch in s:
        if ch.isdigit():
            num.append(ch)
        else:
            num.append(str(ord(ch) - 55))
    n = "".join(num)
    rem = 0
    for i in range(0, len(n), 9):
        rem = int(str(rem) + n[i:i + 9]) % 97
    return rem == 1


def _mask_preserve(s: str) -> str:
    """Маска, сохраняющая разделители (буквы/цифры → *)."""
    return "".join("*" if ch.isalnum() else ch for ch in s)


def _mask_phone(s: str) -> str:
    """Маска телефона: сохраняем формат и последние 4 цифры."""
    digits = _digits(s)
    if not digits:
        return _mask_preserve(s)
    # Сколько звёздочек до последних 4 цифр
    to_mask = max(0, len(digits) - 4)
    out = []
    masked = 0
    kept = 0
    for ch in s:
        if ch.isdigit():
            if masked < to_mask:
                out.append("*")
                masked += 1
            else:
                # сохраняем последние 4 цифры в исходном порядке
                out.append(digits[to_mask + kept])
                kept += 1
        else:
            out.append(ch)
    return "".join(out)


def _mask_email(s: str) -> str:
    """Маска email: скрываем локальную часть, домен сохраняем."""
    if "@" not in s:
        return _mask_preserve(s)
    local, domain = s.split("@", 1)
    if len(local) <= 2:
        mlocal = "*" * len(local)
    else:
        mlocal = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{mlocal}@{domain}"


def _pseudo_id(kind: str, original: str, secret: bytes) -> str:
    """Детерминированный короткий псевдоним по HMAC-SHA256 (6 символов base64url)."""
    digest = hmac.new(secret, f"{kind}:{original}".encode("utf-8"), _hash.sha256).digest()
    code = _b64.urlsafe_b64encode(digest[:4]).decode("ascii").rstrip("=")
    return code.upper()


# ------------------------------- ПАТТЕРНЫ -------------------------------

@dataclass
class PatternSpec:
    kind: str
    pattern: str
    validate: Callable[[str], bool] | None = None  # опциональная доп. проверка


# ------------------------------ ОСНОВНОЙ КЛАСС ------------------------------

class DocumentAnonymizer(DocumentProcessor):
    """Класс для обезличивания персональных данных в документах"""

    def __init__(self):
        super().__init__(name="DocumentAnonymizer", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.anonymization_map: dict[str, str] = {}

        # Набор паттернов с минимизацией фолс-позитивов
        self._specs: List[PatternSpec] = [
            # ФИО: 2–3 слова, каждое ≥2 символов (отсечь «И.П.»)
            PatternSpec("names", r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b"),
            # Телефоны: международные/локальные, суммарно 10–15 цифр
            PatternSpec(
                "phones",
                r"(?:\+?\d[\d\-\s().]{6,}\d)",
                validate=lambda s: 10 <= len(_digits(s)) <= 15,
            ),
            # Email: поддерживаем unicode \w в локальной части
            PatternSpec("emails", r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-zА-Яа-яЁё]{2,24}\b"),
            # Адреса: явные маркеры
            PatternSpec(
                "addresses",
                r"\b(?:г\.\s*[А-ЯЁ][а-яё\- ]+|ул\.\s*[А-ЯЁ][а-яё\- ]+|просп\.?\s*[А-ЯЁ][а-яё\- ]+|"
                r"пр-кт\.?\s*[А-ЯЁ][а-яё\- ]+|пер\.\s*[А-ЯЁ][а-яё\- ]+|дом\s*\d+\w*|д\.\s*\d+\w*)\b",
            ),
            # Документы РФ: паспорт 4+6, СНИЛС, ИНН 10/12 (с проверкой)
            PatternSpec(
                "documents",
                r"\b(?:\d{4}\s?\d{6}|\d{3}-?\d{3}-?\d{3}\s?\d{2}|\d{10}|\d{12})\b",
                validate=lambda s: (
                    _snils_ok(s) or _inn_ok(s) or re.fullmatch(r"\d{4}\s?\d{6}", s) is not None
                ),
            ),
            # Банковские реквизиты: р/с 20, БИК 9, карты 13–19 (Luhn)
            PatternSpec(
                "bank_details",
                r"\b(?:\d[\d\-\s]{11,}\d|\d{9}|\d{20})\b",
                validate=lambda s: (
                    len(_digits(s)) == 20
                    or len(_digits(s)) == 9
                    or (13 <= len(_digits(s)) <= 19 and _luhn_ok(_digits(s)))
                ),
            ),
            # IBAN (международные счета)
            PatternSpec("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", validate=_iban_ok),
            # Даты рождения (простая форма)
            PatternSpec("dates", r"\b(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\.(19\d{2}|20\d{2})\b"),
        ]

    # --------------------------------- API ---------------------------------

    async def process(
        self,
        file_path: str | Path,
        anonymization_mode: str = "replace",  # replace | mask | remove | pseudonym
        exclude_types: List[str] | None = None,
        **kwargs,
    ) -> DocumentResult:
        """
        Обезличивание документа.
        """
        # 1) Достаём текст
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # 2) Готовим исключения/карту
        self.anonymization_map = {}
        exclude_set = {item.lower() for item in (exclude_types or [])}

        # 3) Анонимизация
        anonymized_text, report = self._anonymize_text(cleaned_text, anonymization_mode, exclude_set)
        report["excluded_types"] = sorted(exclude_set) if exclude_set else []

        # 4) Ответ
        result_data = {
            "anonymized_text": anonymized_text,
            "anonymization_report": report,
            "anonymization_map": self.anonymization_map.copy(),
            "original_file": str(file_path),
            "mode": anonymization_mode,
        }
        return DocumentResult.success_result(
            data=result_data, message="Обезличивание документа успешно завершено"
        )

    # ----------------------------- ЯДРО ЛОГИКИ -----------------------------

    def _anonymize_text(
        self, text: str, mode: str, exclude: set[str] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Один проход слева-направо с единым комбинированным паттерном.
        Исключённые типы пропускаются (не заменяются и не считаются).
        Перекрывающиеся совпадения не ломают индексы.
        """
        exclude = exclude or set()

        stats: Dict[str, int] = {
            "names": 0,
            "phones": 0,
            "emails": 0,
            "addresses": 0,
            "documents": 0,
            "bank_details": 0,
            "iban": 0,
            "dates": 0,
        }
        processed_items: List[Dict[str, Any]] = []

        # Включаем только не-исключённые спецификации
        active_specs: List[PatternSpec] = [s for s in self._specs if s.kind not in exclude]
        if not active_specs:
            return text, {"processed_items": [], "statistics": stats}

        # Комбинированный паттерн с именованными группами
        named_parts = [f"(?P<{s.kind}>{s.pattern})" for s in active_specs]
        combined = re.compile("|".join(named_parts), flags=re.IGNORECASE | re.UNICODE)

        out: List[str] = []
        idx = 0
        counts_by_type: Dict[str, int] = {}  # для нумерации [Тип-1], [Тип-2], ...

        for m in combined.finditer(text):
            start, end = m.start(), m.end()
            if start < idx:
                # Перекрытие с предыдущей заменой — пропускаем
                continue

            # Определяем, какой вид сработал
            kind = None
            for s in active_specs:
                if m.group(s.kind):
                    kind = s.kind
                    spec = s
                    break
            if not kind:
                continue

            original = m.group(0)

            # Доп. валидация (если есть)
            if spec.validate and not spec.validate(original):
                continue

            # Хвост до совпадения
            out.append(text[idx:start])

            # Замена
            replacement = self._get_replacement_deterministic(
                original=original,
                kind=kind,
                data_type=self._pretty_type(kind),
                mode=mode,
                counts_by_type=counts_by_type,
            )
            out.append(replacement)
            idx = end

            # Отчёт/сниппет
            left = max(0, start - 20)
            right = min(len(text), end + 20)
            processed_items.append(
                {
                    "type": kind,
                    "value": original,
                    "start": start,
                    "end": end,
                    "snippet": text[left:right],
                }
            )
            stats[kind] = stats.get(kind, 0) + 1

        out.append(text[idx:])
        return "".join(out), {"processed_items": processed_items, "statistics": stats}

    # ------------------------- ПОДМЕНА / ФОРМАТИРОВАНИЕ -------------------------

    @staticmethod
    def _pretty_type(kind: str) -> str:
        return {
            "names": "Лицо",
            "phones": "Телефон",
            "emails": "Email",
            "addresses": "Адрес",
            "documents": "Документ",
            "bank_details": "БанкРеквизит",
            "iban": "IBAN",
            "dates": "Дата",
        }.get(kind, "Данные")

    def _get_replacement_deterministic(
        self,
        *,
        original: str,
        kind: str,
        data_type: str,
        mode: str,
        counts_by_type: Dict[str, int],
    ) -> str:
        """
        remove   → [УДАЛЕНО]
        mask     → сохраняем формат (телефон/email — спец. маска)
        replace  → [Тип-N] (инкремент по типу)
        pseudonym→ [Тип~HMAC] (детерминированный по секрету ANON_SECRET)
        """
        if original in self.anonymization_map:
            return self.anonymization_map[original]

        if mode == "remove":
            replacement = "[УДАЛЕНО]"
        elif mode == "mask":
            if kind == "phones":
                replacement = _mask_phone(original)
            elif kind == "emails":
                replacement = _mask_email(original)
            else:
                replacement = _mask_preserve(original)
        elif mode == "pseudonym":
            secret_env = os.getenv("ANON_SECRET", "")
            secret = secret_env.encode("utf-8") if secret_env else b"default-anon-secret"
            code = _pseudo_id(kind, original, secret)
            replacement = f"[{data_type}~{code}]"
        else:  # replace
            n = counts_by_type.get(data_type, 0) + 1
            counts_by_type[data_type] = n
            replacement = f"[{data_type}-{n}]"

        self.anonymization_map[original] = replacement
        return replacement

    # -------------------------------- ВОССТАНОВЛЕНИЕ ----------------------------

    def restore_data(
        self, anonymized_text: str, restoration_key: Dict[str, str] | None = None
    ) -> str:
        """
        Восстановление данных из обезличенного текста.
        Важно: сортируем ключи по длине убыв., чтобы не частично «раскрасить» метки.
        """
        if not restoration_key:
            restoration_key = {v: k for k, v in self.anonymization_map.items()}

        restored_text = anonymized_text
        for key in sorted(restoration_key.keys(), key=len, reverse=True):
            restored_text = restored_text.replace(key, restoration_key[key])

        return restored_text
