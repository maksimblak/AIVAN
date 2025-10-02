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
from src.core.settings import AppSettings

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

def _is_passport(doc: str) -> bool:
    digits = _digits(doc)
    return len(digits) == 10




def _iban_ok(iban: str) -> bool:
    """IBAN mod-97 (упрощённая проверка формата + контроль)."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\\d{2}[A-Z0-9]{11,30}", s):
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
    label: str | None = None


# ------------------------------ ОСНОВНОЙ КЛАСС ------------------------------

class DocumentAnonymizer(DocumentProcessor):
    """Класс для обезличивания персональных данных в документах"""

    def __init__(self, settings: AppSettings | None = None):
        super().__init__(name="DocumentAnonymizer", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.anonymization_map: dict[str, str] = {}

        if settings is None:
            from src.core.app_context import get_settings  # avoid circular import

            settings = get_settings()
        self._settings = settings
        self._secret = settings.get_str("ANON_SECRET", "") or ""

        # Набор паттернов с минимизацией фолс-позитивов
        self._specs: List[PatternSpec] = [
            # ФИО: 2–3 слова, каждое ≥2 символов (отсечь «И.П.»)
            PatternSpec("names", r"\b[А-ЯЁ][а-яё]+(?:ов|ев|ёв|ин|ын|кий|цкий|ская|цкая|ова|ева|ёва|ина|ына|ский|ской)\b\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b"),
            # Телефоны: международные/локальные, суммарно 10–15 цифр
            PatternSpec(
                "phones",
                r"(?:\+?\\d[\\d\-\s().]{6,}\\d)",
                validate=lambda s: 10 <= len(_digits(s)) <= 15,
            ),
            # Email: поддерживаем unicode \w в локальной части
            PatternSpec("emails", r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-zА-Яа-яЁё]{2,24}\b"),
            # Адреса: явные маркеры
            PatternSpec(
                "addresses",
                r"\b(?:г\.\s*[А-ЯЁ][а-яё\- ]+|ул\.\s*[А-ЯЁ][а-яё\- ]+|просп\.?\s*[А-ЯЁ][а-яё\- ]+|"
                r"пр-кт\.?\s*[А-ЯЁ][а-яё\- ]+|пер\.\s*[А-ЯЁ][а-яё\- ]+|дом\s*\\d+\w*|д\.\s*\\d+\w*)\b",
            ),
            # Документы РФ: паспорт 4+6, СНИЛС, ИНН 10/12 (с проверкой)
            PatternSpec(
                "documents",
                r"\b(?:(?:серия\s*)?\d{4}(?:\s*№\s*|\s+)?\d{6}|\d{3}-?\d{3}-?\d{3}\s?\d{2}|\d{10}|\d{12})\b",
                validate=lambda s: (
                    _snils_ok(s) or _inn_ok(s) or _is_passport(s)
                ),
            ),
            # Банковские реквизиты: р/с 20, БИК 9, карты 13–19 (Luhn)
            PatternSpec(
                "bank_details",
                r"(?:(?:р/с|к/с|р\s?с|к\s?с|бик|p/c|k/c)[:\s]*)?(?:\d[\d\s-]{11,}\d|\d{9}|\d{20})",
                validate=lambda s: (
                    len(_digits(s)) == 20
                    or len(_digits(s)) == 9
                    or (13 <= len(_digits(s)) <= 19 and _luhn_ok(_digits(s)))
                ),
            ),
            # IBAN (международные счета)
            PatternSpec("iban", r"\b[A-Z]{2}\\d{2}[A-Z0-9]{11,30}\b", validate=_iban_ok),
            # Даты рождения (простая форма)
            PatternSpec("dates", r"\b(0[1-9]|[12]\\d|3[01])\.(0[1-9]|1[0-2])\.(19\\d{2}|20\д{2})\b"),
        ]

        label_overrides = {
            "names": "ФИО",
            "phones": "Телефон",
            "emails": "Email",
            "addresses": "Адрес",
            "documents": "Документ",
            "bank_details": "Банк. реквизит",
            "iban": "IBAN",
            "dates": "Дата",
        }
        for spec in self._specs:
            if spec.label is None and spec.kind in label_overrides:
                spec.label = label_overrides[spec.kind]

        self._specs.extend([
            PatternSpec("badge_numbers", r"\b(?:таб\.?|табельный)\s*(?:номер|№)\s*\\d{3,10}\b", label="Табельный номер"),
            PatternSpec("registration_numbers", r"\b(?:огрн(?:ип)?|грн|рег\.?\s*№?|окпо|оквэд|свид\.?|гос\.?рег\.?№?)\s*[:№-]*[a-z0-9\-]{5,25}\b", label="Регистрационный номер"),
            PatternSpec("domains", r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", label="Домен"),
            PatternSpec("urls", r"\bhttps?://[^\s<'\"]+", label="Ссылка"),
        ])

        self._base_specs: List[PatternSpec] = list(self._specs)

    @staticmethod
    def _normalize_pattern(pattern: str) -> str:
        match = re.match(r'^\(\?([aiLmsux]+)\)(.*)$', pattern, flags=re.DOTALL)
        if match:
            flags, rest = match.groups()
            return f'(?{flags}:{rest})'
        return pattern

    # --------------------------------- API ---------------------------------

    async def process(
        self,
        file_path: str | Path,
        anonymization_mode: str = "replace",  # replace | mask | remove | pseudonym
        exclude_types: List[str] | None = None,
        custom_patterns: List[dict[str, str]] | List[str] | None = None,
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
        custom_specs = self._prepare_custom_specs(custom_patterns)
        anonymized_text, report = self._anonymize_text(
            cleaned_text,
            anonymization_mode,
            exclude_set,
            custom_specs=custom_specs,
        )
        report["excluded_types"] = sorted(exclude_set) if exclude_set else []
        if custom_specs and "custom_patterns" not in report:
            report["custom_patterns"] = [
                {"kind": spec.kind, "label": spec.label} for spec in custom_specs
            ]

        # 4) Ответ
        result_data = {
            "anonymized_text": anonymized_text,
            "anonymization_report": report,
            "anonymization_map": self.anonymization_map.copy(),
            "original_file": str(file_path),
            "mode": anonymization_mode,
        }
        if custom_specs:
            result_data["applied_custom_patterns"] = [
                {"kind": spec.kind, "label": spec.label} for spec in custom_specs
            ]
        return DocumentResult.success_result(
            data=result_data, message="Обезличивание документа успешно завершено"
        )

    # ----------------------------- ЯДРО ЛОГИКИ -----------------------------

    def _anonymize_text(
        self,
        text: str,
        mode: str,
        exclude: set[str] | None = None,
        custom_specs: List[PatternSpec] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Один проход слева-направо с единым комбинированным паттерном.
        Пользовательские шаблоны подключаются динамически.
        """
        exclude = set(exclude or set())
        specs: List[PatternSpec] = list(self._base_specs)
        if custom_specs:
            specs.extend(custom_specs)

        exclude_custom_all = any(key in exclude for key in ("custom", "custom_patterns"))

        active_specs: List[PatternSpec] = []
        label_map: Dict[str, str] = {}
        for spec in specs:
            if spec.kind in exclude:
                continue
            if exclude_custom_all and spec.kind.startswith("custom_"):
                continue
            active_specs.append(spec)
            label_map[spec.kind] = spec.label or self._pretty_type(spec.kind)

        if not active_specs:
            return text, {"processed_items": [], "statistics": {}, "type_labels": {}}

        stats: Dict[str, int] = {kind: 0 for kind in label_map}
        processed_items: List[Dict[str, Any]] = []

        named_parts = [f"(?P<{s.kind}>{s.pattern})" for s in active_specs]
        combined = re.compile("|".join(named_parts), flags=re.IGNORECASE | re.UNICODE)

        out: List[str] = []
        idx = 0
        counts_by_type: Dict[str, int] = {}

        for match in combined.finditer(text):
            start, end = match.start(), match.end()
            if start < idx:
                continue

            spec = None
            for candidate in active_specs:
                if match.group(candidate.kind):
                    spec = candidate
                    break
            if spec is None:
                continue

            original = match.group(0)

            if spec.validate and not spec.validate(original):
                continue

            out.append(text[idx:start])

            data_type = label_map.get(spec.kind, self._pretty_type(spec.kind))
            replacement = self._get_replacement_deterministic(
                original=original,
                kind=spec.kind,
                data_type=data_type,
                mode=mode,
                counts_by_type=counts_by_type,
            )
            out.append(replacement)
            idx = end

            left = max(0, start - 20)
            right = min(len(text), end + 20)
            processed_items.append(
                {
                    "type": spec.kind,
                    "label": data_type,
                    "value": original,
                    "start": start,
                    "end": end,
                    "snippet": text[left:right],
                }
            )
            stats[spec.kind] = stats.get(spec.kind, 0) + 1

        out.append(text[idx:])
        report = {"processed_items": processed_items, "statistics": stats, "type_labels": label_map}
        if custom_specs:
            report["custom_patterns"] = [
                {"kind": spec.kind, "label": label_map.get(spec.kind, spec.label)}
                for spec in custom_specs
            ]
        return "".join(out), report

    def _prepare_custom_specs(
        self, custom_patterns: List[dict[str, str]] | List[str] | None
    ) -> List[PatternSpec]:
        specs: List[PatternSpec] = []
        if not custom_patterns:
            return specs

        used_kinds = {spec.kind for spec in self._base_specs}

        for idx, entry in enumerate(custom_patterns, start=1):
            pattern = ""
            label = ""
            if isinstance(entry, dict):
                pattern = str(entry.get("pattern", "")).strip()
                label = str(entry.get("name") or entry.get("label") or "").strip()
            else:
                pattern = str(entry).strip()

            if not pattern:
                continue

            normalized_pattern = self._normalize_pattern(pattern)
            try:
                re.compile(normalized_pattern, re.IGNORECASE | re.UNICODE)
            except re.error as exc:
                logger.warning("Skip custom pattern %s: %s", pattern, exc)
                continue

            slug_source = label or f"pattern_{idx}"
            slug = re.sub(r"[^a-z0-9]+", "_", slug_source.lower()).strip("_")
            if not slug:
                slug = f"pattern_{idx}"

            base_kind = f"custom_{slug}"
            kind = base_kind
            counter = 2
            while kind in used_kinds:
                kind = f"{base_kind}_{counter}"
                counter += 1
            used_kinds.add(kind)

            human_label = label or "Пользовательский шаблон"
            specs.append(PatternSpec(kind, normalized_pattern, label=human_label))

        return specs

    # ------------------------- ПОДМЕНА / ФОРМАТИРОВАНИЕ -------------------------

    @staticmethod
    def _pretty_type(kind: str) -> str:
        mapping = {
            "names": "Лицо",
            "phones": "Телефон",
            "emails": "Email",
            "addresses": "Адрес",
            "documents": "Документ",
            "bank_details": "БанкРеквизит",
            "iban": "IBAN",
            "dates": "Дата",
            "badge_numbers": "Табельный номер",
            "registration_numbers": "Регистрационный номер",
            "domains": "Домен",
            "urls": "Ссылка",
        }
        if kind.startswith("custom_"):
            return "Пользовательский шаблон"
        return mapping.get(kind, "Данные")

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
            secret_bytes = self._secret.encode("utf-8") if self._secret else b"default-anon-secret"
            secret = secret_bytes
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
