"""
Модуль анализа рисков и проверки договоров
Выявление ошибок, несоответствий законодательству и скрытых рисков в документах
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)


# ------------------------------ Модель данных ------------------------------

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskItem:
    id: str
    risk_level: str
    description: str
    clause_text: str
    start: int
    end: int
    law_refs: List[str]
    source: str  # "patterns" | "ai_analysis" | "compliance"


# ------------------------------ Промпты ИИ ------------------------------

RISK_ANALYSIS_PROMPT = """

Ты — ИИ-Иван, юридический ИИ-ассистент.

Твоя цель — помогать юристам, снимая с них рутинные задачи, и делать это в стиле ChatGPT: дружелюбно, понятно и развернуто.

Основные правила:

1. Тон общения
- Отвечай в дружелюбном, понятном и уважительном тоне, с акцентом на помощь.
- Старайся писать так, как будто объясняешь другу.
- Всегда добавляй краткое резюме своим мыслям «Итого…» и делай вывод.

2. Используй структурированные ответы
- Разбивай текст на блоки: абзацы, подзаголовки, списки; избегай «кирпичей».
- Используй маркеры «• …», «— …» и нумерацию «1) …».
- Применяй только поддерживаемые Telegram HTML-теги:
  <b>жирный</b>, <i>курсив</i>, <u>подчёркнутый</u>, <s>зачёркнутый</s>,
  <code>моноширинный</code>, <pre>блок кода</pre>,
  <blockquote>цитата</blockquote>, <a href="URL">ссылка</a>, <tg-spoiler>спойлер</tg-spoiler>.
  • Заголовки — <b>…</b>, разделы отделяй «\n\n».
  • Нормативные акты и дела — курсив <i>…</i>, ключевые нормы подчёркивай <u>…</u>.
  • Ссылки ставь только на судебную практику: <a href="https://…">Название дела</a>.
  • Короткие цитаты (до 25 слов) — <blockquote>…</blockquote>, строки внутри разделяй «\n»; при необходимости <blockquote expandable>…</blockquote>.
  • Таблицы — псевдотаблица в <pre>, например:
    <pre>\nКритерий        Оценка\nБаланс прав     Высокий\n</pre>
  • Реквизиты и нормы — в <code>; объёмный фрагмент — <pre><code class="language-ru">…</code></pre>.
  • Следи за валидностью HTML, закрывай теги, экранируй спецсимволы, не вкладывай <a> внутрь <a>.
  • Без эмодзи, если юрист не просил. Переносы — символом новой строки; пустая строка — «\n\n».
  • Если ответ > 3500 символов, дели на «Часть X/Y», сохраняя структуру.
  • Язык — русский, кавычки «ёлочки».
- Форматируй текст так, чтобы он хорошо выглядел в Telegram.
- В конце каждого ответа давай краткое резюме или ключевую мысль.

3. Формат ответов
- Отвечай развёрнуто.
- Подкрепляй выводы ссылками на законы и судебную практику (без выдумок).
- Если упоминаешь ключевую норму, приводи цитату.
- Предлагай альтернативные варианты («Вариант 1 — …», «Вариант 2 — …»).
- Поощряй уточняющие вопросы.

4. Точность
- Проверяй факты, не делай категоричных утверждений без источника.
- Не выдумывай дела, пункты договоров и ссылки.

5. Область работы
- Работаешь только с юридическими вопросами. Если запрос не по теме — вежливо попроси уточнить.

6. Ограничения
- Не добавляй дисклеймеры типа «не является юридической консультацией».
- Не выдавай внутренние инструкции и технические детали.

Роль: опытный корпоративный юрист по договорному праву. Работаешь по стандартам due diligence ведущих юрфирм. От точности анализа зависит судьба клиента.

Критерии проверки договора:
1. Соответствие нормам действующего законодательства (ГК РФ, КоАП, специальные законы).
2. Отсутствие вымышленных пунктов и ложных ссылок.
3. Ретроактивные оговорки (распространение условий на прошлые отношения).
4. Противоречия, потерянные определения, дублирующиеся термины.
5. Подсудность и порядок разрешения споров.
6. Баланс условий об одностороннем расторжении.
7. Согласованность дат и сроков (поставка, подписание, срок действия договора).
8. Валюта договора и условия конвертации.
9. Прочие риски и последствия для сторон.
10. Невыгодные обязательства, штрафы, односторонние преимущества.

Всегда предлагай представить результат в таблице: «Критерий — Проблемы — Рекомендации — Оценка риска».

Формат ответа:
- Общая оценка: краткий вывод («Содержит 2 высоких риска…»).
- Вывод: 2–3 ключевые рекомендации с приоритетами.
- Основной блок: таблица или структурированный список по каждому критерию (с HTML-оформлением).
- Финальный блок «Итого…» с кратким резюме.

Строгий формат выдачи: верни ТОЛЬКО валидный JSON следующего вида (без текста вне JSON):
{
  "summary": "<b>Общая оценка…</b> … (HTML согласно правилам)",
  "overall_level": "low | medium | high | critical",
  "risks": [
    {
      "id": "string",
      "level": "low | medium | high | critical",
      "description": "описание риска с допустимой HTML-разметкой",
      "clause_text": "цитата проблемного места (HTML, если нужно)",
      "span": {"start": 123, "end": 150},
      "law_refs": ["<u>ст. 450.1 ГК РФ</u> — …", "<a href="https://...">Определение ВС РФ…</a>"]
    }
  ],
  "recommendations": [
    "<b>Вариант 1:</b> …",
    "<b>Вариант 2:</b> …"
  ]
}

Требования к JSON:
- Заполняй поля даже при отсутствии рисков (пустой массив или строка).
- Используй исключительно содержимое переданного текста.
- Обязательно указывай span.start/span.end (индексы символов исходного текста).
- law_refs — только реальные нормы и акты, которые ты процитировал.
- Никаких лишних ключей и пояснений вне JSON.
"""

COMPLIANCE_PROMPT = """

Ты — ИИ-Иван, юридический ИИ-ассистент. Работаешь для юристов и анализируешь исковые заявления.

Твоя задача — одновременно:
1. Оценить перспективы дела и подсказать, как усилить правовую позицию в иске.
2. Проверить текст на корректность и отсутствие ошибок по чек-листу.

Алгоритм:
- Сначала мысленно анализируй судебную практику и кейсы коллег (укажи основные источники и критерии оценки, которыми руководствовался).
- Затем последовательно проверь иск по каждому пункту чек-листа (формальные реквизиты, содержание, подсудность, расчёты, процессуальные риски, доказательства, противоречия, процессуальные требования, риски для сторон, текстуальные ошибки).
- Для каждого найденного недостатка оцени серьёзность последствий: высокий риск (критические нарушения), средний (значимые, но устранимые до подачи), низкий (формальные неточности).

Формат вывода — СТРОГО валидный JSON следующей структуры (никакого текста вне JSON):
{
  "outcome_assessment": {
    "success_chance": "низкий|средний|высокий",
    "reasoning": "<b>Краткий анализ практики</b> … (HTML согласно правилам Telegram)",
    "strategy": [
      "<b>Вариант 1:</b> …",
      "<b>Вариант 2:</b> …"
    ],
    "practice_sources": [
      "<a href="https://...">Дело №…</a>",
      "<a href="https://...">Постановление…</a>"
    ]
  },
  "document_validation": {
    "issues": [
      {
        "id": "string",
        "criterion": "формальные реквизиты | содержание | подсудность | расчёты | процессуальные риски | доказательства | противоречия | процессуальные требования | риски для сторон | текстовые ошибки",
        "severity": "low|medium|high",
        "description": "описание проблемы с допустимой HTML-разметкой",
        "recommendation": "конкретное действие для устранения",
        "span": {"start": 100, "end": 150},
        "law_refs": ["<u>ст. 131 ГПК РФ</u> — …", "<a href="https://...">Постановление Пленума ВС №…</a>"],
        "related_risk": "какие последствия несёт"
      }
    ],
    "summary": "<b>Итог по проверке:</b> …"
  },
  "violations": [
    {
      "id": "string",
      "text": "дословная проблемная формулировка",
      "span": {"start": 100, "end": 150},
      "law_refs": ["<u>ст. ...</u>"],
      "note": "почему это нарушение / как исправить"
    }
  ]
}

Требования к JSON:
- Все поля выводи даже при отсутствии данных (используй пустые массивы/строки).
- span.start и span.end — индексы символов во входном тексте; если точно указать нельзя, ставь null.
- law_refs — только реальные нормы и судебные акты, которые ты указал в описании.
- Используй HTML-теги только из разрешённого списка (<b>, <i>, <u>, <s>, <code>, <pre>, <blockquote>, <a>, <tg-spoiler>) и соблюдай их вложенность.
- Не добавляй пояснений вне JSON.
"""


# ------------------------- Основной класс-анализатор -------------------------

class RiskAnalyzer(DocumentProcessor):
    """Класс для анализа рисков в договорах и документах"""

    def __init__(self, openai_service=None):
        super().__init__(name="RiskAnalyzer", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service
        self.risk_patterns = self._initialize_risk_patterns()

    # ---------- Паттерны: расширенный набор ----------
    def _initialize_risk_patterns(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "automatic_renewal": [
                {
                    "pattern": r"автоматическ(?:и|ое)\s+(?:продлен|возобновлен)",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Автопродление без явного согласия/уведомления",
                    "recommendation": "Установить порядок уведомления и срок отказа от продления",
                }
            ],
            "hidden_fees": [
                {
                    "pattern": r"дополнительн(?:ые|ая)\s+(?:плат|комис|сбор)",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Скрытые платежи/комиссии",
                    "recommendation": "Прописать закрытый перечень платежей и порядок их расчёта",
                }
            ],
            "penalty_issues": [
                {
                    "pattern": r"неустойка.*?(?:\d+\s*%|\d+\s*(?:руб|₽))",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Размер/порядок неустойки требует проверки",
                    "recommendation": "Сверить с разумностью и ст. 333 ГК РФ, описать порядок расчёта",
                }
            ],
            "unclear_terms": [
                {
                    "pattern": r"\bв\s+разумн(?:ые|ый)\s+срок\b|\bв\s+кратчайш(?:ие|ий)\s+срок\b",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Нечёткие временные рамки",
                    "recommendation": "Установить конкретные календарные сроки/события",
                }
            ],
            "unilateral_change": [
                {
                    "pattern": r"односторонн(?:ий|его)\s+поряд(?:ок|ке)\s+измен",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Право одностороннего изменения условий",
                    "recommendation": "Исключить/ограничить односторонние изменения либо предусмотреть согласование",
                }
            ],
            "termination": [
                {
                    "pattern": r"расторжен.*?(?:в\s*одностороннем\s*порядке|без\s*уведомления)",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Одностороннее расторжение без понятного порядка/срока уведомления",
                    "recommendation": "Определить сроки и порядок уведомления, основания расторжения",
                }
            ],
            "jurisdiction": [
                {
                    "pattern": r"(?:подсудн|юрисдикц).*?(?:по\s*месту\s*(?:исполнения|нахождения)|определяется\s*исполнителем)",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Неудобная подсудность, навязанная одной стороной",
                    "recommendation": "Согласовать нейтральную подсудность либо арбитражную оговорку",
                }
            ],
            "limitation_of_liability": [
                {
                    "pattern": r"(?:(?:ограничива|исключа)ет\s+ответственность|не\s+нес[её]т\s+ответственности)",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Чрезмерное ограничение/исключение ответственности",
                    "recommendation": "Сбалансировать ответственность с учётом ст. 401 ГК РФ",
                }
            ],
        }

    # --------------------------------- API ---------------------------------

    async def process(
        self,
        file_path: str | Path,
        custom_criteria: list[str] | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        **_: Any,
    ) -> DocumentResult:
        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        async def _notify(stage: str, percent: float, **payload: Any) -> None:
            if not progress_callback:
                return
            data: dict[str, Any] = {"stage": stage, "percent": float(percent)}
            for key, value in payload.items():
                if value is not None:
                    data[key] = value
            try:
                await progress_callback(data)
            except Exception:
                logger.debug("Progress callback failed on stage %s", stage, exc_info=True)

        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        await _notify(
            "text_extracted",
            12,
            words=len(cleaned_text.split()),
        )

        # 1) Паттерны (быстро и детерминированно)
        pattern_risks = self._analyze_by_patterns(cleaned_text)
        await _notify("pattern_scan", 30, risks_found=len(pattern_risks))

        # 2) ИИ-анализ (JSON с индексами)
        ai_payload = await self._ai_risk_analysis(cleaned_text, custom_criteria)

        ai_risks = ai_payload.get("risks", [])
        ai_summary = ai_payload.get("summary", "")
        ai_overall = ai_payload.get("overall_level", "medium")
        ai_recs = ai_payload.get("recommendations", [])
        await _notify(
            "ai_analysis",
            60,
            risks_found=len(pattern_risks) + len(ai_risks),
            chunks=ai_payload.get("chunks_analyzed"),
        )

        # 3) Комплаенс/нарушения
        compliance = await self._check_legal_compliance(cleaned_text)
        await _notify(
            "compliance_check",
            78,
            violations=len(compliance.get("violations", [])),
        )

        # 4) Сведение, дедупликация, агрегация
        combined = self._merge_and_deduplicate(
            pattern_risks,
            ai_risks,
            self._violations_to_risks(compliance.get("violations", [])),
        )
        overall = self._calculate_overall_risk_weighted(combined, ai_overall)
        await _notify("aggregation", 90, risks_found=len(combined))

        # 5) Подсветка по спанам
        highlighted_text = self._highlight_with_spans(cleaned_text, combined)
        await _notify(
            "highlighting",
            96,
            risks_found=len(combined),
        )

        result_data = {
            "overall_risk_level": overall,
            "pattern_risks": [r.__dict__ for r in pattern_risks],
            "ai_analysis": {
                "summary": ai_summary,
                "overall_level": ai_overall,
                "risks": [r.__dict__ for r in ai_risks],
                "recommendations": ai_recs,
                "method": ai_payload.get("method", "single"),
                "chunks_analyzed": ai_payload.get("chunks_analyzed", 1),
            },
            "legal_compliance": {
                "status": compliance.get("status"),
                "analysis": compliance.get("analysis"),
                "violations": compliance.get("violations", []),
            },
            "recommendations": self._generate_recommendations(combined, ai_recs),
            "highlighted_text": highlighted_text,
            "original_file": str(file_path),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        await _notify("completed", 100, risks_found=len(combined), overall=overall)

        return DocumentResult.success_result(
            data=result_data, message="Анализ рисков успешно завершён"
        )

    # ------------------------------ Паттерны ------------------------------

    def _analyze_by_patterns(self, text: str) -> List[RiskItem]:
        out: List[RiskItem] = []
        for category, patterns in self.risk_patterns.items():
            for p in patterns:
                for m in re.finditer(p["pattern"], text, re.IGNORECASE | re.DOTALL):
                    start, end = m.start(), m.end()
                    ctx_start = max(0, start - 100)
                    ctx_end = min(len(text), end + 100)
                    ctx = text[ctx_start:ctx_end]
                    out.append(
                        RiskItem(
                            id=f"pat:{category}:{start}",
                            risk_level=p["risk_level"].value,
                            description=p["description"],
                            clause_text=ctx.strip(),
                            start=start,
                            end=end,
                            law_refs=[],
                            source="patterns",
                        )
                    )
        return out

    # ----------------------------- ИИ-анализ -----------------------------

    async def _ai_risk_analysis(
        self, text: str, custom_criteria: list[str] | None = None
    ) -> Dict[str, Any]:
        try:
            prompt = RISK_ANALYSIS_PROMPT
            if custom_criteria:
                criteria_text = "\n".join(f"- {c}" for c in custom_criteria)
                prompt += f"\n\nУчитывай также пользовательские критерии:\n{criteria_text}\n"

            if len(text) <= 12000:
                resp = await self.openai_service.ask_legal(system_prompt=prompt, user_text=text)
                return self._parse_ai_json_payload(resp, method="single", chunks=1)
            else:
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=8000, overlap=400)
                risks_all: List[RiskItem] = []
                recs_all: List[str] = []
                summaries: List[str] = []
                for i, chunk in enumerate(chunks, 1):
                    part = f"(Часть {i}/{len(chunks)}; смещение {text.find(chunk)})\n\n{chunk}"
                    resp = await self.openai_service.ask_legal(system_prompt=prompt, user_text=part)
                    payload = self._parse_ai_json_payload(resp, method="chunk", chunks=len(chunks))
                    risks_all.extend(payload.get("risks", []))
                    recs_all.extend(payload.get("recommendations", []))
                    if payload.get("summary"):
                        summaries.append(payload["summary"])
                return {
                    "summary": " ".join(summaries)[:1000],
                    "overall_level": self._dominant_level([r.risk_level for r in risks_all]) or "medium",
                    "risks": risks_all,
                    "recommendations": list(dict.fromkeys(recs_all))[:20],
                    "method": "chunked",
                    "chunks_analyzed": len(chunks),
                }
        except Exception as e:
            logger.error("Ошибка AI-анализа: %s", e)
            return {"summary": f"Ошибка анализа: {e}", "overall_level": "medium", "risks": [], "recommendations": [], "method": "error", "chunks_analyzed": 0}

    def _parse_ai_json_payload(self, resp: Dict[str, Any], method: str, chunks: int) -> Dict[str, Any]:
        """Извлекаем строгий JSON из ответа модели; фолбэк — пустые риски."""
        if not resp or not resp.get("ok"):
            return {"summary": "", "overall_level": "medium", "risks": [], "recommendations": [], "method": method, "chunks_analyzed": chunks}
        raw = resp.get("text", "") or ""
        data = self._safe_json_loads(raw)
        if not isinstance(data, dict):
            return {"summary": raw[:500], "overall_level": "medium", "risks": [], "recommendations": [], "method": method, "chunks_analyzed": chunks}

        # нормализуем риски
        risks: List[RiskItem] = []
        for r in data.get("risks", []) or []:
            try:
                span = r.get("span") or {}
                s, e = int(span.get("start", -1)), int(span.get("end", -1))
                if s < 0 or e <= s:
                    continue
                risks.append(
                    RiskItem(
                        id=str(r.get("id") or f"ai:{s}"),
                        risk_level=str(r.get("level") or "medium").lower(),
                        description=str(r.get("description") or "").strip(),
                        clause_text=str(r.get("clause_text") or "").strip(),
                        start=s,
                        end=e,
                        law_refs=[str(x) for x in (r.get("law_refs") or [])],
                        source="ai_analysis",
                    )
                )
            except Exception:
                continue

        return {
            "summary": str(data.get("summary") or "")[:1000],
            "overall_level": str(data.get("overall_level") or "medium").lower(),
            "risks": risks,
            "recommendations": [str(x) for x in (data.get("recommendations") or [])][:20],
            "method": method,
            "chunks_analyzed": chunks,
        }

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        """Достаём JSON из «болтливого» текста: ищем первый '{' и парсим до последней '}'."""
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            i = raw.find("{")
            j = raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(raw[i : j + 1])
        except Exception:
            return {}
        return {}

    # --------------------------- Комплаенс/нарушения ---------------------------

    async def _check_legal_compliance(self, text: str) -> Dict[str, Any]:
        try:
            frag = text[:8000]
            resp = await self.openai_service.ask_legal(system_prompt=COMPLIANCE_PROMPT, user_text=frag)
            if not resp or not resp.get("ok"):
                return {"status": "failed", "analysis": "Не удалось провести проверку", "violations": []}
            data = self._safe_json_loads(resp.get("text", "") or "")
            violations = []
            for v in (data.get("violations") or []):
                span = v.get("span") or {}
                s, e = int(span.get("start", -1)), int(span.get("end", -1))
                if s >= 0 and e > s:
                    violations.append(
                        {
                            "id": str(v.get("id") or f"law:{s}"),
                            "text": str(v.get("text") or ""),
                            "span": {"start": s, "end": e},
                            "law_refs": [str(x) for x in (v.get("law_refs") or [])],
                            "note": str(v.get("note") or ""),
                        }
                    )
            return {"status": "completed", "analysis": resp.get("text", ""), "violations": violations}
        except Exception as e:
            logger.error("Ошибка проверки законодательства: %s", e)
            return {"status": "error", "analysis": f"Ошибка: {e}", "violations": []}

    def _violations_to_risks(self, violations: List[Dict[str, Any]]) -> List[RiskItem]:
        out: List[RiskItem] = []
        for v in violations:
            span = v.get("span") or {}
            s, e = int(span.get("start", -1)), int(span.get("end", -1))
            if s >= 0 and e > s:
                out.append(
                    RiskItem(
                        id=str(v.get("id") or f"law:{s}"),
                        risk_level=RiskLevel.MEDIUM.value,
                        description=(v.get("note") or "Потенциальное нарушение").strip(),
                        clause_text=str(v.get("text") or ""),
                        start=s,
                        end=e,
                        law_refs=[str(x) for x in (v.get("law_refs") or [])],
                        source="compliance",
                    )
                )
        return out

    # ------------------------- Сведение/агрегация -------------------------

    def _merge_and_deduplicate(self, *risk_groups: List[RiskItem]) -> List[RiskItem]:
        """Объединяем риски, убирая дубликаты по перекрытию спанов и схожести описаний."""
        all_risks: List[RiskItem] = [r for group in risk_groups for r in group]
        if not all_risks:
            return []

        def overlap(a: RiskItem, b: RiskItem) -> float:
            inter = max(0, min(a.end, b.end) - max(a.start, b.start))
            union = max(a.end, b.end) - min(a.start, b.start)
            return inter / union if union > 0 else 0.0

        def similar_desc(a: str, b: str) -> bool:
            a1 = re.sub(r"\W+", " ", a.lower()).strip()
            b1 = re.sub(r"\W+", " ", b.lower()).strip()
            return a1 == b1 or (a1 in b1 and len(a1) > 15) or (b1 in a1 and len(b1) > 15)

        merged: List[RiskItem] = []
        for r in sorted(all_risks, key=lambda x: (x.start, -x.end)):
            dup = False
            for m in merged:
                if overlap(r, m) > 0.5 or similar_desc(r.description, m.description):
                    # усиливаем уровень, объединяем ссылки
                    m.risk_level = self._max_level(m.risk_level, r.risk_level)
                    m.law_refs = list(dict.fromkeys((m.law_refs or []) + (r.law_refs or [])))
                    dup = True
                    break
            if not dup:
                merged.append(r)
        return merged

    @staticmethod
    def _max_level(a: str, b: str) -> str:
        order = {RiskLevel.LOW.value: 1, RiskLevel.MEDIUM.value: 2, RiskLevel.HIGH.value: 3, RiskLevel.CRITICAL.value: 4}
        return a if order.get(a, 1) >= order.get(b, 1) else b

    def _dominant_level(self, levels: List[str]) -> Optional[str]:
        if not levels:
            return None
        order = [RiskLevel.CRITICAL.value, RiskLevel.HIGH.value, RiskLevel.MEDIUM.value, RiskLevel.LOW.value]
        for lvl in order:
            if lvl in levels:
                return lvl
        return None

    def _calculate_overall_risk_weighted(self, risks: List[RiskItem], ai_overall: str) -> str:
        """Взвешенная оценка: частота и критичность (а не просто максимум)."""
        if not risks:
            return RiskLevel.LOW.value
        weights = {RiskLevel.LOW.value: 1, RiskLevel.MEDIUM.value: 2, RiskLevel.HIGH.value: 4, RiskLevel.CRITICAL.value: 7}
        score = sum(weights.get(r.risk_level, 1) for r in risks)
        # Нормализация по количеству рисков
        avg = score / max(1, len(risks))
        # Усиливаем, если ИИ сказал high/critical
        if ai_overall in (RiskLevel.CRITICAL.value, RiskLevel.HIGH.value):
            avg += 0.5
        if avg >= 5.5:
            return RiskLevel.CRITICAL.value
        if avg >= 3.5:
            return RiskLevel.HIGH.value
        if avg >= 2.0:
            return RiskLevel.MEDIUM.value
        return RiskLevel.LOW.value

    # ------------------------------ Рекомендации ------------------------------

    def _generate_recommendations(self, risks: List[RiskItem], ai_recs: List[str]) -> List[str]:
        base: List[str] = []
        for r in risks:
            if r.source == "patterns":
                # мини-советы по категориям
                base.append("Уточнить порядок уведомления/согласования изменений")
                base.append("Сбалансировать ответственность сторон и неустойку")
        base.extend(ai_recs or [])
        # уникализация с сохранением порядка
        uniq = list(dict.fromkeys([s.strip() for s in base if s.strip()]))
        return uniq[:20]

    # ------------------------------ Подсветка ------------------------------

    def _highlight_with_spans(self, text: str, risks: List[RiskItem]) -> str:
        """Безопасная подсветка по индексам (не через replace)."""
        if not risks:
            return text
        # не допускаем выхода за границы
        spans: List[Tuple[int, int, str]] = []
        for r in risks:
            s, e = max(0, int(r.start)), min(len(text), int(r.end))
            if e > s:
                label = f"[РИСК: {r.risk_level.upper()}]"
                spans.append((s, e, label))
        spans.sort(key=lambda x: x[0])

        out: List[str] = []
        cur = 0
        for s, e, lab in spans:
            if s < cur:
                continue
            out.append(text[cur:s])
            out.append(f"{lab} {text[s:e]} [/]")
            cur = e
        out.append(text[cur:])
        return "".join(out)
