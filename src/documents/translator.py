"""
РњРѕРґСѓР»СЊ РїРµСЂРµРІРѕРґР° РґРѕРєСѓРјРµРЅС‚РѕРІ
РџСЂРѕС„РµСЃСЃРёРѕРЅР°Р»СЊРЅС‹Р№ РїРµСЂРµРІРѕРґ РґРѕРєСѓРјРµРЅС‚РѕРІ СЃ СЃРѕС…СЂР°РЅРµРЅРёРµРј СЋСЂРёРґРёС‡РµСЃРєРѕР№ С‚РµСЂРјРёРЅРѕР»РѕРіРёРё
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = """
РўС‹ вЂ” РїСЂРѕС„РµСЃСЃРёРѕРЅР°Р»СЊРЅС‹Р№ РїРµСЂРµРІРѕРґС‡РёРє СЋСЂРёРґРёС‡РµСЃРєРёС… РґРѕРєСѓРјРµРЅС‚РѕРІ.

Р—РђР”РђР§Рђ: РџРµСЂРµРІРµРґРё РґРѕРєСѓРјРµРЅС‚ СЃ {source_lang} РЅР° {target_lang}, СЃРѕС…СЂР°РЅСЏСЏ:
- Р®СЂРёРґРёС‡РµСЃРєСѓСЋ С‚РµСЂРјРёРЅРѕР»РѕРіРёСЋ
- РЎС‚СЂСѓРєС‚СѓСЂСѓ РґРѕРєСѓРјРµРЅС‚Р°
- Р¤РѕСЂРјР°Р»СЊРЅС‹Р№ СЃС‚РёР»СЊ
- РўРѕС‡РЅРѕСЃС‚СЊ РїСЂР°РІРѕРІС‹С… РїРѕРЅСЏС‚РёР№

РћРЎРћР‘Р«Р• РўР Р•Р‘РћР’РђРќРРЇ:
- РСЃРїРѕР»СЊР·СѓР№ РїСЂРёРЅСЏС‚С‹Рµ РїРµСЂРµРІРѕРґС‹ СЋСЂРёРґРёС‡РµСЃРєРёС… С‚РµСЂРјРёРЅРѕРІ
- РЎРѕС…СЂР°РЅСЏР№ С„РѕСЂРјР°С‚РёСЂРѕРІР°РЅРёРµ (СЃРїРёСЃРєРё, РїСѓРЅРєС‚С‹, С‚Р°Р±Р»РёС†С‹)
- РђРґР°РїС‚РёСЂСѓР№ Рє РјРµСЃС‚РЅРѕРјСѓ Р·Р°РєРѕРЅРѕРґР°С‚РµР»СЊСЃС‚РІСѓ РіРґРµ РЅРµРѕР±С…РѕРґРёРјРѕ
- Р”РѕР±Р°РІР»СЏР№ РїРѕСЏСЃРЅРµРЅРёСЏ РІ СЃРєРѕР±РєР°С… РґР»СЏ СЃР»РѕР¶РЅС‹С… С‚РµСЂРјРёРЅРѕРІ

РЇР—Р«РљРћР’Р«Р• РџРђР Р«:
- Р СѓСЃСЃРєРёР№ в†” РђРЅРіР»РёР№СЃРєРёР№
- Р СѓСЃСЃРєРёР№ в†” РљРёС‚Р°Р№СЃРєРёР№
- Р СѓСЃСЃРєРёР№ в†” РќРµРјРµС†РєРёР№

РџРµСЂРµРІРѕРґРё С‚РѕР»СЊРєРѕ С‚РµРєСЃС‚ РґРѕРєСѓРјРµРЅС‚Р°, СЃРѕС…СЂР°РЅСЏСЏ РµРіРѕ СЃС‚СЂСѓРєС‚СѓСЂСѓ.
"""


class DocumentTranslator(DocumentProcessor):
    """РљР»Р°СЃСЃ РґР»СЏ РїРµСЂРµРІРѕРґР° РґРѕРєСѓРјРµРЅС‚РѕРІ"""

    def __init__(self, openai_service=None):
        super().__init__(name="DocumentTranslator", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service

        self.supported_languages = {
            "ru": "СЂСѓСЃСЃРєРёР№",
            "en": "Р°РЅРіР»РёР№СЃРєРёР№",
            "zh": "РєРёС‚Р°Р№СЃРєРёР№",
            "de": "РЅРµРјРµС†РєРёР№",
        }

    async def process(
        self, file_path: str | Path, source_lang: str = "ru", target_lang: str = "en", **kwargs
    ) -> DocumentResult:
        """
        РџРµСЂРµРІРѕРґ РґРѕРєСѓРјРµРЅС‚Р°

        Args:
            file_path: РїСѓС‚СЊ Рє С„Р°Р№Р»Сѓ
            source_lang: РёСЃС…РѕРґРЅС‹Р№ СЏР·С‹Рє (ru, en, zh, de)
            target_lang: С†РµР»РµРІРѕР№ СЏР·С‹Рє (ru, en, zh, de)
        """

        if not self.openai_service:
            raise ProcessingError(
                "OpenAI СЃРµСЂРІРёСЃ РЅРµ РёРЅРёС†РёР°Р»РёР·РёСЂРѕРІР°РЅ", "SERVICE_ERROR"
            )

        if (
            source_lang not in self.supported_languages
            or target_lang not in self.supported_languages
        ):
            raise ProcessingError(
                "РќРµРїРѕРґРґРµСЂР¶РёРІР°РµРјР°СЏ СЏР·С‹РєРѕРІР°СЏ РїР°СЂР°", "LANGUAGE_ERROR"
            )

        # РР·РІР»РµРєР°РµРј С‚РµРєСЃС‚ РёР· С„Р°Р№Р»Р°
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(
                f"РќРµ СѓРґР°Р»РѕСЃСЊ РёР·РІР»РµС‡СЊ С‚РµРєСЃС‚: {text}", "EXTRACTION_ERROR"
            )

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError(
                "Р”РѕРєСѓРјРµРЅС‚ РЅРµ СЃРѕРґРµСЂР¶РёС‚ С‚РµРєСЃС‚Р°", "EMPTY_DOCUMENT"
            )

        # РџРµСЂРµРІРѕРґРёРј РґРѕРєСѓРјРµРЅС‚
        translated_text, chunk_details = await self._translate_text(
            cleaned_text, source_lang, target_lang
        )

        result_data = {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "original_file": str(file_path),
            "chunk_details": chunk_details,
            "translation_metadata": {
                "original_length": len(cleaned_text),
                "translated_length": len(translated_text),
                "language_pair": f"{source_lang} -> {target_lang}",
                "chunks_processed": len(chunk_details),
            },
        }

        return DocumentResult.success_result(
            data=result_data,
            message=f"РџРµСЂРµРІРѕРґ СЃ {self.supported_languages[source_lang]} РЅР° {self.supported_languages[target_lang]} Р·Р°РІРµСЂС€РµРЅ",
        )

    async def _translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """РџРµСЂРµРІРѕРґ С‚РµРєСЃС‚Р° СЃ РїРѕРјРѕС‰СЊСЋ AI"""

        source_lang_name = self.supported_languages[source_lang]
        target_lang_name = self.supported_languages[target_lang]

        prompt = TRANSLATION_PROMPT.format(
            source_lang=source_lang_name, target_lang=target_lang_name
        )
        chunk_details: list[dict[str, Any]] = []

        try:
            if len(text) > 6000:
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=4000)
                translated_chunks: list[str] = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"РџРµСЂРµРІРѕРґРёРј С‡Р°СЃС‚СЊ {i+1}/{len(chunks)}")

                    chunk_prompt = (
                        f"Р§Р°СЃС‚СЊ {i + 1} РёР· {len(chunks)} РґРѕРєСѓРјРµРЅС‚Р° РґР»СЏ РїРµСЂРµРІРѕРґР°.\n"
                        "РЎРѕС…СЂР°РЅРё СЃС‚СЂСѓРєС‚СѓСЂСѓ, СЃРїРёСЃРєРё, С‚Р°Р±Р»РёС†С‹ Рё РЅСѓРјРµСЂР°С†РёСЋ.\n\n"
                        f"{chunk}"
                    )

                    result = await self.openai_service.ask_legal(
                        system_prompt=prompt, user_text=chunk_prompt
                    )

                    if result.get("ok"):
                        translated_part = result.get("text", "")
                        translated_chunks.append(translated_part)
                        chunk_details.append(
                            {
                                "chunk_number": i + 1,
                                "source_preview": chunk[:200],
                                "translated_preview": translated_part[:200],
                            }
                        )
                    else:
                        raise ProcessingError(
                            f"РћС€РёР±РєР° РїРµСЂРµРІРѕРґР° С‡Р°СЃС‚Рё {i + 1}", "TRANSLATION_ERROR"
                        )

                return "\n\n".join(translated_chunks), chunk_details

            result = await self.openai_service.ask_legal(system_prompt=prompt, user_text=text)

            if result.get("ok"):
                translated_text = result.get("text", "")
                chunk_details.append(
                    {
                        "chunk_number": 1,
                        "source_preview": text[:200],
                        "translated_preview": translated_text[:200],
                    }
                )
                return translated_text, chunk_details

            raise ProcessingError(
                "РќРµ СѓРґР°Р»РѕСЃСЊ РїРµСЂРµРІРµСЃС‚Рё РґРѕРєСѓРјРµРЅС‚", "TRANSLATION_ERROR"
            )

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° РїРµСЂРµРІРѕРґР°: {e}")
            raise ProcessingError(f"РћС€РёР±РєР° РїРµСЂРµРІРѕРґР°: {str(e)}", "TRANSLATION_ERROR")

    def get_supported_languages(self) -> dict[str, str]:
        """РџРѕР»СѓС‡РёС‚СЊ СЃРїРёСЃРѕРє РїРѕРґРґРµСЂР¶РёРІР°РµРјС‹С… СЏР·С‹РєРѕРІ"""
        return self.supported_languages.copy()

    def detect_language(self, text: str) -> str:
        """РџСЂРѕСЃС‚РѕРµ РѕРїСЂРµРґРµР»РµРЅРёРµ СЏР·С‹РєР° С‚РµРєСЃС‚Р°"""
        # РћС‡РµРЅСЊ РїСЂРѕСЃС‚Р°СЏ СЌРІСЂРёСЃС‚РёРєР° - РјРѕР¶РЅРѕ СѓР»СѓС‡С€РёС‚СЊ
        if re.search(r"[Р°-СЏС‘]", text.lower()):
            return "ru"
        elif re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        elif re.search(r"[Г¤Г¶ГјГџ]", text.lower()):
            return "de"
        else:
            return "en"  # РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ
