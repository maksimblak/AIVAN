from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
from contextlib import suppress

import httpx
from pathlib import Path
from typing import Any, Optional

from openai import APIStatusError, AsyncOpenAI

try:
    from src.bot.openai_gateway import _make_async_client as _gateway_make_async_client  # type: ignore
except Exception:  # noqa: BLE001
    _gateway_make_async_client = None

logger = logging.getLogger(__name__)

_SUPPORTED_TTS_FORMATS = {"mp3", "aac", "opus", "flac", "pcm", "wav"}
_TTS_FORMAT_ALIASES = {
    "ogg": ("opus", "ogg"),
    "oga": ("opus", "ogg"),
}
_DEFAULT_TTS_FORMAT = ("mp3", "mp3")
_DEFAULT_TTS_CHUNK_CHAR_LIMIT = 6000


_KWARG_ERROR_RE = re.compile(r"unexpected keyword argument '([^']+)'")


async def _acquire_client() -> AsyncOpenAI:
    """Reuse gateway client factory when available to keep proxy/timeouts consistent."""
    if _gateway_make_async_client is not None:
        return await _gateway_make_async_client()
    return AsyncOpenAI()


class AudioService:
    """Handles speech-to-text and text-to-speech interactions."""

    def __init__(
        self,
        stt_model: str,
        tts_model: str,
        tts_voice: str,
        tts_format: str = "ogg",
        *,
        max_duration_seconds: int = 120,
        tts_voice_male: Optional[str] = None,
        tts_chunk_char_limit: int = _DEFAULT_TTS_CHUNK_CHAR_LIMIT,
        tts_speed: Optional[float] = None,
        tts_style: Optional[str] = None,
        tts_sample_rate: Optional[int] = None,
        tts_backend: str = "auto",
    ) -> None:
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_voice_male = tts_voice_male or tts_voice
        self.tts_format = tts_format
        self.max_duration_seconds = max_duration_seconds
        self.tts_chunk_char_limit = max(int(tts_chunk_char_limit), 0)
        self.tts_speed = tts_speed
        self.tts_style = tts_style
        self.tts_sample_rate = tts_sample_rate
        self._backend_config = (tts_backend or "auto").strip().lower() or "auto"
        if self._backend_config not in {"auto", "speech", "responses"}:
            logger.warning("Unsupported TTS backend '%s'; falling back to 'auto'", tts_backend)
            self._backend_config = "auto"

    def _decide_backend(self) -> str:
        if self._backend_config == "auto":
            model_name = (self.tts_model or "").lower()
            if "audio-preview" in model_name or "realtime" in model_name:
                return "responses"
            return "speech"
        return self._backend_config

    def _resolve_tts_format(self) -> tuple[str, str]:
        """Normalize configured TTS format for the OpenAI API and local files."""
        requested = (self.tts_format or "").strip().lower()
        if requested in _SUPPORTED_TTS_FORMATS:
            return requested, requested
        alias = _TTS_FORMAT_ALIASES.get(requested)
        if alias:
            return alias
        if requested:
            logger.warning(
                "Unsupported TTS format '%s'; falling back to '%s'",
                self.tts_format,
                _DEFAULT_TTS_FORMAT[0],
            )
        return _DEFAULT_TTS_FORMAT

    async def transcribe(self, file_path: Path, *, language: Optional[str] = None) -> str:
        """Convert voice message to text using configured STT model."""
        client = await _acquire_client()
        async with client as oai:
            with file_path.open("rb") as audio_file:
                response = await oai.audio.transcriptions.create(
                    model=self.stt_model,
                    file=audio_file,
                    response_format="text",
                    language=language,
                )
        if isinstance(response, str):
            text = response.strip()
        else:
            text = getattr(response, "text", "") or ""
            text = text.strip()
        if not text:
            raise RuntimeError("Speech recognition returned empty text")
        return text

    async def synthesize(
        self,
        text: str,
        *,
        prefer_male: bool = False,
        voice_override: Optional[str] = None,
    ) -> list[Path]:
        """Generate speech files from text and return list of audio file paths."""
        slices = self._split_text_into_chunks(text)
        if not slices:
            raise ValueError("Text-to-speech received empty input")

        client = await _acquire_client()
        api_format, file_extension = self._resolve_tts_format()
        voice_to_use = voice_override or (
            self.tts_voice_male if prefer_male and self.tts_voice_male else self.tts_voice
        )

        paths: list[Path] = []
        async with client as oai:
            try:
                for chunk in slices:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f".{file_extension}", delete=False
                    )
                    tmp_path = Path(tmp.name)
                    tmp.close()
                    speech = await self._create_speech_payload(
                        oai=oai,
                        text=chunk,
                        voice=voice_to_use,
                        api_format=api_format,
                    )
                    await self._store_speech_response(speech, tmp_path)
                    paths.append(tmp_path)
            except Exception:
                for path in paths:
                    with suppress(Exception):
                        path.unlink()
                raise

        if len(paths) > 1:
            logger.info(
                "TTS input split into %s chunks (limit=%s chars)",
                len(paths),
                self.tts_chunk_char_limit,
            )

        return paths

    async def _create_speech_payload(
        self,
        oai: AsyncOpenAI,
        text: str,
        voice: str,
        api_format: str,
    ) -> Any:
        base_kwargs: dict[str, Any] = {
            "model": self.tts_model,
            "voice": voice,
            "input": text,
        }
        optional_kwargs: dict[str, Any] = {}
        if self.tts_speed is not None:
            optional_kwargs["speed"] = self.tts_speed
        if self.tts_style:
            optional_kwargs["style"] = self.tts_style
        if self.tts_sample_rate is not None:
            optional_kwargs["sample_rate"] = self.tts_sample_rate

        backend = self._decide_backend()
        allow_fallback = self._backend_config == "auto"

        if backend == "responses":
            return await self._create_speech_via_responses(
                oai=oai,
                text=text,
                voice=voice,
                api_format=api_format,
                optional_kwargs=optional_kwargs,
            )

        last_error: Exception | None = None
        for format_key in ("format", "response_format"):
            kwargs = {**base_kwargs, format_key: api_format, **optional_kwargs}
            try:
                return await self._invoke_tts(
                    oai=oai, kwargs=kwargs, optional_keys=optional_kwargs.keys()
                )
            except TypeError as error:
                last_error = error
            except APIStatusError as api_error:
                if allow_fallback and self._should_try_responses_api(api_error):
                    return await self._create_speech_via_responses(
                        oai=oai,
                        text=text,
                        voice=voice,
                        api_format=api_format,
                        optional_kwargs=optional_kwargs,
                    )
                last_error = api_error
            except httpx.HTTPStatusError as http_error:
                if allow_fallback and self._should_try_responses_api(http_error):
                    return await self._create_speech_via_responses(
                        oai=oai,
                        text=text,
                        voice=voice,
                        api_format=api_format,
                        optional_kwargs=optional_kwargs,
                    )
                last_error = http_error
        if allow_fallback and self._should_try_responses_api(last_error):
            return await self._create_speech_via_responses(
                oai=oai,
                text=text,
                voice=voice,
                api_format=api_format,
                optional_kwargs=optional_kwargs,
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected failure in text-to-speech payload creation")


    async def _invoke_tts(
        self,
        *,
        oai: AsyncOpenAI,
        kwargs: dict[str, Any],
        optional_keys: Any,
    ) -> Any:
        mutable_kwargs = dict(kwargs)
        optional = {key for key in optional_keys}
        while True:
            try:
                return await oai.audio.speech.create(**mutable_kwargs)
            except TypeError as err:
                match = _KWARG_ERROR_RE.search(str(err))
                if not match:
                    raise
                bad_key = match.group(1)
                if bad_key not in mutable_kwargs or bad_key not in optional:
                    raise
                mutable_kwargs.pop(bad_key, None)
                optional.discard(bad_key)

    def _should_try_responses_api(self, error: Exception | None = None) -> bool:
        if self._backend_config != "auto":
            return False
        model_name = (self.tts_model or "").lower()
        if "audio-preview" in model_name or "realtime" in model_name:
            return True
        if isinstance(error, APIStatusError) and error.status_code in {404, 405}:
            return True
        if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {404, 405}:
            return True
        return False

    async def _create_speech_via_responses(
        self,
        *,
        oai: AsyncOpenAI,
        text: str,
        voice: str,
        api_format: str,
        optional_kwargs: dict[str, Any],
    ) -> bytes:
        audio_payload: dict[str, Any] = {
            "voice": voice,
            "format": api_format,
        }
        for key in ("speed", "style", "sample_rate"):
            if key in optional_kwargs and optional_kwargs[key] is not None:
                audio_payload[key] = optional_kwargs[key]

        optional_keys = [key for key in audio_payload if key not in {"voice", "format"}]
        while True:
            try:
                response = await oai.responses.create(
                    model=self.tts_model,
                    input=text,
                    extra_body={
                        "modalities": ["text", "audio"],
                        "audio": audio_payload,
                    },
                )
                break
            except APIStatusError:
                if not optional_keys:
                    raise
                removed = optional_keys.pop()
                audio_payload.pop(removed, None)
                continue
            except httpx.HTTPStatusError as http_error:
                if not optional_keys or http_error.response.status_code not in {400, 422}:
                    raise
                removed = optional_keys.pop()
                audio_payload.pop(removed, None)
                continue

        payload = self._to_plain_data(response)
        audio_b64 = self._extract_audio_base64(payload)
        if not audio_b64:
            raise RuntimeError("Responses API did not provide audio content")
        try:
            return base64.b64decode(audio_b64)
        except Exception as decode_error:  # noqa: BLE001
            raise RuntimeError("Failed to decode audio payload from responses API") from decode_error

    @staticmethod
    def _to_plain_data(obj: Any) -> Any:
        for attr in ("model_dump", "dict"):
            candidate = getattr(obj, attr, None)
            if callable(candidate):
                try:
                    return candidate()
                except Exception:  # noqa: BLE001
                    continue
        return obj

    def _extract_audio_base64(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            audio_section = payload.get("audio")
            if isinstance(audio_section, dict):
                for key in ("data", "b64", "base64"):
                    value = audio_section.get(key)
                    if isinstance(value, str):
                        return value
            for value in payload.values():
                found = self._extract_audio_base64(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._extract_audio_base64(item)
                if found:
                    return found
        return None

    async def _store_speech_response(self, speech: Any, target_path: Path) -> None:
        if hasattr(speech, "stream_to_file"):
            result = speech.stream_to_file(target_path)
            if asyncio.iscoroutine(result):
                await result
            return
        data = None
        if hasattr(speech, "read"):
            maybe = speech.read()  # type: ignore[attr-defined]
            if asyncio.iscoroutine(maybe):
                data = await maybe
            else:
                data = maybe  # type: ignore[assignment]
        elif isinstance(speech, (bytes, bytearray)):
            data = bytes(speech)
        if data is None:
            raise RuntimeError("Unsupported speech response payload")
        target_path.write_bytes(data)

    def _split_text_into_chunks(self, text: str) -> list[str]:
        normalized = (text or "").strip()
        if not normalized:
            return []
        limit = max(self.tts_chunk_char_limit, 0)
        if limit <= 0 or len(normalized) <= limit:
            return [normalized]

        chunks: list[str] = []
        remaining = normalized
        while remaining:
            if len(remaining) <= limit:
                chunk = remaining.rstrip()
                if chunk:
                    chunks.append(chunk)
                break
            cut = self._find_split_index(remaining, limit)
            if cut <= 0:
                cut = limit
            chunk = remaining[:cut].rstrip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].lstrip()
            if not remaining:
                break
        return chunks

    @staticmethod
    def _find_split_index(text: str, limit: int) -> int:
        for delimiter in ("\n\n", "\n", ". ", "! ", "? ", "; "):
            idx = text.rfind(delimiter, 0, limit)
            if idx > 0:
                return idx + len(delimiter)
        idx = text.rfind(" ", 0, limit)
        if idx > 0:
            return idx + 1
        return limit

    async def ensure_short_enough(self, duration_seconds: Optional[int]) -> None:
        """Raise if voice message is too long."""
        if duration_seconds is None:
            return
        if duration_seconds > self.max_duration_seconds:
            raise ValueError(
                f"Voice message longer than allowed {self.max_duration_seconds} seconds"
            )
