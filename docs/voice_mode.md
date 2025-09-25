# Voice Mode Setup

Voice interaction lets the bot accept Telegram voice messages, convert them to text, process the request, and answer back with both text and a synthesized male voice reply.

## Enablement Checklist

1. **Models and quota**
   - Speech-to-text model (default: gpt-4o-mini-transcribe).
   - Text-to-speech model (default: gpt-4o-mini-tts).
2. **Environment variables** (add to .env or deployment secrets):

   `env
   ENABLE_VOICE_MODE=1
   VOICE_STT_MODEL=gpt-4o-mini-transcribe
   VOICE_TTS_MODEL=gpt-4o-mini-tts
   VOICE_TTS_VOICE=alloy
   VOICE_TTS_FORMAT=ogg
   VOICE_MAX_DURATION_SECONDS=120
   `

   Adjust values if you prefer another OpenAI model or a different voice preset.

3. **Binary format**
   - VOICE_TTS_FORMAT=ogg keeps the reply compatible with Telegram voice messages.
   - Set VOICE_TTS_FORMAT=mp3 if you prefer classic audio files (the bot still uses send_voice).

4. **Operational notes**
   - Long recordings are rejected with a friendly warning (VOICE_MAX_DURATION_SECONDS).
   - Transcribed text is echoed back so the user can see what the bot understood before the legal answer appears.
   - If text-to-speech fails, the bot still delivers the text answer and logs the failure for troubleshooting.

## Usage Tips

- Users just send a voice message; no commands required.
- The bot streams the legal answer as usual, then drops a synthesized voice reply.
- Monitor OpenAI usage—voice features trigger additional requests per interaction.
