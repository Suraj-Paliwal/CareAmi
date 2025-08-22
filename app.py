import os, io, tempfile, time, uuid, base64, re
import streamlit as st
import numpy as np
from dotenv import load_dotenv

# Avoid OpenMP duplicate runtime crash on Windows (NumPy/PyTorch, etc.)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load env
load_dotenv(override=True)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://172.21.80.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TTS_BACKEND   = os.getenv("TTS_BACKEND", "elevenlabs").lower()

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "sk_518ebcb5fcd711685d76f22680a3a2cd4bf4a19d9940095c")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVEN_OUTPUT_FORMAT = os.getenv("ELEVEN_OUTPUT_FORMAT", "mp3_44100_128")

st.set_page_config(page_title="Local Voice Chatbot (LM Studio)", page_icon="🎙️", layout="centered")
st.title("🎙️ Local Voice Chatbot — LM Studio + Whisper + ElevenLabs")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.write("**LM Studio** server must be running (Developer → Start Server).")
    st.text_input("LLM Base URL", LLM_BASE_URL, key="cfg_base_url")
    st.text_input("Model ID", LLM_MODEL, key="cfg_model_id")
    st.text_input("API Key (placeholder; LM Studio ignores)", OPENAI_API_KEY, key="cfg_api_key", type="password")
    st.selectbox("Whisper model", ["tiny","base","small","medium","large-v3"],
                 index=["tiny","base","small","medium","large-v3"].index(WHISPER_MODEL), key="cfg_whisper")
    st.selectbox("TTS backend", ["elevenlabs","none"],
                 index=["elevenlabs","none"].index(TTS_BACKEND), key="cfg_tts")
    if st.session_state.cfg_tts == "elevenlabs":
        st.text_input("ElevenLabs API Key", ELEVEN_API_KEY, key="cfg_eleven_key", type="password")
        st.text_input("ElevenLabs Voice ID", ELEVEN_VOICE_ID, key="cfg_eleven_voice")
        st.text_input("ElevenLabs Output Format", ELEVEN_OUTPUT_FORMAT, key="cfg_eleven_fmt")
    st.markdown("---")
    st.caption("Tip: If the mic widget isn't available, upload a WAV file instead.")

# Lazy imports
from openai import OpenAI
from faster_whisper import WhisperModel
import soundfile as sf
import requests

# Cache heavy models
@st.cache_resource(show_spinner=True)
def load_whisper_model(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8")

def transcribe_bytes(wav_bytes: bytes, model_size: str) -> str:
    # Save to temp wav and run faster-whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav_bytes)
        tmp = f.name
    model = load_whisper_model(model_size)
    segments, info = model.transcribe(tmp, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text

# ---------- Final-only extraction helpers ----------
FINAL_TAG_OPEN  = "<final>"
FINAL_TAG_CLOSE = "</final>"

def _extract_final(text: str) -> str:
    # 1) Preferred: <final>...</final>
    m = re.search(rf"{re.escape(FINAL_TAG_OPEN)}(.*?){re.escape(FINAL_TAG_CLOSE)}", text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    # 2) Fallback: take last 'Assistant:' block
    parts = re.split(r"(?i)\bAssistant\s*:\s*", text)
    if len(parts) > 1:
        return parts[-1].strip()
    # 3) Cleanup tokens/labels
    text = re.sub(r"<\|.*?\|>", "", text)          # remove <|channel|> etc.
    text = re.sub(r"(?i)\b(System|User)\s*:\s*", "", text)
    return text.strip()

# ---------- LLM call (completions endpoint) ----------
def chat_llm(messages: list[dict]) -> str:
    """
    - Appends a system guard to force the model to wrap its final reply in <final>...</final>.
    - Uses stop sequences to keep the model from emitting new role labels.
    - Post-processes to extract only the final content.
    """
    client = OpenAI(base_url=st.session_state.cfg_base_url, api_key=st.session_state.cfg_api_key)

    history = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    guard = (
        "\n\nSystem: Reply with ONLY the final message for the user, "
        f"wrapped in {FINAL_TAG_OPEN}{FINAL_TAG_CLOSE}. "
        "Do not include roles, labels, analysis, or extra tokens."
    )
    prompt = history + guard + "\nAssistant: "

    try:
        resp = client.completions.create(
            model=st.session_state.cfg_model_id,
            prompt=prompt,
            temperature=0.7,
            max_tokens=512,
            stop=["\nUser:", "\nSystem:", "User:", "System:"],  # stop before looping back
        )
        raw = (resp.choices[0].text or "").strip()
        return _extract_final(raw)
    except Exception as e:
        return f"(LLM error: {e})"

def elevenlabs_tts(text: str) -> bytes|None:
    key = st.session_state.get("cfg_eleven_key") or ""
    if not key:
        st.warning("Set ELEVEN_API_KEY in the sidebar for TTS.")
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{st.session_state.cfg_eleven_voice}"
    headers = {
        "xi-api-key": key,
        "Accept": "audio/mpeg"  # return MP3 bytes
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "output_format": st.session_state.cfg_eleven_fmt
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        st.error(f"ElevenLabs error {r.status_code}: {r.text[:200]}")
        return None
    return r.content

# Session state: chat history (we keep it for context, but won't render it)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful, concise voice assistant."}]

# UI blocks
st.subheader("🎤 Record your question")
audio_bytes = None

# Try mic component first
try:
    from streamlit_mic_recorder import mic_recorder
    val = mic_recorder(start_prompt="Start recording", stop_prompt="Stop", just_once=False, key="rec")
    if isinstance(val, dict) and "bytes" in val:
        audio_bytes = val["bytes"]
    elif isinstance(val, (bytes, bytearray)):
        audio_bytes = bytes(val)
except Exception:
    st.info("Mic widget unavailable. Use file uploader below.")

# Fallback: file uploader (WAV)
uploaded = st.file_uploader("...or upload a WAV file", type=["wav"])
if uploaded is not None:
    audio_bytes = uploaded.read()

col1, col2 = st.columns(2)
with col1:
    user_text = st.text_area("Or type your question", height=100, key="typed_text", placeholder="Ask anything...")
with col2:
    auto_tts = st.checkbox("🔊 Auto-speak replies", value=True)

go = st.button("Send", type="primary")

# If we have audio, preview it
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

# Process input
if go:
    # Derive user prompt: prefer typed text; else transcribe audio
    prompt = (user_text or "").strip()
    if not prompt and audio_bytes:
        with st.spinner("Transcribing..."):
            prompt = transcribe_bytes(audio_bytes, st.session_state.cfg_whisper)
    if not prompt:
        st.warning("Please type or record a question first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        reply = chat_llm(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # TTS
    audio_reply = None
    if auto_tts and st.session_state.cfg_tts == "elevenlabs" and reply and not reply.startswith("(LLM error"):
        with st.spinner("Generating speech..."):
            audio_reply = elevenlabs_tts(reply)

    # ---- Show ONLY the assistant's final message ----
    with st.container(border=True):
        st.markdown("**Assistant**")
        st.write(reply)
        if audio_reply:
            st.audio(audio_reply, format="audio/mp3")

st.markdown("---")

# (Optional) Reset button — we don't render history to the user
colr1, colr2 = st.columns(2)
with colr1:
    if st.button("Reset conversation"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful, concise voice assistant."}]
        st.experimental_rerun()
with colr2:
    st.caption("Use Chrome on localhost for easiest mic access. Some browsers require HTTPS for mic permissions.")
