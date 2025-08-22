

# 🏥 NurseCare Robot

An **interactive assistive robot** that uses **HEBI actuators**, **MediaPipe pose detection**, and **speech/sound feedback** to recognize human gestures and respond with **lifelike robot actions**.

---

## ✨ Features

* 👋 **Wave detection** → Robot waves back using actuator
* 🙇 **Bow detection** → Robot performs a smooth bow
* 👏 **Clap detection** → Detects hand clap, plays sound + small motion
* 🪑 **Sit detection** → Detects seated posture
* 🚨 **Fall detection** → Triggers alarm + speech warning
* 🔊 **Speech feedback** (`pyttsx3`)
* 🎵 **Sound effects** (`playsound`)
* 🌀 **Idle sway** for natural presence
* 📷 OpenCV HUD with gesture labels

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe hebi-py pyttsx3 playsound==1.2.2
```

---

## 🚀 Usage

1. Connect your HEBI actuators.
2. Edit `nurse_robot.py` with your HEBI family/module names:

   ```python
   family_name = "Enggar1"
   module_names = ["Rear_right", "Rear_left"]
   ```
3. Run:

   ```bash
   python nurse_robot.py
   ```

---

## 📷 System Flow

```
Camera ──▶ MediaPipe Pose ──▶ Gesture Recognition
                │
        ┌───────┴────────┐
        ▼                ▼
   Speech + Sound   HEBI Actuator Motion
```

---

## 📜 License

MIT License

---

---

# 🎙️ Voice Chatbot (LM Studio + Whisper + ElevenLabs)

A **Streamlit web app** for voice-enabled AI chat, combining:

* **LM Studio** for local LLM responses
* **Whisper (faster-whisper)** for speech-to-text
* **ElevenLabs** for natural text-to-speech

---

## ✨ Features

* 🎤 Record voice or upload `.wav` files
* 🧠 Query local LLM (LM Studio server)
* 🗣️ Transcribe audio with Whisper
* 🔊 Generate spoken replies with ElevenLabs TTS
* ⚙️ Customizable via sidebar (model, voice, output format)

---

## ⚙️ Installation

```bash
pip install streamlit numpy python-dotenv openai faster-whisper soundfile requests streamlit-mic-recorder
```

---

## 🔑 Environment Setup

Create a `.env` file:

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=openai/gpt-oss-20b
OPENAI_API_KEY=lm-studio
WHISPER_MODEL=small
TTS_BACKEND=elevenlabs
ELEVEN_API_KEY=your-elevenlabs-key
ELEVEN_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
ELEVEN_OUTPUT_FORMAT=mp3_44100_128
```

---

## 🚀 Usage

1. Start LM Studio (Developer → Start Server).
2. Run:

   ```bash
   streamlit run app.py
   ```
3. Interact:

   * Record your question 🎤
   * Upload audio 🎵
   * Or type a message ⌨️
   * Hear spoken reply 🔊

---

## 📷 Workflow

```
Mic / Audio ──▶ Whisper STT ──▶ LM Studio LLM ──▶ ElevenLabs TTS ──▶ Spoken Reply
```

---

## 📜 License

MIT License

---

👉 This way you can place:

* `nurse-robot/README.md`
* `voice-chatbot/README.md`


