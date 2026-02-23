# 🎤 Building Real-Time Voice AI Agents

### بناء وكلاء صوت ذكية في الوقت الحقيقي

**EEExplore Extra Workshop** — University of Khartoum

Instructor: **Mohammed Elhaj Sami** | [LinkedIn](https://linkedin.com/in/mohammedelhaj) | mohamedelhaj2000@gmail.com

---

## 🇸🇩 نبذة عن الورشة

ورشة عملية من 3 جلسات مسجلة لبناء وكيل صوتي ذكي يسمع ويفكر ويتكلم في الوقت الحقيقي. من الأساسيات النظرية لحد بناء وكيل كامل بقدرات البحث في الإنترنت والوصول لقاعدة معرفة خاصة.

## 🇬🇧 About This Workshop

A 3-session hands-on workshop for building a real-time voice AI agent that can listen, think, and speak. From theory to a fully functional agent with web search, RAG, and feedback collection.

| Session | Title | Focus |
|---------|-------|-------|
| **1** | The New Age of AI | Generative AI concepts, voice pipeline, prompt engineering |
| **2** | From Text to Talk | Building a real-time voice agent (~20 lines of code) |
| **3** | Giving Your Agent Superpowers | Tool calling, RAG, web search, feedback |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/eeexplore-voice-ai-workshop.git
cd eeexplore-voice-ai-workshop
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run!

```bash
# Session 1: Start with the basics, then see the latency problem
python session1_basics/00_hello_gemini.py
python session1_basics/04_basic_pipeline.py

# Session 2: Talk to your real-time voice agent
python session2_realtime_agent/01_basic_agent.py dev

# Session 3: Build the RAG index, then run the full agent
python rag/build_index.py
python session3_superpowers/03_agent_full.py dev
```

---

## 🔑 API Keys You Need

| Service | What For | Get It Here | Cost |
|---------|----------|-------------|------|
| **LiveKit Cloud** | Real-time voice infrastructure | [cloud.livekit.io](https://cloud.livekit.io) | Free tier |
| **Google AI Studio** | Gemini LLM + RAG embeddings | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier |
| **ElevenLabs** | STT + TTS (ears + voice) | [elevenlabs.io](https://elevenlabs.io) | Free tier |

> **Note:** Google Gemini is used for both the agent's LLM brain AND for RAG embeddings (`text-embedding-004`). No OpenAI key needed!

---

## 📁 Repository Structure

```
eeexplore-voice-ai-workshop/
├── README.md                          # This file
├── requirements.txt                   # All dependencies
├── .env.example                       # API keys template
├── .gitignore
│
├── session1_basics/                   # Session 1: Concepts & Latency Demo
│   ├── README.md
│   ├── 00_hello_gemini.py            # First API call (simplest example)
│   ├── 01_chat_with_gemini.py        # Multi-turn conversations & memory
│   ├── 02_streaming_response.py      # Streaming tokens in real-time
│   ├── 03_prompt_engineering.py       # Interactive prompt comparison
│   └── 04_basic_pipeline.py           # Waterfall STT→LLM→TTS (shows latency)
│
├── session2_realtime_agent/           # Session 2: Real-Time Voice Agent
│   ├── README.md
│   ├── 01_basic_agent.py             # Minimal ~20 line agent (live coding)
│   └── 02_agent_with_personality.py  # Agent + Arabic personality
│
├── session3_superpowers/              # Session 3: Tools, RAG, Full Agent
│   ├── README.md
│   ├── 01_agent_with_search.py       # Agent + web search
│   ├── 02_agent_with_rag.py          # Agent + RAG knowledge base
│   ├── 03_agent_full.py             # Final agent: RAG + search + feedback
│   └── tools/
│       ├── __init__.py
│       ├── web_search.py             # DuckDuckGo search utility
│       └── feedback.py               # Feedback collection (JSON)
│
├── rag/                               # RAG System
│   ├── knowledge_base.md             # Arabic knowledge base document
│   ├── build_index.py                # Build Annoy vector index
│   ├── query_index.py                # Test RAG queries
│   └── index/                        # Generated index files (gitignored)
│
├── demos/                             # Bonus Demos
│   ├── gemini_native_demo.py         # Gemini speech-to-speech
│   └── elevenlabs_voice_clone.py     # Voice cloning example
│
└── feedback_logs/                     # Feedback data (gitignored)
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Framework** | LiveKit Agents SDK | Real-time voice infrastructure |
| **VAD** | Silero | Voice Activity Detection |
| **STT** | ElevenLabs | Speech-to-Text (ears) |
| **LLM** | Google Gemini | Language Model (brain) |
| **TTS** | ElevenLabs | Text-to-Speech (voice) |
| **Vector DB** | Annoy | RAG similarity search |
| **Embeddings** | Gemini `text-embedding-004` | RAG document embedding |
| **Web Search** | DuckDuckGo | Live internet search |

---

## 🧪 Testing Your Agent

1. Run any agent with the `dev` flag:
   ```bash
   python session2_realtime_agent/01_basic_agent.py dev
   ```

2. Open [LiveKit Playground](https://agents-playground.livekit.io/) in your browser

3. Connect to your LiveKit project

4. Start talking! The agent appears as a participant in the room

---

## 📚 Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Course on DeepLearning.AI](https://www.deeplearning.ai/short-courses/voice-agents-livekit/)
- [Andrew Ng — Agentic AI Workflows](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)
- [ElevenLabs Documentation](https://docs.elevenlabs.io/)
- [Google Gemini API](https://ai.google.dev/)

---

## 👤 Credits

- **Instructor:** Mohammed Elhaj Sami — AI System Architect & Voice AI Specialist
- **Program:** EEExplore Extra — University of Khartoum

---

*Built with ❤️ for the EEExplore community*
