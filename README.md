# 🎬 Vidzeo Local - AI Video Generator

Create professional AI-generated videos with custom scripts, AI images, and natural voiceovers - all running locally on your computer!

![Version](https://img.shields.io/badge/version-0.1-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- 📝 **AI Script Generation** - Generate engaging scripts from titles using Groq AI
- 🎨 **AI Image Generation** - Create stunning visuals with Pollinations AI (FREE) or Together AI
- 🎙️ **Natural TTS** - High-quality voiceover with Kokoro TTS (multiple voices)
- 📱 **Auto Captions** - Synchronized word-by-word captions with Whisper
- 🔧 **Full Customization** - Fonts, colors, animations, effects
- 📦 **Bulk Generation** - Generate multiple videos from CSV

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- FFmpeg

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/vidzeo-local.git
cd vidzeo-local

# Run the launcher (Windows)
start.bat
```

Or manually:
```bash
# Create virtual environment
python -m venv venv310
source venv310/Scripts/activate  # Windows
# source venv310/bin/activate    # Linux/Mac

# Install dependencies
pip install -r server/requirements.txt
pip install git+https://github.com/hexgrad/kokoro.git
pip install openai-whisper

# Start the server
cd server
python app.py
```

Then open http://localhost:5000 in your browser.

## 🔑 API Keys Required

Set these in the **Settings** tab:

| Service | Purpose | Get Key |
|---------|---------|---------|
| Groq | Script generation & scene extraction | [console.groq.com](https://console.groq.com) |
| Pollinations AI | FREE image generation | [enter.pollinations.ai](https://enter.pollinations.ai) (optional) |
| Together AI | Alternative image generation | [together.ai](https://api.together.xyz) |

## 📁 Project Structure

```
vidzeo-local/
├── server/
│   ├── app.py              # Main Flask server
│   ├── config.py           # Configuration
│   └── services/           # TTS, transcription, video, image
├── public/
│   ├── index.html          # Frontend UI
│   ├── app.js              # Frontend logic
│   └── style.css           # Styling
├── fonts/                  # Caption fonts
├── output/                 # Generated videos
└── start.bat               # Windows launcher
```

## 🎥 Usage

1. **Enter a title** or write your own script
2. **Choose image source**: Custom upload or AI-generated
3. **Select voice** and caption style
4. **Click Generate** and wait for your video!

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: Vanilla JS, CSS
- **TTS**: Kokoro TTS
- **STT**: OpenAI Whisper
- **Video**: FFmpeg
- **AI**: Groq (Llama), Pollinations AI, Together AI

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

Made with ❤️ by Vidzeo Team
