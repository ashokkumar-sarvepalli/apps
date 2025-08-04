# 🎬 Shorts Video Creator (1080x1920)

Generate vertical YouTube Shorts-style videos from images and videos with optional GPS-based captions and background music — all from your browser!

Powered by [Gradio](https://www.gradio.app/), `ffmpeg`, and Python 🐍

---

## 🚀 Features

- 📸 Upload images (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`) or videos (`.mp4`, `.mov`)
- 📦 Or upload a `.zip` file of your media
- 🧠 Auto-captions based on photo GPS data
- 🎵 Optional background music (`.mp3`)
- ⚙️ Adjustable frame rate (FPS)
- 📤 100% browser-based — no desktop app required!

---

## 📦 Installation (Local)

```bash
git clone https://github.com/ashokkumar-sarvepalli/apps.git
cd apps/shorts_video_creator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python shorts_generate.py
