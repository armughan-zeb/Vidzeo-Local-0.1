# =============================================================================
# 🚀 KAGGLE INSTALLATION - WITH AUTO-CAPTIONS + FONTS
# =============================================================================
# Based on: https://github.com/nikhil-reddy05/auto-captions
# =============================================================================

import subprocess
import sys
import os
import urllib.request

print("="*60)
print("📦 INSTALLING (With Auto-Captions + Custom Fonts)")
print("="*60)

BASE = '/kaggle/working' if os.path.exists('/kaggle') else '/content'
FONTS_DIR = f"{BASE}/fonts"
os.makedirs(FONTS_DIR, exist_ok=True)

# =============================================================================
# FONT INSTALLATION - Verified Working URLs from google/fonts GitHub
# =============================================================================

print("\n🔤 Installing Custom Fonts...")

# All fonts from google/fonts GitHub repo (verified working)
FONTS = {
    # BOLD - Impact fonts
    "BebasNeue-Regular": "https://raw.githubusercontent.com/dharmatype/Bebas-Neue/master/fonts/BebasNeue(2018)ByDhamraType/ttf/BebasNeue-Regular.ttf",
    "Anton-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    
    # MODERN
    "Montserrat-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Montserrat-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Poppins-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Bold.ttf",
    "Poppins-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
    "Roboto-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "Roboto-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    
    # BUBBLY
    "LilitaOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/lilitaone/LilitaOne-Regular.ttf",
    "FredokaOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/fredokaone/FredokaOne-Regular.ttf",
    
    # HORROR
    "Creepster-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/creepster/Creepster-Regular.ttf",
    "Nosifer-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/nosifer/Nosifer-Regular.ttf",
    
    # HANDWRITING
    "Caveat-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "Caveat-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "Pacifico-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "DancingScript-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "PatrickHand-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    
    # FORMAL
    "PlayfairDisplay-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Cinzel-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",
    
    # FAT/BOLD
    "TitanOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/titanone/TitanOne-Regular.ttf",
    "Bungee-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/bungee/Bungee-Regular.ttf",
    
    # CREATIVE
    "Righteous-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/righteous/Righteous-Regular.ttf",
    "Audiowide-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/audiowide/Audiowide-Regular.ttf",
    "Orbitron-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf",
    
    # INFORMAL
    "ComicNeue-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
    "OpenSans-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "OpenSans-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
}

def download_font(name, url, dest_dir):
    """Download a single font file"""
    try:
        out_path = os.path.join(dest_dir, f"{name}.ttf")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        font_data = response.read()
        
        with open(out_path, 'wb') as f:
            f.write(font_data)
        
        return True
    except Exception as e:
        return False

# Download all fonts
print(f"   📁 Font directory: {FONTS_DIR}")
downloaded = 0
failed = []

for name, url in FONTS.items():
    display_name = name.replace("-", " ").replace("_", " ")
    
    if download_font(name, url, FONTS_DIR):
        downloaded += 1
        print(f"   ✅ {display_name}")
    else:
        failed.append(display_name)
        print(f"   ⚠️ {display_name} - failed")

# Install fonts system-wide (Linux)
print("\n   Installing fonts to system...")
subprocess.run(['cp', '-r', f"{FONTS_DIR}/.", '/usr/share/fonts/truetype/'], capture_output=True)
subprocess.run(['fc-cache', '-f', '-v'], capture_output=True)

print(f"\n   ✅ {downloaded}/{len(FONTS)} fonts installed")
if failed:
    print(f"   ⚠️ Failed: {', '.join(failed[:5])}")

# =============================================================================
# CORE PACKAGES
# =============================================================================

# Core packages (SAME ORDER AS WORKING VERSION)
print("\n1️⃣ Core packages...")
core = ['gradio>=5.0.0', 'soundfile', 'pillow', 'pydub', 'moviepy']
for pkg in core:
    print(f"   {pkg}...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], capture_output=True, timeout=120)

# OpenAI Whisper for auto-captions (BEFORE other packages to set numpy version)
print("\n2️⃣ OpenAI Whisper (for captions)...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openai-whisper'], capture_output=True, timeout=300)

# Kokoro TTS (BEFORE AI packages)
print("\n3️⃣ Kokoro TTS...")
KOKORO_DIR = f"{BASE}/Kokoro-TTS-Subtitle"
if not os.path.exists(KOKORO_DIR):
    print("   Cloning Kokoro...")
    subprocess.run(['git', 'clone', '-q', 'https://github.com/NeuralFalconYT/Kokoro-TTS-Subtitle.git', KOKORO_DIR], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kokoro>=0.9.4'], capture_output=True, timeout=180)

# espeak-ng
subprocess.run(['apt-get', '-qq', '-y', 'install', 'espeak-ng'], capture_output=True)

# AI Image Generation APIs (AFTER Kokoro to avoid conflicts)
print("\n4️⃣ AI Image Generation APIs...")
ai_apis = ['groq', 'openai', 'requests']
for pkg in ai_apis:
    print(f"   {pkg}...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', pkg], capture_output=True, timeout=120)

# OpenNSFW2 - Install with --no-deps to avoid numpy conflicts
print("\n5️⃣ OpenNSFW2 (NSFW Detection)...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'opennsfw2'], capture_output=True, timeout=180)
print("   ✅ NSFW detection ready")

# DuckDuckGo Search for AI Script Research
print("\n6️⃣ DuckDuckGo Search (for script research)...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'duckduckgo-search'], capture_output=True, timeout=120)
print("   ✅ Web search enabled")

print("\n" + "="*60)
print("✅ INSTALLATION COMPLETE")
print("="*60)
print(f"\n🔤 {downloaded} fonts installed")
print("⚠️ NOW RUN: !python KAGGLE_APP.py")
