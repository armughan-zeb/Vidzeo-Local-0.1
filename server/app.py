# =============================================================================
# VIDZEO LOCAL - Flask API Server (Full Version)
# =============================================================================

# Suppress HuggingFace connection warnings (use cached models)
import os
os.environ['HF_HUB_OFFLINE'] = '1'  # Don't try to connect to HuggingFace
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Offline mode for transformers
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'  # Disable telemetry
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Disable progress bars

import sys
import json
import time
import hashlib
import threading
import webbrowser
import urllib.request
from pathlib import Path

# Add server directory to path
SERVER_DIR = Path(__file__).parent
BASE_DIR = SERVER_DIR.parent
sys.path.insert(0, str(SERVER_DIR))

# Create directories
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
FONTS_DIR = BASE_DIR / "fonts"
PUBLIC_DIR = BASE_DIR / "public"

for d in [OUTPUT_DIR, TEMP_DIR, FONTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Import Flask
try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
except ImportError as e:
    print(f"ERROR: Flask not installed. Run: pip install flask flask-cors")
    input("Press Enter to exit...")
    sys.exit(1)

# =============================================================================
# LOAD CONFIG
# =============================================================================

try:
    from config import (
        VOICES, FONTS_FLAT, FONT_CATEGORIES, FONT_URLS,
        IMAGE_STYLE_NAMES, IMAGE_MODELS, IMAGE_RESOLUTIONS, IMAGE_STYLES, STYLE_HEADER,
        SCENE_COUNT_MODES, SCRIPT_DURATIONS, EFFECTS, ANIMATIONS
    )
    print("✅ Config loaded")
except Exception as e:
    print(f"⚠️ Using default config: {e}")
    VOICES = {'Adam (Male US)': 'am_adam', 'Sky (Female US)': 'af_sky'}
    FONTS_FLAT = ['Arial']
    FONT_CATEGORIES = {"Default": ["Arial"]}
    FONT_URLS = {}
    IMAGE_STYLE_NAMES = ["No Style"]
    IMAGE_MODELS = {"Flux Schnell (Free)": "flux"}
    IMAGE_RESOLUTIONS = {"512x768": {"width": 768, "height": 512}}
    IMAGE_STYLES = {"No Style": ""}
    STYLE_HEADER = ""
    SCENE_COUNT_MODES = ["30 images"]
    SCRIPT_DURATIONS = ["2 minutes"]
    EFFECTS = ["Zoom Combo", "None"]
    ANIMATIONS = {"Pop": {}, "None": {}}

# =============================================================================
# APP SETUP
# =============================================================================

app = Flask(__name__, static_folder=str(PUBLIC_DIR))
CORS(app)

# Global state
_tts_ready = False
_whisper_ready = False
_ffmpeg_ready = False
_tts_pipeline = None
_whisper_model = None

# Bulk queue
bulk_queue = []
bulk_results = []


def init_tts():
    """Initialize Kokoro TTS using standard kokoro package"""
    global _tts_ready, _tts_pipeline
    print("🎙️ Loading TTS...", flush=True)
    
    try:
        print("   Step 1: Importing kokoro...", flush=True)
        from kokoro import KPipeline
        print("   Step 2: KPipeline imported, creating instance...", flush=True)
        
        # Initialize with American English
        _tts_pipeline = KPipeline(lang_code='a')
        print("   Step 3: KPipeline created successfully", flush=True)
        
        _tts_ready = True
        print("✅ Kokoro TTS ready", flush=True)
    except ImportError as ie:
        print(f"⚠️ TTS import error: {ie}", flush=True)
        _tts_ready = False
    except Exception as e:
        import traceback
        print("⚠️ TTS initialization error:", flush=True)
        traceback.print_exc()
        print(f"   Error: {e}", flush=True)
        _tts_ready = False


def init_whisper():
    """Initialize Whisper"""
    global _whisper_ready, _whisper_model
    try:
        import whisper
        _whisper_model = whisper.load_model("base")
        _whisper_ready = True
        print("✅ Whisper ready")
    except Exception as e:
        print(f"⚠️ Whisper not available: {e}")
        _whisper_ready = False


def check_ffmpeg():
    """Check FFmpeg"""
    global _ffmpeg_ready
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        _ffmpeg_ready = result.returncode == 0
        print(f"{'✅' if _ffmpeg_ready else '❌'} FFmpeg: {'found' if _ffmpeg_ready else 'not found'}")
    except:
        _ffmpeg_ready = False
        print("❌ FFmpeg: not found")


def download_fonts():
    """Download fonts if missing"""
    count = 0
    for name, url in FONT_URLS.items():
        path = FONTS_DIR / f"{name}.ttf"
        if not path.exists():
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                data = urllib.request.urlopen(req, timeout=30).read()
                with open(path, 'wb') as f:
                    f.write(data)
                count += 1
            except:
                pass
    if count > 0:
        print(f"📥 Downloaded {count} fonts")


# =============================================================================
# STATIC FILES
# =============================================================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    try:
        return send_from_directory(app.static_folder, path)
    except:
        return "Not found", 404

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)

@app.route('/temp/<path:filename>')
def serve_temp(filename):
    return send_from_directory(str(TEMP_DIR), filename)


# =============================================================================
# API - INFO
# =============================================================================

@app.route('/api/status')
def api_status():
    return jsonify({
        "ready": True,
        "tts": _tts_ready,
        "whisper": _whisper_ready,
        "ffmpeg": _ffmpeg_ready
    })

@app.route('/api/voices')
def api_voices():
    return jsonify(list(VOICES.keys()))

@app.route('/api/fonts')
def api_fonts():
    return jsonify({"flat": FONTS_FLAT, "categories": FONT_CATEGORIES})

@app.route('/api/styles')
def api_styles():
    return jsonify(IMAGE_STYLE_NAMES)

@app.route('/api/models')
def api_models():
    return jsonify(list(IMAGE_MODELS.keys()))

@app.route('/api/resolutions')
def api_resolutions():
    return jsonify(list(IMAGE_RESOLUTIONS.keys()))

@app.route('/api/durations')
def api_durations():
    return jsonify(SCRIPT_DURATIONS)

@app.route('/api/effects')
def api_effects():
    return jsonify(EFFECTS)

@app.route('/api/animations')
def api_animations():
    return jsonify(list(ANIMATIONS.keys()))

@app.route('/api/scene-modes')
def api_scene_modes():
    return jsonify(SCENE_COUNT_MODES)


# =============================================================================
# API - TTS
# =============================================================================

@app.route('/api/preview-voice', methods=['POST'])
def api_preview_voice():
    if not _tts_ready:
        return jsonify({"error": "TTS not available"}), 503
    
    data = request.json
    voice = data.get('voice', 'Adam (Male US)')
    text = data.get('text', 'Hello! This is a preview of my voice.')
    
    try:
        import soundfile as sf
        import numpy as np
        
        voice_id = VOICES.get(voice, 'am_adam')
        
        # KPipeline yields (graphemes, phonemes, audio) tuples
        audio_parts = []
        for _, _, audio in _tts_pipeline(text, voice=voice_id, speed=1.0, split_pattern=r'\n+'):
            if audio is not None and len(audio) > 0:
                audio_parts.append(audio)
        
        if not audio_parts:
            return jsonify({"error": "No audio generated"}), 500
        
        samples = np.concatenate(audio_parts)
        
        preview_path = TEMP_DIR / f"preview_{voice_id}.wav"
        sf.write(str(preview_path), samples, 24000)
        
        return jsonify({"audio": f"/temp/preview_{voice_id}.wav"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API - SCRIPT GENERATION
# =============================================================================

@app.route('/api/generate-script', methods=['POST'])
def api_generate_script():
    data = request.json
    title = data.get('title')
    duration = data.get('duration', '2 minutes')
    style = data.get('style', '')
    groq_key = data.get('groq_api_key')
    
    if not title:
        return jsonify({"error": "Title required"}), 400
    if not groq_key:
        return jsonify({"error": "Groq API key required"}), 400
    
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"You are a viral video scriptwriter. Write engaging, conversational scripts for YouTube. {style}"},
                {"role": "user", "content": f"Write a {duration} video script about: {title}. Write ONLY the spoken words, no directions."}
            ],
            temperature=0.8,
            max_tokens=4000
        )
        
        script = response.choices[0].message.content.strip()
        return jsonify({"script": script})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API - IMAGE GENERATION
# =============================================================================

@app.route('/api/generate-images', methods=['POST'])
def api_generate_images():
    data = request.json
    script = data.get('script')
    groq_key = data.get('groq_api_key')
    together_key = data.get('together_api_key')
    
    if not script or not groq_key or not together_key:
        return jsonify({"error": "Script and API keys required"}), 400
    
    try:
        from groq import Groq
        from openai import OpenAI
        import requests
        
        # Extract scenes using Groq
        groq = Groq(api_key=groq_key)
        num = int(data.get('custom_count', 30))
        
        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Convert script to {num} image prompts. Return JSON array of strings only."},
                {"role": "user", "content": script}
            ],
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        if result.startswith("```"):
            result = result.split("```")[1].lstrip("json").strip()
        prompts = json.loads(result)
        
        # Generate images using Together AI
        together = OpenAI(api_key=together_key, base_url="https://api.together.xyz/v1")
        
        style = data.get('image_style', 'No Style')
        style_prompt = IMAGE_STYLES.get(style, "")
        res = IMAGE_RESOLUTIONS.get(data.get('resolution', '512x768'), {'width': 768, 'height': 512})
        model = IMAGE_MODELS.get(data.get('model', 'Flux Schnell (Free)'), 'black-forest-labs/FLUX.1-schnell-Free')
        
        images = []
        for i, p in enumerate(prompts[:num]):
            try:
                full_prompt = f"{STYLE_HEADER}, {style_prompt}, {p}" if style_prompt else f"{STYLE_HEADER}, {p}"
                
                img_response = together.images.generate(
                    model=model,
                    prompt=full_prompt,
                    n=1,
                    size=f"{res['width']}x{res['height']}"
                )
                
                img_url = img_response.data[0].url
                img_data = requests.get(img_url, timeout=60).content
                
                img_path = TEMP_DIR / f"ai_img_{i}_{int(time.time())}.png"
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                images.append(f"/temp/{img_path.name}")
            except Exception as e:
                print(f"Image {i} failed: {e}")
                images.append(None)
        
        return jsonify({"prompts": prompts[:num], "images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API - VIDEO GENERATION
# =============================================================================

@app.route('/api/generate', methods=['POST'])
def api_generate():
    if not _tts_ready:
        return jsonify({"error": "TTS not available"}), 503
    
    data = request.json
    
    # Debug logging
    print(f"\n📥 Request received:")
    print(f"   image_source: {data.get('image_source')}")
    print(f"   images count: {len(data.get('images', []))}")
    print(f"   groq_api_key: {'SET' if data.get('groq_api_key') else 'NOT SET'}")
    print(f"   together_api_key: {'SET' if data.get('together_api_key') else 'NOT SET'}")
    print(f"   image_model: {data.get('image_model')}")
    
    script = data.get('script')
    if not script or len(script) < 10:
        return jsonify({"error": "Script too short"}), 400
    
    try:
        import soundfile as sf
        import numpy as np
        from services.video_service import generate_video
        from services.transcription_service import generate_captions
        
        voice = data.get('voice', 'Adam (Male US)')
        name = data.get('name', 'Video')
        effect = data.get('effect', 'Zoom Combo')
        
        # Caption settings
        captions_enabled = data.get('captions_enabled', False)
        
        # Generate audio using Kokoro KPipeline
        print("🎙️ Generating audio...")
        voice_id = VOICES.get(voice, 'am_adam')
        
        # KPipeline yields (graphemes, phonemes, audio) tuples
        audio_parts = []
        for _, _, audio in _tts_pipeline(script, voice=voice_id, speed=1.0, split_pattern=r'\n+'):
            if audio is not None and len(audio) > 0:
                audio_parts.append(audio)
        
        if not audio_parts:
            return jsonify({"error": "Audio generation failed"}), 500
        
        samples = np.concatenate(audio_parts)
        sample_rate = 24000
        
        jid = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        audio_path = str(TEMP_DIR / f"audio_{jid}.wav")
        sf.write(audio_path, samples, sample_rate)
        
        duration = len(samples) / sample_rate
        print(f"   ✅ {duration:.1f}s audio")
        
        # Get images - either custom or AI generated
        images = data.get('images', [])
        image_source = data.get('image_source', 'Custom Images')
        
        image_paths = []
        
        if image_source == 'AI Generated':
            # Generate AI images from script
            print("🎨 Generating AI images...")
            groq_key = data.get('groq_api_key')
            image_provider = data.get('image_provider', 'Pollinations AI (Free)')
            
            # Groq is always needed for scene extraction
            if not groq_key:
                return jsonify({"error": "Groq API key required for scene extraction"}), 400
            
            try:
                from services.image_service import extract_scenes_from_script, generate_ai_image, generate_pollination_image
                
                scene_mode = data.get('scene_mode', '10 images')
                image_model = data.get('image_model', 'Flux (Free)')
                image_resolution = data.get('image_resolution', '512x768 (Landscape)')
                image_style = data.get('image_style', 'No Style')
                custom_count = data.get('custom_count', 10)
                
                # Determine number of scenes based on mode
                if scene_mode == 'Custom':
                    num_scenes = int(custom_count)
                elif 'Scene-by-Scene' in scene_mode or 'scene' in scene_mode.lower():
                    # Count sentences in script for Scene-by-Scene mode
                    import re
                    sentences = re.split(r'[.!?]+', script)
                    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                    num_scenes = max(3, min(len(sentences), 20))  # 3-20 images for scene-by-scene
                    print(f"   📝 Scene-by-Scene: Found {len(sentences)} sentences, using {num_scenes} images")
                elif 'images' in scene_mode.lower():
                    # Extract number from "30 images", "50 images", etc.
                    num_scenes = int(scene_mode.split()[0])
                else:
                    num_scenes = 10  # Default to 10 images, not 30
                
                print(f"   🎬 Scene Mode: {scene_mode} → {num_scenes} images")
                
                # Extract scenes using Groq
                prompts = extract_scenes_from_script(script, num_scenes, groq_key, image_style)
                
                # Get resolution dimensions
                res = IMAGE_RESOLUTIONS.get(image_resolution, {'width': 768, 'height': 512})
                
                # Generate images based on provider
                print(f"\n🎨 AI Image Generation: {len(prompts)} images")
                print(f"   🔧 Provider: {image_provider}")
                print(f"   🎭 Style: {image_style}")
                
                ai_images = []
                for i, prompt in enumerate(prompts, 1):
                    print(f"   [{i}/{len(prompts)}] Generating...")
                    
                    if 'Pollinations' in image_provider:
                        # Use FREE Pollinations AI (optional API key for higher limits)
                        pollinations_key = data.get('pollinations_api_key', '')
                        img_path = generate_pollination_image(
                            prompt=prompt,
                            width=res['width'],
                            height=res['height'],
                            model=IMAGE_MODELS.get('Pollinations AI (Free)', {}).get(image_model, 'flux'),
                            api_key=pollinations_key
                        )
                    else:
                        # Use Together AI (requires API key)
                        together_key = data.get('together_api_key')
                        if not together_key:
                            return jsonify({"error": "Together AI API key required for Together AI images"}), 400
                        
                        img_path = generate_ai_image(
                            prompt=prompt,
                            model=image_model,
                            resolution=image_resolution,
                            together_api_key=together_key
                        )
                    
                    ai_images.append(img_path)
                
                # Filter out None values from failed generations
                image_paths = [img for img in ai_images if img]
                
                if not image_paths:
                    return jsonify({"error": "AI image generation failed - no images created"}), 500
                
                print(f"   ✅ Generated {len(image_paths)} AI images")
                
            except Exception as img_error:
                return jsonify({"error": f"AI image generation failed: {str(img_error)}"}), 500
        else:
            # Custom images - convert URLs to paths
            if not images:
                return jsonify({"error": "No images provided"}), 400
            
            for img in images:
                if img.startswith('/temp/'):
                    img_path = str(TEMP_DIR / img.split('/')[-1])
                else:
                    img_path = str(img)
                image_paths.append(img_path)
        
        # Generate captions if enabled
        subtitle_path = None
        if captions_enabled:
            print("📝 Generating captions...")
            subtitle_path = generate_captions(
                audio_path=audio_path,
                caption_mode=data.get('caption_mode', 'single'),
                words_per_group=data.get('words_per_group', 3),
                font=data.get('font', 'Arial'),
                fontsize=data.get('fontsize', 48),
                bold=data.get('bold', True),
                uppercase=data.get('uppercase', True),
                text_color=data.get('text_color', '#FFFFFF'),
                highlight_color=data.get('highlight_color', '#FFFF00'),
                outline_color=data.get('outline_color', '#000000'),
                outline_size=data.get('outline_size', 3),
                shadow_on=data.get('shadow_on', True),
                shadow_color=data.get('shadow_color', '#000000'),
                shadow_depth=data.get('shadow_depth', 2),
                bg_on=data.get('bg_on', False),
                bg_color=data.get('bg_color', '#000000'),
                bg_opacity=data.get('bg_opacity', 80),
                position=data.get('position', 'bottom'),
                margin_v=data.get('margin_v', 50),
                animation=data.get('animation', 'Pop')
            )
        
        # Music settings
        music_path = data.get('music_path')
        music_volume = data.get('music_volume', 20)
        
        # Generate video with effects and transitions
        print("🎬 Creating video...")
        final_path = generate_video(
            images=image_paths,
            audio_path=audio_path,
            total_duration=duration,
            effect=effect,
            subtitle_path=subtitle_path,
            music_path=music_path,
            music_volume=music_volume,
            output_name=name
        )
        
        # Get just the filename for the URL
        output_filename = os.path.basename(final_path)
        print(f"✅ Video: {output_filename}")
        
        return jsonify({
            "success": True,
            "video": f"/output/{output_filename}",
            "duration": duration
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# API - FILE UPLOAD
# =============================================================================

@app.route('/api/upload/images', methods=['POST'])
def api_upload_images():
    if 'files' not in request.files:
        return jsonify({"error": "No files"}), 400
    
    files = request.files.getlist('files')
    paths = []
    urls = []
    
    for f in files:
        if f.filename:
            name = f"upload_{int(time.time())}_{f.filename}"
            path = TEMP_DIR / name
            f.save(str(path))
            paths.append(str(path))
            urls.append(f"/temp/{name}")
    
    return jsonify({"success": True, "paths": paths, "urls": urls})


@app.route('/api/upload/music', methods=['POST'])
def api_upload_music():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    f = request.files['file']
    if f.filename:
        name = f"music_{int(time.time())}_{f.filename}"
        path = TEMP_DIR / name
        f.save(str(path))
        return jsonify({"success": True, "path": str(path), "url": f"/temp/{name}"})
    
    return jsonify({"error": "Invalid"}), 400


# =============================================================================
# API - QUEUE
# =============================================================================

@app.route('/api/queue/add', methods=['POST'])
def api_queue_add():
    data = request.json
    bulk_queue.append({'id': len(bulk_queue), 'name': data.get('name', 'Video'), 'status': 'queued'})
    return jsonify({"success": True})

@app.route('/api/queue/status')
def api_queue_status():
    return jsonify({"queue": bulk_queue, "results": bulk_results})

@app.route('/api/queue/clear', methods=['POST'])
def api_queue_clear():
    bulk_queue.clear()
    bulk_results.clear()
    return jsonify({"success": True})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 50)
    print("  VIDZEO LOCAL - AI Video Generator")
    print("=" * 50)
    print()
    
    # Initialize services
    download_fonts()
    init_tts()
    init_whisper()  # Local whisper for app.py status
    check_ffmpeg()
    
    # Initialize video service (GPU detection)
    from services.video_service import init_video_service
    init_video_service()
    
    # Initialize transcription service for captions
    from services.transcription_service import init_whisper as init_transcription_whisper
    init_transcription_whisper("cpu")
    
    print()
    print("  Server: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print()
    print("=" * 50)
    print()
    
    # Open browser automatically
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open('http://localhost:5000')
        print("🌍 Browser opened automatically")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
