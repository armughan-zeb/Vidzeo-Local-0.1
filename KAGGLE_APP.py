"""
🎬 KAGGLE VIDEO GENERATOR - ADVANCED AUTO-CAPTIONS v3
Fixed: Image freeze + Jitter + Perfect sync

Features:
- One Word at a Time / Karaoke / Line modes
- FIXED: No last image freeze (dynamic duration)
- FIXED: Smooth crossfade transitions (no jitter)
- Full styling: fonts, colors, shadows, borders
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import subprocess
import time
import hashlib
import numpy as np
import json
import base64
import io
import requests
from pathlib import Path
from PIL import Image
import gradio as gr

# Optional AI image generation imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed - AI scene extraction disabled")

try:
    from openai import OpenAI
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("⚠️ OpenAI package not installed - Together AI disabled")

# =============================================================================
# SETUP
# =============================================================================

BASE = '/kaggle/working' if os.path.exists('/kaggle') else '/content'
KOKORO_DIR = f"{BASE}/Kokoro-TTS-Subtitle"

# Auto-clone Kokoro if not exists (makes app self-contained)
if not os.path.exists(KOKORO_DIR):
    print("📦 Kokoro not found - cloning...")
    subprocess.run(['git', 'clone', '-q', 'https://github.com/NeuralFalconYT/Kokoro-TTS-Subtitle.git', KOKORO_DIR], 
                   capture_output=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kokoro>=0.9.4'], 
                   capture_output=True, timeout=180)
    print("✅ Kokoro installed")

sys.path.insert(0, KOKORO_DIR)
os.chdir(KOKORO_DIR)

DIRS = {'output': f'{BASE}/video_outputs', 'temp': f'{BASE}/video_temp'}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# =============================================================================
# GPU
# =============================================================================

def get_gpu():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split('\n')[0]
            nv = 'h264_nvenc' in subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                                capture_output=True, text=True).stdout
            print(f"✅ GPU: {name}")
            return name, 'h264_nvenc' if nv else 'libx264', True
    except:
        pass
    print("⚠️ CPU mode")
    return 'CPU', 'libx264', False

GPU_NAME, ENCODER, HAS_GPU = get_gpu()

# =============================================================================
# KOKORO TTS
# =============================================================================

print("\n🧠 Loading Kokoro...")
try:
    from kokoro import KPipeline
    import soundfile as sf
    PIPE = KPipeline(lang_code='a')
    KOK = True
    print("✅ Kokoro ready")
except Exception as e:
    print(f"❌ Kokoro: {e}")
    PIPE = None
    KOK = False

VOICES = {
    # American Female
    'Sky (Female US)': 'af_sky',
    'Bella (Female US)': 'af_bella',
    'Sarah (Female US)': 'af_sarah',
    # American Male
    'Adam (Male US)': 'am_adam',
    'Michael (Male US)': 'am_michael',
    'Fenrir (Male US)': 'am_fenrir',
    'Echo (Male US)': 'am_echo',
    'Eric (Male US)': 'am_eric',
    'Liam (Male US)': 'am_liam',
    'Onyx (Male US)': 'am_onyx',
    'Puck (Male US)': 'am_puck',
    'Santa (Male US)': 'am_santa',
    # British Male
    'George (Male UK)': 'bm_george',
    'Lewis (Male UK)': 'bm_lewis',
    'Daniel (Male UK)': 'bm_daniel',
    'Fable (Male UK)': 'bm_fable',
}

# =============================================================================
# WHISPER
# =============================================================================

print("\n📝 Loading Whisper...")
try:
    import whisper
    WHISPER_MODEL = None
    WHIS = True
    print("✅ Whisper available")
except:
    WHIS = False
    print("❌ Whisper not available")

def load_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None and WHIS:
        device = "cuda" if HAS_GPU else "cpu"
        print(f"   Loading whisper ({device})...")
        WHISPER_MODEL = whisper.load_model("base", device=device)
        print("   ✅ Loaded")
    return WHISPER_MODEL

# =============================================================================
# AI IMAGE GENERATION
# =============================================================================

# Available image models
IMAGE_MODELS = {
    'DreamShaper XL': 'dreamshaper-xl-v2-turbo',
    'Flux Schnell (Free)': 'black-forest-labs/FLUX.1-schnell-Free'
}

IMAGE_RESOLUTIONS = {
    '512x768 (Landscape)': {'width': 768, 'height': 512},
    '1024x576 (Landscape HD)': {'width': 1024, 'height': 576},
}

SCENE_COUNT_MODES = [
    'Scene-by-Scene (1 per sentence)',
    '30 images',
    '50 images',
    '80 images',
    'Custom'
]

# Global style lock header (prepended to all prompts)
STYLE_HEADER = """MASTERPIECE, HIGH QUALITY,
SINGLE CLEAR SUBJECT,
CENTERED COMPOSITION,
NO TEXT, NO LOGOS, NO SYMBOLS,
CLEAN EDGES, STABLE STRUCTURE,
CONSISTENT STYLE THROUGHOUT"""

# 20 Image Style Presets for AI-generated scenes
IMAGE_STYLES = {
    "No Style": "",
    
    "2D Cartoon (Flat)": """2D CARTOON ILLUSTRATION,
FLAT VECTOR LOOK,
SOLID COLOR AREAS ONLY,
THICK BLACK OUTLINES,
NO DEPTH CUES,
NO GRADIENTS,
SIMPLE GEOMETRIC FORMS,
BRIGHT CARTOON PALETTE,
EXAGGERATED EXPRESSIONS,
PLAIN BACKGROUND""",
    
    "3D Cartoon (Toy-like)": """3D CARTOON CHARACTER,
SMOOTH PLASTIC SURFACE,
ROUNDED SHAPES,
STYLIZED NON-REALISTIC PROPORTIONS,
SOFT STUDIO LIGHTING,
GLOBAL ILLUMINATION,
SATURATED COLORS,
CLEAN STUDIO BACKGROUND""",
    
    "2D Stickman Pencil": """2D STICKMAN DRAWING,
BLACK PENCIL ON WHITE PAPER,
CIRCULAR HEAD,
STRAIGHT LINE LIMBS,
NO FILL COLOR,
NO SHADING,
HAND-DRAWN SKETCH STYLE""",
    
    "2D Stickman Cartoon": """2D STICKMAN CARTOON,
THICK BLACK OUTLINES,
SIMPLE CIRCLE HEAD,
STRAIGHT LINE ARMS AND LEGS,
FLAT SOLID COLORS,
MINIMAL CARTOON FACE,
CLEAN SHAPES,
SIMPLE BACKGROUND""",
    
    "3D Stickman Cartoon": """3D STICKMAN CARTOON CHARACTER,
SPHERE HEAD,
CYLINDER LIMBS,
SMOOTH MATTE MATERIAL,
BRIGHT PLAYFUL COLORS,
NON-HUMAN PROPORTIONS,
SOFT STUDIO LIGHTING""",
    
    "Pencil Sketch (Fine Art)": """REALISTIC PENCIL SKETCH,
BLACK AND WHITE ONLY,
GRAPHITE LINE WORK,
CROSS-HATCH SHADING,
VISIBLE PAPER TEXTURE,
HAND-DRAWN ART STYLE""",
    
    "Photorealistic": """PHOTOREALISTIC IMAGE,
REAL CAMERA PHOTOGRAPH,
NATURAL LIGHT INTERACTION,
REALISTIC SURFACE TEXTURE,
HIGH DYNAMIC RANGE,
SHALLOW DEPTH OF FIELD,
85MM LENS LOOK""",
    
    "Cinematic Realism": """CINEMATIC REALISM,
MOVIE STILL FRAME,
DRAMATIC LIGHT DIRECTION,
CONTROLLED COLOR GRADING,
VOLUMETRIC LIGHT RAYS,
ANAMORPHIC LENS FEEL,
EPIC FRAMING""",
    
    "1950s Vintage Photo": """1950s VINTAGE PHOTOGRAPH,
MONOCHROME OR FADED COLOR,
VISIBLE FILM GRAIN,
SOFT FOCUS LENS,
LOW CONTRAST,
AGED PHOTO CHARACTER""",
    
    "Comic Book": """COMIC BOOK ILLUSTRATION,
BOLD INKED OUTLINES,
HALFTONE DOT SHADING,
HIGH CONTRAST COLORS,
DRAMATIC POSE,
DYNAMIC COMPOSITION""",
    
    "Anime (Cel Shading)": """ANIME STYLE ILLUSTRATION,
CLEAN LINE ART,
CEL SHADING,
FLAT COLOR ZONES,
LARGE EXPRESSIVE EYES,
STYLIZED PROPORTIONS""",
    
    "Studio Ghibli": """STUDIO GHIBLI STYLE,
HAND-PAINTED LOOK,
SOFT PASTEL COLORS,
GENTLE LIGHTING,
WATERCOLOR-LIKE TEXTURE,
DREAMY ATMOSPHERE""",
    
    "Horror": """HORROR ART STYLE,
DARK TONAL PALETTE,
LOW-KEY LIGHTING,
HEAVY SHADOWS,
FOG AND GRAIN,
TENSE ATMOSPHERE""",
    
    "Medieval Art": """MEDIEVAL ILLUSTRATION,
ILLUMINATED MANUSCRIPT STYLE,
FLAT PERSPECTIVE,
MUTED EARTH TONES,
PARCHMENT TEXTURE,
DECORATIVE DETAILS""",
    
    "Furry Art": """FURRY ART STYLE,
ANTHROPOMORPHIC ANIMAL CHARACTER,
SOFT FUR DETAIL,
EXPRESSIVE FACE,
DIGITAL CHARACTER ILLUSTRATION""",
    
    "Digital Illustration": """PROFESSIONAL DIGITAL ILLUSTRATION,
CLEAR SUBJECT SEPARATION,
STYLIZED DESIGN LANGUAGE,
BALANCED COLOR PALETTE,
CLEAN COMPOSITION""",
    
    "Watercolor Painting": """WATERCOLOR PAINTING,
SOFT COLOR BLEEDING,
VISIBLE PAPER GRAIN,
LIGHT WASHES,
ORGANIC EDGES,
HAND-PAINTED LOOK""",
    
    "B&W Historical Photo": """BLACK AND WHITE HISTORICAL PHOTOGRAPH,
DOCUMENTARY REALISM,
NATURAL LIGHT,
VISIBLE FILM GRAIN,
CLASSIC CONTRAST""",
    
    "B&W Pencil Sketch": """BLACK AND WHITE PENCIL SKETCH,
FINE GRAPHITE LINES,
REALISTIC SHADING,
HAND-DRAWN STYLE,
WHITE PAPER BACKGROUND""",
    
    "Oil Painting": """OIL PAINTING,
SOFT DISTORTED BRUSH STROKES,
IMPRESSIONISTIC STYLE,
BLENDED COLORS,
VISIBLE CANVAS TEXTURE,
ARTISTIC COMPOSITION"""
}

# List of style names for dropdown
IMAGE_STYLE_NAMES = list(IMAGE_STYLES.keys())

# =============================================================================
# AI SCRIPT GENERATION - Title to Script with Research
# =============================================================================

from datetime import datetime

# Video duration options (in minutes)
SCRIPT_DURATIONS = [
    "30 seconds", "1 minute", "2 minutes", "3 minutes", "5 minutes",
    "10 minutes", "15 minutes", "30 minutes", "45 minutes",
    "1 hour", "1.5 hours", "2 hours", "3 hours"
]

# Words per minute (average speaking pace)
WORDS_PER_MINUTE = 150

# Keywords that trigger automatic research (current data needed)
RESEARCH_TRIGGERS = [
    # News & Events
    "news", "latest", "recent", "current", "today", "2024", "2025", "update",
    "breaking", "new", "trending", "viral",
    # Facts & Statistics  
    "facts", "statistics", "stats", "data", "numbers", "study", "research",
    "percent", "million", "billion", "how many", "how much",
    # Technology & Science
    "ai", "technology", "tech", "science", "discovery", "invention", "innovation",
    "crypto", "bitcoin", "blockchain", "smartphone", "app", "software",
    # Business & Economy
    "market", "economy", "stock", "price", "cost", "investment", "business",
    "company", "startup", "billionaire", "richest",
    # World Events
    "country", "countries", "world", "global", "war", "election", "president",
    "government", "politics", "climate", "environment",
    # Pop Culture
    "movie", "film", "celebrity", "famous", "star", "singer", "actor",
    "sport", "championship", "record", "winner"
]

def get_duration_minutes(duration_str):
    """Convert duration string to minutes"""
    if "second" in duration_str:
        return int(duration_str.split()[0]) / 60
    elif "hour" in duration_str:
        hours = float(duration_str.split()[0])
        return hours * 60
    else:
        return int(duration_str.split()[0])

def needs_research(title, style_prompt=""):
    """
    Auto-detect if topic needs current/fresh data from web search.
    Returns True if the topic likely needs recent facts, news, or statistics.
    """
    text = f"{title} {style_prompt}".lower()
    
    # Check for year references (2023+)
    import re
    if re.search(r'\b20[2-9][3-9]\b', text):
        return True
    
    # Check trigger keywords
    for trigger in RESEARCH_TRIGGERS:
        if trigger in text:
            return True
    
    return False

def search_web_for_facts(query, max_results=5):
    """Search DuckDuckGo for relevant facts and figures"""
    try:
        from duckduckgo_search import DDGS
        
        # Add "2024" or "latest" to get recent results
        current_year = datetime.now().year
        search_query = f"{query} {current_year}"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        
        facts = []
        for r in results:
            facts.append({
                'title': r.get('title', ''),
                'body': r.get('body', ''),
                'href': r.get('href', '')
            })
        
        # Format as context
        context = "\n\n".join([
            f"**{f['title']}**\n{f['body']}"
            for f in facts if f['body']
        ])
        
        print(f"   📚 Found {len(facts)} research sources")
        return context
    except Exception as e:
        print(f"   ⚠️ Web search failed: {e}")
        return ""

def generate_script_chunk(groq_client, title, style_prompt, chunk_num, total_chunks, 
                          previous_summary, target_words, research_context=""):
    """Generate a single chunk of the script with context continuity"""
    
    chunk_words = target_words // total_chunks
    
    # Get current date for context
    current_date = datetime.now().strftime("%B %Y")  # e.g., "December 2024"
    current_year = datetime.now().year
    
    # Build the prompt
    if chunk_num == 1:
        position_context = f"""
This is the BEGINNING of the script (Part 1/{total_chunks}).
Start with a powerful HOOK that grabs attention in the first 3 seconds.
Introduce the topic and build initial curiosity."""
    elif chunk_num == total_chunks:
        position_context = f"""
This is the ENDING of the script (Part {chunk_num}/{total_chunks}).
Previous content summary: {previous_summary}
Build to a climax, deliver the key insight, and end with a strong CALL TO ACTION."""
    else:
        position_context = f"""
This is the MIDDLE of the script (Part {chunk_num}/{total_chunks}).
Previous content summary: {previous_summary}
Continue the story naturally, add depth and new information."""
    
    research_section = ""
    if research_context:
        research_section = f"""

CURRENT RESEARCH & FACTS (use relevant ones naturally):
{research_context[:2500]}
"""
    
    system_prompt = f"""You are an expert viral video scriptwriter creating content in {current_date}.

IMPORTANT CONTEXT:
- Today's date is {current_date}
- You are writing for a modern {current_year} audience
- Use contemporary language, trends, and references
- Any statistics or facts should be presented as current (use research provided)
- Avoid outdated references or information

STYLE GUIDELINES:
- Use short, punchy sentences that feel fresh and modern
- Add suspense and curiosity (hooks throughout)
- Include storytelling elements with relatable examples
- Use emotional hooks that resonate with today's audience
- Write in second person ("you") to connect with viewer
- No stage directions, just the spoken words
- No emojis or special characters
- Natural conversational flow like a friend talking
- Reference current trends/culture when relevant

{f'USER STYLE PREFERENCE: {style_prompt}' if style_prompt else ''}

{position_context}
{research_section}"""

    user_prompt = f"""Write approximately {chunk_words} words for a video script about:

TITLE: {title}

Write ONLY the spoken script text. No headings, no formatting marks, just the natural speech.
Make it engaging, informative, modern, and keep viewers watching.
If you mention any statistics or facts, present them as current and relevant."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=min(chunk_words * 2, 8000)  # Buffer for tokens
        )
        
        script_text = response.choices[0].message.content.strip()
        return script_text
    except Exception as e:
        raise ValueError(f"Script generation failed: {str(e)}")

def summarize_chunk(groq_client, script_chunk):
    """Create a brief summary of a script chunk for continuity"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize the key points of this script section in 2-3 sentences for continuity with the next section."},
                {"role": "user", "content": script_chunk}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return ""

def generate_script_from_title(title, duration, style_prompt, groq_api_key, use_research="auto"):
    """
    Generate a complete video script from a title.
    
    Features:
    - Duration-based length calculation
    - AUTO web research (detects when needed based on topic)
    - Current date injection (model knows it's 2024/2025)
    - Smart chunking for long scripts
    - Quality script elements (hooks, suspense, CTAs)
    
    Args:
        use_research: "auto" (smart detection), True (always), False (never)
    
    Returns: Generated script text
    """
    if not GROQ_AVAILABLE or not groq_api_key:
        raise ValueError("Groq API key required for script generation")
    
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    
    # Calculate target word count
    duration_mins = get_duration_minutes(duration)
    target_words = int(duration_mins * WORDS_PER_MINUTE)
    
    current_date = datetime.now().strftime("%B %Y")
    
    print(f"\n📝 Generating Script...")
    print(f"   Title: {title}")
    print(f"   Duration: {duration} (~{target_words} words)")
    print(f"   Date Context: {current_date}")
    
    # Smart research detection
    research_context = ""
    should_research = False
    
    if use_research == "auto" or use_research == True:
        should_research = needs_research(title, style_prompt) if use_research == "auto" else True
        
        if should_research:
            print(f"   🔍 Auto-detected: Topic needs current data, researching...")
            research_context = search_web_for_facts(f"{title} facts statistics", max_results=5)
        else:
            print(f"   ℹ️ Topic doesn't require fresh data, using model knowledge")
    
    # Determine chunking strategy
    MAX_WORDS_PER_CHUNK = 800  # ~1000 tokens
    
    if target_words <= MAX_WORDS_PER_CHUNK:
        total_chunks = 1
    else:
        total_chunks = (target_words // MAX_WORDS_PER_CHUNK) + 1
    
    print(f"   📊 Generating in {total_chunks} part(s)...")
    
    # Generate script chunks
    script_parts = []
    previous_summary = ""
    
    for chunk_num in range(1, total_chunks + 1):
        print(f"   ✍️ Writing part {chunk_num}/{total_chunks}...")
        
        chunk_text = generate_script_chunk(
            client, title, style_prompt, chunk_num, total_chunks,
            previous_summary, target_words, research_context
        )
        
        script_parts.append(chunk_text)
        
        # Get summary for continuity (except last chunk)
        if chunk_num < total_chunks:
            previous_summary = summarize_chunk(client, chunk_text)
    
    # Combine all parts
    full_script = "\n\n".join(script_parts)
    
    word_count = len(full_script.split())
    print(f"   ✅ Script complete: {word_count} words")
    
    return full_script

# =============================================================================
# NSFW SAFETY PIPELINE - Industry Standard Protection
# =============================================================================

# Step 1: Prompt Sanitization - Block obvious NSFW words BEFORE generation
BLOCK_WORDS = [
    "nude", "naked", "sex", "sexual", "erotic", "porn", "pornographic",
    "fetish", "boobs", "breasts", "ass", "lingerie", "bikini", "underwear",
    "nsfw", "xxx", "adult", "explicit", "sensual", "seductive", "intimate",
    "provocative", "revealing", "topless", "bottomless", "stripper", "escort"
]

def is_prompt_safe(prompt: str) -> bool:
    """Check if prompt contains blocked NSFW words"""
    p = prompt.lower()
    return not any(word in p for word in BLOCK_WORDS)

def sanitize_prompt(prompt: str) -> str:
    """Remove blocked words from prompt and add safe prefix"""
    p = prompt
    for word in BLOCK_WORDS:
        p = p.lower().replace(word, "")
    return p.strip()

# Step 2: Safe Positive Constraints (append to every prompt)
SAFE_CONTENT_PREFIX = (
    "professional photography, family-friendly, PG-13, modest clothing, "
    "no suggestive content, appropriate for all ages, tasteful composition, "
    "high quality stock photo style"
)

# Step 3: Visual Quality Negatives ONLY (no NSFW words!)
QUALITY_NEGATIVES = "blurry, low quality, deformed, bad anatomy, watermark, text"

# Step 4: OpenNSFW2 Image Detection
NSFW_CLASSIFIER = None
NSFW_THRESHOLD = 0.3  # Discard images above this score
MAX_REGEN_TRIES = 3   # Auto-regenerate up to 3 times

def load_nsfw_classifier():
    """Load OpenNSFW2 model for image-level detection"""
    global NSFW_CLASSIFIER
    if NSFW_CLASSIFIER is None:
        try:
            from opennsfw2 import predict_image
            NSFW_CLASSIFIER = predict_image
            print("✅ OpenNSFW2 loaded - Image safety enabled")
        except ImportError:
            print("⚠️ OpenNSFW2 not installed - Image safety disabled")
            print("   Run: pip install opennsfw2")
            NSFW_CLASSIFIER = None
    return NSFW_CLASSIFIER

def classify_nsfw(image_path: str) -> float:
    """Get NSFW score for an image (0.0 = safe, 1.0 = explicit)"""
    classifier = load_nsfw_classifier()
    if classifier is None:
        return 0.0  # Assume safe if classifier not available
    
    try:
        score = classifier(image_path)
        # Handle if it returns a dict or tuple
        if isinstance(score, dict):
            score = score.get('nsfw', score.get('porn', 0.0))
        elif isinstance(score, (list, tuple)):
            score = float(score[0]) if len(score) > 0 else 0.0
        return float(score)
    except Exception as e:
        print(f"   ⚠️ NSFW check failed: {e}")
        return 0.0

def is_image_safe(image_path: str) -> bool:
    """Check if image passes NSFW threshold"""
    score = classify_nsfw(image_path)
    is_safe = score < NSFW_THRESHOLD
    if not is_safe:
        print(f"   ⚠️ NSFW detected (score: {score:.2f}) - Will regenerate")
    return is_safe

def extract_scenes_from_script(script: str, num_scenes: int, groq_api_key: str, style_description: str = "") -> list:
    """
    Use Groq Llama 3.1 8B to split script into scenes with image prompts.
    Returns list of image prompts.
    """
    if not GROQ_AVAILABLE or not groq_api_key:
        raise ValueError("Groq API key required for AI scene extraction")
    
    try:
        client = Groq(api_key=groq_api_key)
        
        style_instruction = ""
        if style_description:
            style_instruction = f"\n\nIMPORTANT: All image prompts must incorporate this style: {style_description}"
        
        system_prompt = f"""You are an expert at converting video scripts into visual scene descriptions for AI image generation.

Your task: Split the given script into exactly {num_scenes} scenes and create a detailed image prompt for each scene.

Rules:
1. Each prompt should be 1-2 sentences describing a single visual scene
2. Include visual details: setting, lighting, colors, mood, composition
3. Make prompts suitable for AI image generation (describe what to draw, not actions)
4. Distribute the script content evenly across all {num_scenes} scenes
5. Return ONLY a JSON array of strings, each string being one image prompt
6. No additional text, explanations, or formatting - just the JSON array
7. IMPORTANT: All scenes must be professional and family-friendly. Focus on: landscapes, nature, architecture, technology, business settings, food, travel, abstract concepts, and lifestyle imagery. Think "stock photography for YouTube"{style_instruction}

Example output format:
["A serene mountain landscape at sunset with golden light", "A cozy coffee shop interior with warm ambient lighting", ...]"""

        user_prompt = f"Script to convert into {num_scenes} image prompts:\n\n{script}"
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON array
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        prompts = json.loads(result_text)
        
        if not isinstance(prompts, list):
            raise ValueError("Expected JSON array of prompts")
        
        print(f"   ✅ Extracted {len(prompts)} scene prompts")
        return prompts
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON parse error: {e}")
        # Fallback: split by newlines or sentences
        lines = [l.strip() for l in script.split('.') if l.strip()]
        step = max(1, len(lines) // num_scenes)
        prompts = []
        for i in range(0, len(lines), step):
            chunk = '. '.join(lines[i:i+step])
            prompts.append(f"Illustration of: {chunk[:200]}")
        return prompts[:num_scenes]
    except Exception as e:
        raise ValueError(f"Scene extraction failed: {str(e)}")


def generate_ai_image(prompt: str, model: str, resolution: str, together_api_key: str, style_ref_base64: str = None) -> str:
    """
    Generate image using Together AI with FULL SAFETY PIPELINE:
    1. Prompt sanitization check
    2. Safe prefix addition
    3. Image generation
    4. NSFW detection
    5. Auto-regenerate if unsafe (up to MAX_REGEN_TRIES)
    
    Returns path to SAFE image, or None if generation fails.
    """
    if not TOGETHER_AVAILABLE or not together_api_key:
        raise ValueError("Together AI API key required for image generation")
    
    # Step 1: Check and sanitize prompt
    if not is_prompt_safe(prompt):
        print(f"   ⚠️ Unsafe prompt detected - sanitizing...")
        prompt = sanitize_prompt(prompt)
        if not prompt:
            print(f"   ❌ Prompt completely unsafe - skipping")
            return None
    
    try:
        client = OpenAI(
            api_key=together_api_key,
            base_url="https://api.together.xyz/v1"
        )
        
        res = IMAGE_RESOLUTIONS.get(resolution, {'width': 768, 'height': 512})
        model_id = IMAGE_MODELS.get(model, 'black-forest-labs/FLUX.1-schnell-Free')
        
        # Step 2: Build safe prompt (positive constraints only)
        enhanced_prompt = prompt
        if style_ref_base64:
            enhanced_prompt = f"{prompt}, similar artistic style to reference image"
        
        # Prepend safe content prefix
        safe_prompt = f"{SAFE_CONTENT_PREFIX}, {enhanced_prompt}"
        
        print(f"   🎨 Generating: {model} ({res['width']}x{res['height']})...")
        
        # Step 3-5: Generate with auto-regeneration loop
        for attempt in range(MAX_REGEN_TRIES):
            try:
                response = client.images.generate(
                    model=model_id,
                    prompt=safe_prompt,
                    n=1,
                    size=f"{res['width']}x{res['height']}"
                )
                
                image_url = response.data[0].url
                
                # Download image
                img_response = requests.get(image_url, timeout=60)
                img_response.raise_for_status()
                
                # Save to temp directory
                img_hash = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
                img_path = f"{DIRS['temp']}/ai_img_{img_hash}_{int(time.time())}.png"
                
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                
                # Step 4: NSFW Detection
                if is_image_safe(img_path):
                    print(f"   ✅ Safe image saved: {os.path.basename(img_path)}")
                    return img_path
                else:
                    # Delete unsafe image and retry
                    print(f"   🔄 Regenerating (attempt {attempt + 2}/{MAX_REGEN_TRIES})...")
                    try:
                        os.remove(img_path)
                    except:
                        pass
                    
            except Exception as gen_error:
                print(f"   ⚠️ Generation attempt {attempt + 1} failed: {gen_error}")
                if attempt < MAX_REGEN_TRIES - 1:
                    time.sleep(1)  # Brief pause before retry
        
        # All attempts failed
        print(f"   ❌ Could not generate safe image after {MAX_REGEN_TRIES} attempts")
        return None
        
    except Exception as e:
        raise ValueError(f"Image generation failed: {str(e)}")


def generate_all_ai_images(script: str, num_scenes: int, scene_mode: str, 
                           groq_api_key: str, together_api_key: str,
                           model: str, resolution: str,
                           image_style: str = "No Style",
                           prog=None) -> tuple:
    """
    Full pipeline: Script → Scenes → Styled Images
    Returns tuple: (list of prompts, list of image paths)
    """
    
    # Determine number of scenes
    if scene_mode == 'Scene-by-Scene (1 per sentence)':
        # Count sentences roughly
        sentences = [s.strip() for s in script.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        num_scenes = len(sentences)
        num_scenes = max(3, min(num_scenes, 100))  # Clamp between 3-100
    elif 'images' in scene_mode:
        num_scenes = int(scene_mode.split()[0])
    # else: use custom num_scenes
    
    print(f"\n🎨 AI Image Generation: {num_scenes} images")
    print(f"   🎭 Style: {image_style}")
    
    # Step 1: Extract basic scene prompts
    if prog:
        prog(0.1, "Extracting scenes from script...")
    print("   📝 Extracting scenes...")
    raw_prompts = extract_scenes_from_script(script, num_scenes, groq_api_key, "")  # No style in extraction
    
    # Step 2: Apply style preset to all prompts
    style_prompt = IMAGE_STYLES.get(image_style, "")
    styled_prompts = []
    for p in raw_prompts:
        if style_prompt:
            full_prompt = f"{STYLE_HEADER}, {style_prompt}, SUBJECT: {p}"
        else:
            full_prompt = f"{STYLE_HEADER}, SUBJECT: {p}"
        styled_prompts.append(full_prompt)
    
    # Step 3: Generate images
    images = []
    total = len(styled_prompts)
    for i, prompt in enumerate(styled_prompts):
        if prog:
            prog(0.1 + 0.4 * (i / total), f"Generating image {i+1}/{total}...")
        try:
            img_path = generate_ai_image(prompt, model, resolution, together_api_key, None)
            images.append(img_path)
        except Exception as e:
            print(f"   ⚠️ Failed image {i+1}: {e}")
            images.append(None)  # Keep placeholder for failed images
            continue
    
    print(f"   ✅ Generated {len([i for i in images if i])}/{total} images")
    return raw_prompts, images  # Return raw prompts for editing, styled images for video


# =============================================================================
# SCENE EDITOR - Preview and Edit AI Generated Scenes
# =============================================================================

# Global state for scene editor
scene_editor_state = {
    'prompts': [],
    'images': [],
    'settings': {}  # Store generation settings for regeneration
}

def generate_scenes_for_editor(script, groq_key, together_key, model, resolution, 
                                scene_mode, custom_count, style_ref, style_desc,
                                prog=gr.Progress()):
    """Generate scenes and return gallery data for editing"""
    global scene_editor_state
    
    if not script or len(script) < 10:
        return None, "❌ Script too short (min 10 chars)", "", 0
    
    if not groq_key or not together_key:
        return None, "❌ Both API keys required", "", 0
    
    try:
        # Determine scene count
        num_scenes = int(custom_count) if scene_mode == 'Custom' else 30
        
        prog(0.05, "Starting scene generation...")
        
        # Store settings for regeneration
        scene_editor_state['settings'] = {
            'groq_key': groq_key,
            'together_key': together_key,
            'model': model,
            'resolution': resolution,
            'style_desc': style_desc,
            'style_ref': style_ref
        }
        
        # Generate all scenes
        prompts, images = generate_all_ai_images(
            script=script,
            num_scenes=num_scenes,
            scene_mode=scene_mode,
            groq_api_key=groq_key,
            together_api_key=together_key,
            model=model,
            resolution=resolution,
            style_description=style_desc,
            style_ref_image=style_ref,
            prog=prog
        )
        
        # Store in state
        scene_editor_state['prompts'] = prompts
        scene_editor_state['images'] = images
        
        # Create gallery data (list of tuples: (image_path, caption))
        gallery_data = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            if img and os.path.exists(img):
                caption = f"Scene {i+1}"
                gallery_data.append((img, caption))
        
        prog(1.0, "Done!")
        
        success_count = len([i for i in images if i])
        status = f"✅ Generated {success_count}/{len(prompts)} scenes\n\n📝 Click a scene to view/edit its prompt"
        
        # Return first prompt for editing
        first_prompt = prompts[0] if prompts else ""
        
        return gallery_data, status, first_prompt, 0
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}", "", 0


def on_scene_select(evt: gr.SelectData):
    """Handle scene selection from gallery"""
    global scene_editor_state
    
    idx = evt.index
    prompts = scene_editor_state.get('prompts', [])
    
    if 0 <= idx < len(prompts):
        return prompts[idx], idx
    return "", idx


def regenerate_scene(scene_idx, new_prompt, prog=gr.Progress()):
    """Regenerate a single scene with edited prompt"""
    global scene_editor_state
    
    settings = scene_editor_state.get('settings', {})
    if not settings:
        return None, "❌ No generation settings found. Generate scenes first.", new_prompt
    
    idx = int(scene_idx)
    prompts = scene_editor_state.get('prompts', [])
    images = scene_editor_state.get('images', [])
    
    if idx < 0 or idx >= len(prompts):
        return None, f"❌ Invalid scene index: {idx}", new_prompt
    
    try:
        prog(0.2, f"Regenerating scene {idx + 1}...")
        
        # Update prompt in state
        scene_editor_state['prompts'][idx] = new_prompt
        
        # Generate new image
        new_image = generate_ai_image(
            prompt=new_prompt,
            model=settings['model'],
            resolution=settings['resolution'],
            together_api_key=settings['together_key']
        )
        
        # Update image in state
        scene_editor_state['images'][idx] = new_image
        
        prog(0.9, "Updating gallery...")
        
        # Rebuild gallery
        gallery_data = []
        for i, (img, prompt) in enumerate(zip(scene_editor_state['images'], scene_editor_state['prompts'])):
            if img and os.path.exists(img):
                caption = f"Scene {i+1}" + (" ✨" if i == idx else "")
                gallery_data.append((img, caption))
        
        prog(1.0, "Done!")
        return gallery_data, f"✅ Scene {idx + 1} regenerated!", new_prompt
        
    except Exception as e:
        return None, f"❌ Regeneration failed: {str(e)}", new_prompt


def get_editor_images():
    """Get current images from editor state for video generation"""
    global scene_editor_state
    images = scene_editor_state.get('images', [])
    return [img for img in images if img and os.path.exists(img)]

# =============================================================================
# CAPTIONS - Fonts with Categories
# =============================================================================

# Font categories with names (matching installed fonts from KAGGLE_INSTALL.py)
FONT_CATEGORIES = {
    "⬛ Bold": [
        "BebasNeue-Regular",
        "Anton-Regular", 
        "Roboto-Bold",
        "TitanOne-Regular",
        "Bungee-Regular"
    ],
    "🔷 Modern": [
        "Montserrat-Bold",
        "Montserrat-Regular",
        "Poppins-Bold",
        "Poppins-Regular",
        "Roboto-Regular"
    ],
    "🟣 Bubbly": [
        "LilitaOne-Regular",
        "FredokaOne-Regular"
    ],
    "👻 Horror": [
        "Creepster-Regular",
        "Nosifer-Regular"
    ],
    "✍️ Handwriting": [
        "Caveat-Bold",
        "Caveat-Regular",
        "Pacifico-Regular",
        "DancingScript-Bold",
        "PatrickHand-Regular"
    ],
    "🎩 Formal": [
        "PlayfairDisplay-Bold",
        "Cinzel-Bold"
    ],
    "🎨 Creative": [
        "Righteous-Regular",
        "Audiowide-Regular",
        "Orbitron-Bold"
    ],
    "😊 Informal": [
        "ComicNeue-Bold",
        "OpenSans-Bold",
        "OpenSans-Regular",
        "Arial"
    ]
}

# Flatten for dropdown
FONTS_FLAT = []
for category, fonts in FONT_CATEGORIES.items():
    for font in fonts:
        FONTS_FLAT.append(f"{category} | {font}")

# Simple list for backwards compatibility
FONTS = [f.split(" | ")[1] for f in FONTS_FLAT]

# Font filename → Font family name mapping (what's inside the TTF file)
# This is REQUIRED because ASS subtitles use the font family name, not the filename
FONT_NAME_MAP = {
    # Bold Impact
    "BebasNeue-Regular": "Bebas Neue",
    "Anton-Regular": "Anton",
    "Roboto-Bold": "Roboto",
    "TitanOne-Regular": "Titan One",
    "Bungee-Regular": "Bungee",
    # Modern
    "Montserrat-Bold": "Montserrat",
    "Montserrat-Regular": "Montserrat",
    "Poppins-Bold": "Poppins",
    "Poppins-Regular": "Poppins",
    "Roboto-Regular": "Roboto",
    # Bubbly
    "LilitaOne-Regular": "Lilita One",
    "FredokaOne-Regular": "Fredoka One",
    # Horror
    "Creepster-Regular": "Creepster",
    "Nosifer-Regular": "Nosifer",
    # Handwriting
    "Caveat-Bold": "Caveat",
    "Caveat-Regular": "Caveat",
    "Pacifico-Regular": "Pacifico",
    "DancingScript-Bold": "Dancing Script",
    "PatrickHand-Regular": "Patrick Hand",
    # Formal
    "PlayfairDisplay-Bold": "Playfair Display",
    "Cinzel-Bold": "Cinzel",
    # Creative
    "Righteous-Regular": "Righteous",
    "Audiowide-Regular": "Audiowide",
    "Orbitron-Bold": "Orbitron",
    # Informal
    "ComicNeue-Bold": "Comic Neue",
    "OpenSans-Bold": "Open Sans",
    "OpenSans-Regular": "Open Sans",
    # System fonts
    "Arial": "Arial",
}

# Fonts directory (where fonts are installed)
FONTS_DIR = f"{BASE}/fonts"

def get_font_name(dropdown_value):
    """Extract font family name from dropdown value like '⬛ Bold | BebasNeue-Regular'"""
    if " | " in dropdown_value:
        filename = dropdown_value.split(" | ")[1]
    else:
        filename = dropdown_value
    
    # Map filename to actual font family name
    return FONT_NAME_MAP.get(filename, filename)

def generate_font_preview(font_selection):
    """Generate HTML preview of selected font"""
    font_name = get_font_name(font_selection) if font_selection else "Arial"
    
    preview_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    ">
        <div style="
            font-family: '{font_name}', sans-serif;
            font-size: 32px;
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            letter-spacing: 2px;
        ">
            PREVIEW TEXT
        </div>
        <div style="
            font-family: '{font_name}', sans-serif;
            font-size: 24px;
            color: #FFFF00;
            margin-top: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        ">
            The Quick Brown Fox
        </div>
        <div style="
            font-size: 12px;
            color: #888;
            margin-top: 15px;
        ">
            Font: {font_name}
        </div>
    </div>
    """
    return preview_html

ANIMATIONS = {
    "None": {"in": ""},
    "Fade": {"in": "\\fad(150,100)"},
    "Pop": {"in": "\\t(0,80,\\fscx110\\fscy110)\\t(80,160,\\fscx100\\fscy100)"},
}

def transcribe_audio(audio_path):
    model = load_whisper()
    if not model:
        return [], []
    
    print("   Transcribing...")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")
    
    words = []
    segments = []
    
    for seg in result.get("segments", []):
        segments.append({
            "text": seg.get("text", "").strip(),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0)
        })
        for wd in seg.get("words", []):
            words.append({
                "word": str(wd.get("word", "")).strip(),
                "start": float(wd.get("start", 0)),
                "end": float(wd.get("end", 0))
            })
    
    print(f"   ✅ {len(words)} words")
    return words, segments

def to_ass_time(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"

def hex_to_ass(c):
    c = c.lstrip("#")
    return f"&H00{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"

def hex_bgr(c):
    c = c.lstrip("#")
    return f"{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"

def esc(s):
    return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def generate_captions(audio_path, caption_mode="single", words_per_group=3,
    font="Arial", fontsize=48, bold=True, uppercase=True,
    text_color="#FFFFFF", highlight_color="#FFFF00",
    outline_color="#000000", outline_size=3,
    shadow_on=True, shadow_color="#000000", shadow_depth=2,
    bg_on=False, bg_color="#000000", bg_opacity=80,
    position="bottom", margin_v=50, animation="Pop"):
    
    # Extract font name if categorized format
    font = get_font_name(font)
    
    words, segments = transcribe_audio(audio_path)
    if not words:
        return None
    
    align = {"top": 8, "middle": 5, "bottom": 2}.get(position, 2)
    shadow = shadow_depth if shadow_on else 0
    shadow_ass = hex_to_ass(shadow_color).replace("&H00", "&H80")
    border_style = 3 if bg_on else 1
    bg_ass = hex_to_ass(bg_color).replace("&H00", f"&H{255 - int(bg_opacity * 2.55):02X}")
    bold_val = -1 if bold else 0
    
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},{hex_to_ass(text_color)},{hex_to_ass(text_color)},{hex_to_ass(outline_color)},{shadow_ass if shadow_on else bg_ass},{bold_val},0,0,0,100,100,0,0,{border_style},{outline_size},{shadow},{align},50,50,{margin_v},1
Style: Highlight,{font},{fontsize},{hex_to_ass(highlight_color)},{hex_to_ass(highlight_color)},{hex_to_ass(outline_color)},{shadow_ass if shadow_on else bg_ass},{bold_val},0,0,0,100,100,0,0,{border_style},{outline_size + 1},{shadow},{align},50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    lines = [header]
    anim = ANIMATIONS.get(animation, ANIMATIONS["None"])
    anim_tag = "{" + anim.get("in", "") + "}" if anim.get("in") else ""
    
    hl_bgr = hex_bgr(highlight_color)
    txt_bgr = hex_bgr(text_color)
    
    if caption_mode == "single":
        for w in words:
            txt = w["word"].upper() if uppercase else w["word"]
            start, end = w["start"], w["end"]
            if end <= start:
                end = start + 0.25
            lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Highlight,,0,0,0,,{anim_tag}{esc(txt)}\n")
    
    elif caption_mode == "line":
        for seg in segments:
            txt = seg["text"].upper() if uppercase else seg["text"]
            lines.append(f"Dialogue: 0,{to_ass_time(seg['start'])},{to_ass_time(seg['end'])},Default,,0,0,0,,{anim_tag}{esc(txt)}\n")
    
    else:  # word mode
        n = len(words)
        i = 0
        while i < n:
            j = min(i + words_per_group, n)
            block = words[i:j]
            for k, w in enumerate(block):
                start, end = w["start"], w["end"]
                if end <= start:
                    end = start + 0.2
                parts = [anim_tag] if k == 0 and anim_tag else []
                for m, word in enumerate(block):
                    txt = word["word"].upper() if uppercase else word["word"]
                    if m == k:
                        parts.append("{\\rHighlight\\1c&H" + hl_bgr + "&}" + esc(txt))
                    else:
                        parts.append("{\\rDefault\\1c&H" + txt_bgr + "&}" + esc(txt))
                    if m < len(block) - 1:
                        parts.append(" ")
                lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Default,,0,0,0,,{''.join(parts)}\n")
            i = j
    
    ass_path = f"{DIRS['temp']}/cap_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.ass"
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"   ✅ Captions saved")
    return ass_path

# =============================================================================
# IMAGE TO VIDEO - ULTRA SMOOTH (No jitter, high quality)
# =============================================================================

FPS = 30  # Higher FPS for smoother motion

def get_smooth_zoom_filter(effect, duration, index=0):
    """
    Ultra-smooth zoom/pan with:
    - Smooth easing (sine curve)
    - High FPS output
    - Reduced zoom intensity for less jitter
    """
    frames = int(duration * FPS)
    # Ensure minimum frames to prevent jitter
    frames = max(frames, FPS)  # At least 1 second worth of frames
    
    if effect == 'Zoom Combo':
        effect = 'Zoom In' if index % 2 == 0 else 'Zoom Out'
    
    # Use smooth sine easing: (1 - cos(pi * on/frames)) / 2 gives 0 to 1 smoothly
    # REDUCED zoom intensity from 12% to 8% for smoother, less jittery motion
    
    if effect == 'Zoom In':
        # Smooth zoom from 1.0 to 1.08 with sine easing (reduced from 1.12)
        zoom_expr = f"1+0.08*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect == 'Zoom Out':
        # Smooth zoom from 1.08 to 1.0 with sine easing
        zoom_expr = f"1.08-0.08*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect == 'Pan Left':
        # Smooth pan with constant zoom (reduced from 1.08 to 1.05)
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*(1-(1-cos(PI*on/{frames}))/2)"
        y_expr = "(ih-ih/zoom)/2"
    elif effect == 'Pan Right':
        # Smooth pan right
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*(1-cos(PI*on/{frames}))/2"
        y_expr = "(ih-ih/zoom)/2"
    else:
        # Subtle slow zoom (for 'None') - very gentle
        zoom_expr = f"1+0.03*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    
    return f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=1920x1080:fps={FPS}"

def create_image_clip(img_path, duration, output_path, effect, index):
    """
    OPTIMIZED: Smooth + Fast
    - Lanczos scaling (smoother than bilinear)
    - Medium preset (balanced)
    - 30fps smooth motion
    - Smoothed fade transitions
    """
    
    # HIGH QUALITY scaling with lanczos (smoother, less jitter)
    vf_parts = [
        "scale=2560:1440:flags=lanczos:force_original_aspect_ratio=increase",
        "crop=2560:1440",
        "setsar=1"
    ]
    
    # Smooth zoom/pan
    zoom_filter = get_smooth_zoom_filter(effect, duration, index)
    vf_parts.append(zoom_filter)
    
    # Longer fade for smoother transitions (0.35s)
    fade_dur = 0.35
    fade_out_start = max(0, duration - fade_dur)
    vf_parts.append(f"fade=t=in:st=0:d={fade_dur}")
    vf_parts.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_dur}")
    
    vf = ",".join(vf_parts)
    
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-loop', '1',
        '-i', img_path,
        '-vf', vf,
        '-t', f"{duration:.3f}",
        '-r', str(FPS),
        '-pix_fmt', 'yuv420p'
    ]
    
    # OPTIMIZED encoding (fast + quality)
    if ENCODER == 'h264_nvenc':
        cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-b:v', '6M']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '22']
    
    cmd += ['-an', output_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return output_path if os.path.exists(output_path) else None
    except subprocess.SubprocessError as e:
        print(f"   ⚠️ Clip error: {e}")
        return None

def validate_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def concatenate_videos_smooth(clips, output_path, crossfade_duration=0.4):
    """Concatenate videos with SMOOTH crossfade transitions
    
    Uses 'fade' transition instead of 'dissolve' for smoother blending.
    Crossfade duration increased to 0.4s for less abrupt transitions.
    """
    
    if len(clips) == 1:
        # Just copy if single clip
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-i', clips[0], '-c', 'copy', output_path], check=True)
        return output_path
    
    # Build xfade filter chain for smooth transitions
    
    n = len(clips)
    
    # Get durations
    durations = []
    for clip in clips:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', clip
        ], capture_output=True, text=True)
        durations.append(float(result.stdout.strip()))
    
    # Build filter complex with 'fade' transition (smoother than dissolve)
    filter_parts = []
    current = "[0:v]"
    offset = durations[0] - crossfade_duration
    
    for i in range(1, n):
        next_in = f"[{i}:v]"
        out = f"[v{i}]" if i < n - 1 else "[vout]"
        # Use 'fade' transition for cleaner blending (less jitter)
        filter_parts.append(f"{current}{next_in}xfade=transition=fade:duration={crossfade_duration}:offset={offset:.3f}{out}")
        current = out
        if i < n - 1:
            offset += durations[i] - crossfade_duration
    
    filter_complex = ";".join(filter_parts)
    
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    for c in clips:
        cmd += ['-i', c]
    
    cmd += ['-filter_complex', filter_complex, '-map', '[vout]']
    
    if ENCODER == 'h264_nvenc':
        cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '6M']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    
    cmd += ['-pix_fmt', 'yuv420p', '-an', output_path]
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return output_path
    except subprocess.SubprocessError:
        # Fallback to simple concat if xfade fails
        print("   ⚠️ Crossfade failed, using simple concat")
        concat_file = output_path + ".txt"
        with open(concat_file, 'w') as f:
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n")
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0', '-i', concat_file,
                       '-c', 'copy', output_path], check=True)
        return output_path

# =============================================================================
# MAIN GENERATION
# =============================================================================

def gen(scr, voi, nam, img, 
        sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
        text_color, highlight_color, outline_color, outline_size,
        shadow_on, shadow_color, shadow_depth,
        bg_on, bg_color, bg_opacity,
        sub_pos, margin_v, animation,
        eff, mus, vol,
        # AI Image Generation parameters
        image_source="Custom Images",
        groq_api_key="", together_api_key="",
        image_model="Flux Schnell (Free)", image_resolution="512x768 (Landscape)",
        scene_count_mode="30 images", custom_scene_count=30,
        image_style="No Style",  # Image style preset
        use_editor_images=False,  # Use images from Scene Editor
        prog=gr.Progress()):
    
    if not PIPE:
        return "", "❌ Kokoro not loaded"
    if not scr or len(scr) < 10:
        return "", "❌ Script too short"
    
    # Check images based on source
    use_ai_images = image_source == "AI Generated"
    
    # Option to use pre-edited images from Scene Editor
    if use_editor_images:
        editor_imgs = get_editor_images()
        if editor_imgs:
            img = editor_imgs
            use_ai_images = False
        else:
            return "", "❌ No images in Scene Editor. Generate scenes first."
    
    if not use_ai_images and not img:
        return "", "❌ No images uploaded"
    if use_ai_images and (not groq_api_key or not together_api_key):
        return "", "❌ Both Groq and Together AI API keys are required for AI images"
    
    try:
        prog(0, "Starting...")
        jid = hashlib.md5(f"{nam}{time.time()}".encode()).hexdigest()[:8]
        
        # Audio
        prog(0.1, "Generating audio...")
        print("\n🎙️ Audio...")
        vk = VOICES.get(voi, 'af_sky')
        pts = [a for _, _, a in PIPE(scr, voice=vk, speed=1.0, split_pattern=r'\n+') if a is not None and len(a) > 0]
        if not pts:
            return "", "❌ Audio failed"
        aud = np.concatenate(pts)
        ap = f"{DIRS['temp']}/aud_{jid}.wav"
        sf.write(ap, aud, 24000)
        total_duration = len(aud) / 24000
        print(f"   ✅ {total_duration:.1f}s")
        
        # Captions
        sp = None
        if sub_on and WHIS:
            prog(0.2, "Generating captions...")
            print("\n📝 Captions...")
            sp = generate_captions(
                ap, caption_mode=caption_mode, words_per_group=int(words_per_group),
                font=font, fontsize=int(fontsize), bold=bold, uppercase=uppercase,
                text_color=text_color, highlight_color=highlight_color,
                outline_color=outline_color, outline_size=int(outline_size),
                shadow_on=shadow_on, shadow_color=shadow_color, shadow_depth=int(shadow_depth),
                bg_on=bg_on, bg_color=bg_color, bg_opacity=int(bg_opacity),
                position=sub_pos, margin_v=int(margin_v), animation=animation
            )
        
        # Get images - either custom or AI generated
        valid_images = []
        
        if use_ai_images:
            # Generate AI images
            prog(0.25, "Generating AI images...")
            try:
                num_scenes = int(custom_scene_count) if scene_count_mode == 'Custom' else 30
                _, ai_images = generate_all_ai_images(
                    script=scr,
                    num_scenes=num_scenes,
                    scene_mode=scene_count_mode,
                    groq_api_key=groq_api_key,
                    together_api_key=together_api_key,
                    model=image_model,
                    resolution=image_resolution,
                    image_style=image_style,
                    prog=prog
                )
                # Filter out None values from failed generations
                valid_images = [img for img in ai_images if img]
            except Exception as e:
                return "", f"❌ AI Image Generation Error: {str(e)}"
        else:
            # Validate custom images
            for im in img:
                p = im.name if hasattr(im, 'name') else im
                if validate_image(p):
                    valid_images.append(p)
        
        if not valid_images:
            return "", "❌ No valid images"
        
        n = len(valid_images)
        
        # FIXED: Calculate EXACT per-image duration
        # Account for crossfade overlap (0.3s between each pair)
        crossfade = 0.3
        total_crossfade_time = crossfade * (n - 1) if n > 1 else 0
        per_image_duration = (total_duration + total_crossfade_time) / n
        
        # Ensure minimum 1.5s per image for smooth animation
        per_image_duration = max(1.5, per_image_duration)
        
        print(f"\n🖼️ Processing {n} images...")
        print(f"   Duration: {per_image_duration:.2f}s each")
        
        # Create clips
        prog(0.4, "Creating clips...")
        clips = []
        for i, img_path in enumerate(valid_images):
            output = f"{DIRS['temp']}/clip_{jid}_{i}.mp4"
            result = create_image_clip(img_path, per_image_duration, output, eff, i)
            if result:
                clips.append(result)
                print(f"   ✅ {i+1}/{n}")
            prog(0.4 + 0.25 * (i + 1) / n, f"Clip {i+1}/{n}...")
        
        if not clips:
            return "", "❌ Failed to create clips"
        
        # Concatenate with smooth crossfade
        prog(0.7, "Smooth transitions...")
        print("\n🎬 Concatenating with crossfade...")
        concat_output = f"{DIRS['temp']}/concat_{jid}.mp4"
        concatenate_videos_smooth(clips, concat_output, crossfade)
        print("   ✅ Done")
        
        # Final assembly
        prog(0.85, "Final assembly...")
        print("\n🎵 Final...")
        final_output = f"{DIRS['output']}/{nam}_{jid}.mp4"
        
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
               '-i', concat_output, '-i', ap]
        
        # Music
        if mus:
            mp = mus.name if hasattr(mus, 'name') else mus
            if os.path.exists(mp):
                cmd += ['-i', mp, '-filter_complex',
                       f"[1:a]volume=1.0[v];[2:a]volume={vol/100}[m];[v][m]amix=inputs=2:duration=first[a]",
                       '-map', '0:v', '-map', '[a]']
            else:
                cmd += ['-map', '0:v', '-map', '1:a']
        else:
            cmd += ['-map', '0:v', '-map', '1:a']
        
        # Subtitles with custom fonts directory
        if sp and os.path.exists(sp):
            # Escape colons in paths for FFmpeg filter syntax (Windows/Linux compatible)
            sp_escaped = sp.replace(":", "\\:")
            fonts_dir_escaped = FONTS_DIR.replace(":", "\\:")
            cmd += ['-vf', f"ass={sp_escaped}:fontsdir={fonts_dir_escaped}"]
        
        # Trim to exact audio duration
        cmd += ['-t', f"{total_duration:.3f}"]
        
        # Encoding
        if ENCODER == 'h264_nvenc':
            cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '8M']
        else:
            cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
        
        cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', '48000', 
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', final_output]
        
        subprocess.run(cmd, check=True, timeout=300)
        
        prog(1.0, "Done!")
        print(f"\n✅ {final_output}")
        return final_output, f"✅ Done!\n📁 {final_output}\n⏱️ {total_duration:.1f}s"
        
    except Exception as e:
        import traceback
        return "", f"❌ {e}\n{traceback.format_exc()}"

# =============================================================================
# UI WITH BULK GENERATION
# =============================================================================

print("\n" + "="*60)
print("✨ READY - v4 (BULK GENERATION)")
print(f"🎙️ Kokoro: {'✅' if KOK else '❌'} | 📝 Whisper: {'✅' if WHIS else '❌'}")
print("="*60)

EFFECTS = ['Zoom Combo', 'Zoom In', 'Zoom Out', 'Pan Left', 'Pan Right', 'None']

# Global queue for bulk generation
bulk_queue = []
bulk_results = []

def add_to_queue(name, script, voice, images, image_source, groq_key, together_key,
                 img_model, img_res, scene_mode, custom_count, style_ref, style_desc):
    """Add video to bulk queue with AI image support"""
    if not script or len(script) < 10:
        return "❌ Script too short", update_queue_display()
    
    use_ai = image_source == "AI Generated"
    if not use_ai and not images:
        return "❌ No images uploaded", update_queue_display()
    if use_ai and (not groq_key or not together_key):
        return "❌ API keys required for AI images", update_queue_display()
    
    bulk_queue.append({
        'name': name or f"Video_{len(bulk_queue)+1}",
        'script': script,
        'voice': voice,
        'images': images,
        'status': '⏳ Queued',
        # AI image settings
        'image_source': image_source,
        'groq_key': groq_key,
        'together_key': together_key,
        'img_model': img_model,
        'img_res': img_res,
        'scene_mode': scene_mode,
        'custom_count': custom_count,
        'style_ref': style_ref,
        'style_desc': style_desc
    })
    img_info = "🎨 AI" if use_ai else f"📁 {len(images) if images else 0} images"
    return f"✅ Added: {name} ({img_info})", update_queue_display()

def update_queue_display():
    """Update queue display HTML with delete buttons and progress"""
    if not bulk_queue:
        return "<div style='padding:20px;text-align:center;color:#888'>Queue empty</div>"
    
    html = "<div style='max-height:400px;overflow-y:auto'>"
    for i, item in enumerate(bulk_queue):
        # Status colors and icons
        status = item.get('status', '⏳ Queued')
        progress = item.get('progress', 0)
        
        if "✅" in status:
            status_color = "#4CAF50"  # Green = Done
            bg_color = "#e8f5e9"
            progress_html = ""
        elif "🔄" in status:
            status_color = "#2196F3"  # Blue = Processing
            bg_color = "#e3f2fd"
            progress_html = f"""
            <div style='background:#ddd;border-radius:4px;height:6px;margin-top:4px'>
                <div style='background:{status_color};width:{progress}%;height:6px;border-radius:4px;transition:width 0.3s'></div>
            </div>"""
        elif "❌" in status:
            status_color = "#f44336"  # Red = Error
            bg_color = "#ffebee"
            progress_html = ""
        else:
            status_color = "#9e9e9e"  # Gray = Queued
            bg_color = "#f5f5f5"
            progress_html = ""
        
        img_badge = "🎨 AI" if item.get('image_source') == "AI Generated" else "📁 Custom"
        mode_badge = "📌 Title" if item.get('script_mode') == "Generate from Title" else "📝 Script"
        
        # Only show delete for queued items
        delete_btn = ""
        if "⏳" in status:
            delete_btn = f"<span data-delete='{i}' style='cursor:pointer;color:#f44336;font-size:14px;float:right' title='Delete'>🗑️</span>"
        
        # Build item HTML
        html += f"""
        <div style='padding:12px;margin:6px 0;background:{bg_color};border-radius:10px;border-left:4px solid {status_color}'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='font-weight:bold;font-size:14px'>{i+1}. {item['name']}</div>
                <div style='font-size:10px'>
                    <span style='background:#fff;padding:2px 6px;border-radius:4px;margin-right:4px'>{img_badge}</span>
                    <span style='background:#fff;padding:2px 6px;border-radius:4px'>{mode_badge}</span>
                    {delete_btn}
                </div>
            </div>
            <div style='font-size:11px;color:#666;margin-top:4px'>{item.get('script', '')[:50]}...</div>
            <div style='font-size:11px;color:{status_color};margin-top:4px;font-weight:500'>{status}</div>
            {progress_html}
        </div>"""
    
    html += "</div>"
    
    # Summary bar
    total = len(bulk_queue)
    done = len([i for i in bulk_queue if "✅" in i.get('status', '')])
    failed = len([i for i in bulk_queue if "❌" in i.get('status', '')])
    html += f"""
    <div style='padding:8px;background:#fafafa;border-radius:6px;margin-top:8px;font-size:12px;text-align:center'>
        📊 Total: {total} | ✅ Done: {done} | ❌ Failed: {failed}
    </div>"""
    
    return html

def delete_from_queue(index):
    """Delete item from queue by index (1-indexed for user)"""
    try:
        idx = int(index) - 1  # Convert from 1-indexed to 0-indexed
        if 0 <= idx < len(bulk_queue):
            removed = bulk_queue.pop(idx)
            return f"🗑️ Removed #{int(index)}: {removed['name']}", update_queue_display()
        return f"❌ Invalid index: {int(index)} (queue has {len(bulk_queue)} items)", update_queue_display()
    except:
        return "❌ Delete failed", update_queue_display()

def clear_queue():
    """Clear the queue"""
    bulk_queue.clear()
    bulk_results.clear()
    return "✅ Queue cleared", update_queue_display(), None, "", []

def process_bulk_queue(sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
                       text_color, highlight_color, outline_color, outline_size,
                       shadow_on, shadow_color, shadow_depth, bg_on, bg_color, bg_opacity,
                       sub_pos, margin_v, animation, eff, mus, vol, prog=gr.Progress()):
    """Process all videos in queue with auto-download"""

    
    if not bulk_queue:
        return None, "❌ Queue is empty", update_queue_display()
    
    total = len(bulk_queue)
    completed_files = []
    
    for idx, item in enumerate(bulk_queue):
        try:
            # Update status
            item['status'] = f"🔄 Processing ({idx+1}/{total})"
            prog((idx) / total, f"Processing {item['name']}...")
            
            # Generate video with AI image params from queue item
            result_path, status = gen(
                item['script'], item['voice'], item['name'], item['images'],
                sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
                text_color, highlight_color, outline_color, outline_size,
                shadow_on, shadow_color, shadow_depth, bg_on, bg_color, bg_opacity,
                sub_pos, margin_v, animation, eff, mus, vol,
                # AI Image parameters from queue item
                image_source=item.get('image_source', 'Custom Images'),
                groq_api_key=item.get('groq_key', ''),
                together_api_key=item.get('together_key', ''),
                image_model=item.get('img_model', 'Flux Schnell (Free)'),
                image_resolution=item.get('img_res', '512x768 (Landscape)'),
                scene_count_mode=item.get('scene_mode', '30 images'),
                custom_scene_count=item.get('custom_count', 30),
                style_ref_image=item.get('style_ref'),
                style_description=item.get('style_desc', ''),
                prog=gr.Progress()
            )
            
            if result_path:
                item['status'] = f"✅ Done: {os.path.basename(result_path)}"
                item['path'] = result_path
                completed_files.append(result_path)
                bulk_results.append(result_path)
            else:
                item['status'] = f"❌ Failed"
                
        except Exception as e:
            item['status'] = f"❌ Error: {str(e)[:30]}"
    
    prog(1.0, "All done!")
    
    # Return the last completed video for preview
    last_video = completed_files[-1] if completed_files else None
    
    summary = f"✅ Completed {len(completed_files)}/{total} videos\n\n"
    summary += "📁 Output files:\n"
    for f in completed_files:
        summary += f"• {os.path.basename(f)}\n"
    
    return last_video, summary, update_queue_display()

def download_all():
    """Create ZIP of all completed videos for auto-download"""
    if not bulk_results:
        return None, "❌ No completed videos"
    
    import zipfile
    from datetime import datetime
    
    # Create ZIP file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(DIRS['output'], f"bulk_videos_{timestamp}.zip")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for video_path in bulk_results:
                if os.path.exists(video_path):
                    zf.write(video_path, os.path.basename(video_path))
        
        return zip_path, f"✅ Downloaded {len(bulk_results)} videos as ZIP"
    except Exception as e:
        # Fallback to individual files
        return bulk_results, f"⚠️ ZIP failed, showing individual files: {str(e)[:30]}"

def get_video_gallery():
    """Get gallery data for all completed videos"""
    gallery_data = []
    for video_path in bulk_results:
        if os.path.exists(video_path):
            # Use video file path as gallery item
            gallery_data.append((video_path, os.path.basename(video_path)))
    return gallery_data

with gr.Blocks(title="Video Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 AI Video Generator v4\n**Single + Bulk Generation with Auto-Download**")
    
    with gr.Tabs():
        # =====================================================================
        # TAB 1: SINGLE GENERATION
        # =====================================================================
        with gr.TabItem("🎬 Single Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Script Input Section - Title OR Script
                    with gr.Accordion("📝 Script", open=True):
                        script_mode = gr.Radio(
                            ["Write Script", "Generate from Title"],
                            value="Write Script",
                            label="Input Mode"
                        )
                        
                        # Manual script input
                        with gr.Group(visible=True) as manual_script_group:
                            scr = gr.Textbox(label="Script", lines=5, placeholder="Enter your video script here...")
                        
                        # Title-to-Video (one-click flow)
                        with gr.Group(visible=False) as ai_script_group:
                            script_title = gr.Textbox(
                                label="📌 Video Title", 
                                placeholder="e.g., '10 Shocking Facts About the Ocean'"
                            )
                            script_duration = gr.Dropdown(
                                SCRIPT_DURATIONS,
                                value="2 minutes",
                                label="⏱️ Video Duration"
                            )
                            script_style = gr.Textbox(
                                label="✨ Style Prompt (describe your script style)",
                                placeholder="e.g., 'dramatic with suspense', 'fun and energetic', 'educational and informative'",
                                lines=2
                            )
                            with gr.Row():
                                script_groq_key = gr.Textbox(
                                    label="🔑 Groq API Key",
                                    type="password",
                                    placeholder="gsk_..."
                                )
                            gr.Markdown("*🔍 Research: Auto-enabled for news, facts, stats, tech, current events*")
                        
                        # Toggle visibility based on mode
                        def toggle_script_mode(mode):
                            return (
                                gr.update(visible=(mode == "Write Script")),
                                gr.update(visible=(mode == "Generate from Title"))
                            )
                        
                        script_mode.change(
                            toggle_script_mode,
                            [script_mode],
                            [manual_script_group, ai_script_group]
                        )
                    
                    voi = gr.Dropdown(list(VOICES.keys()), value='Adam (Male US)', label="🎙️ Voice")
                    nam = gr.Textbox(label="🎬 Video Name", value="Video")
                    
                    # Image Source Selection
                    with gr.Accordion("🖼️ Images", open=True):
                        image_source = gr.Radio(
                            ["Custom Images", "AI Generated"],
                            value="Custom Images",
                            label="Image Source"
                        )
                        
                        # Custom Images Section
                        with gr.Group(visible=True) as custom_images_group:
                            img = gr.File(label="� Upload Images", file_count="multiple", file_types=["image"], type="filepath")
                        
                        # AI Images Section
                        with gr.Group(visible=False) as ai_images_group:
                            gr.Markdown("#### 🔑 API Keys (BYOK)")
                            with gr.Row():
                                groq_api_key = gr.Textbox(label="Groq API Key", type="password", placeholder="gsk_...")
                                together_api_key = gr.Textbox(label="Together AI Key", type="password", placeholder="...")
                            
                            gr.Markdown("#### 🎨 Image Settings")
                            with gr.Row():
                                image_model = gr.Dropdown(
                                    list(IMAGE_MODELS.keys()),
                                    value="Flux Schnell (Free)",
                                    label="Image Model"
                                )
                                image_resolution = gr.Dropdown(
                                    list(IMAGE_RESOLUTIONS.keys()),
                                    value="512x768 (Landscape)",
                                    label="Resolution"
                                )
                            
                            gr.Markdown("#### 📊 Scene Count")
                            with gr.Row():
                                scene_count_mode = gr.Dropdown(
                                    SCENE_COUNT_MODES,
                                    value="30 images",
                                    label="Number of Images"
                                )
                                custom_scene_count = gr.Number(
                                    value=30,
                                    label="Custom Count",
                                    visible=False,
                                    minimum=1,
                                    maximum=200
                                )
                            
                            gr.Markdown("#### 🎨 Image Style Preset")
                            image_style = gr.Dropdown(
                                IMAGE_STYLE_NAMES,
                                value="No Style",
                                label="Select Style (applies to all scenes)"
                            )
                            
                            # Scene Preview & Edit (integrated from Scene Editor tab)
                            with gr.Accordion("🖼️ Preview & Edit Scenes (Optional)", open=False):
                                gr.Markdown("*Generate scenes first, edit if needed, then continue to video*")
                                with gr.Row():
                                    preview_scenes_btn = gr.Button("🎨 Generate Scenes Preview", variant="secondary")
                                    clear_preview_btn = gr.Button("🗑️ Clear", variant="stop")
                                
                                scene_gallery = gr.Gallery(
                                    label="Generated Scenes", 
                                    columns=4, 
                                    height=250,
                                    object_fit="cover"
                                )
                                
                                with gr.Row():
                                    selected_scene = gr.Number(label="Scene # to Edit", value=1, minimum=1)
                                    new_prompt = gr.Textbox(label="New Prompt for Scene", lines=2, placeholder="Describe the new scene...")
                                    regen_btn = gr.Button("🔄 Regenerate Scene", variant="secondary")
                                
                                scene_preview_status = gr.Textbox(label="Preview Status", lines=2)
                        
                        # Toggle visibility based on image source
                        def toggle_image_source(source):
                            return (
                                gr.update(visible=(source == "Custom Images")),
                                gr.update(visible=(source == "AI Generated"))
                            )
                        
                        image_source.change(
                            toggle_image_source,
                            [image_source],
                            [custom_images_group, ai_images_group]
                        )
                        
                        # Toggle custom scene count visibility
                        def toggle_custom_count(mode):
                            return gr.update(visible=(mode == "Custom"))
                        
                        scene_count_mode.change(
                            toggle_custom_count,
                            [scene_count_mode],
                            [custom_scene_count]
                        )
                    
                    with gr.Accordion("📝 Captions", open=False):
                        sub_on = gr.Checkbox(label="Enable", value=True)
                        with gr.Row():
                            caption_mode = gr.Radio(["single", "word", "line"], value="single", label="Mode")
                            words_per_group = gr.Slider(2, 5, value=3, step=1, label="Words/Group")
                        
                        gr.Markdown("#### 🔤 Font Selection")
                        font = gr.Dropdown(FONTS_FLAT, value="⬛ Bold | BebasNeue-Regular", label="Font (Category | Name)")
                        font_preview = gr.HTML(value=generate_font_preview("⬛ Bold | BebasNeue-Regular"))
                        
                        with gr.Row():
                            fontsize = gr.Slider(20, 80, value=48, step=2, label="Size")
                            bold = gr.Checkbox(label="Bold", value=True)
                            uppercase = gr.Checkbox(label="UPPERCASE", value=True)
                        
                        with gr.Row():
                            text_color = gr.ColorPicker(value="#FFFFFF", label="Text")
                            highlight_color = gr.ColorPicker(value="#FFFF00", label="Highlight")
                        with gr.Row():
                            outline_color = gr.ColorPicker(value="#000000", label="Outline")
                            outline_size = gr.Slider(0, 8, value=3, step=1, label="Size")
                        with gr.Row():
                            shadow_on = gr.Checkbox(label="Shadow", value=True)
                            shadow_color = gr.ColorPicker(value="#000000", label="Color")
                            shadow_depth = gr.Slider(0, 6, value=2, step=1, label="Depth")
                        with gr.Row():
                            bg_on = gr.Checkbox(label="BG Box", value=False)
                            bg_color = gr.ColorPicker(value="#000000", label="Color")
                            bg_opacity = gr.Slider(0, 100, value=70, step=10, label="Opacity")
                        with gr.Row():
                            sub_pos = gr.Dropdown(["bottom", "middle", "top"], value="bottom", label="Position")
                            margin_v = gr.Slider(20, 150, value=50, step=10, label="Margin")
                        animation = gr.Dropdown(list(ANIMATIONS.keys()), value="Pop", label="Animation")
                        
                        # Font preview update
                        font.change(generate_font_preview, [font], [font_preview])
                    
                    with gr.Accordion("🎨 Transitions", open=False):
                        eff = gr.Dropdown(EFFECTS, value='Zoom Combo', label="Effect")
                    
                    with gr.Accordion("🎵 Music", open=False):
                        mus = gr.File(label="Music", file_types=['audio'])
                        vol = gr.Slider(0, 100, value=20, label="Volume %")
                    
                    btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 System")
                    gr.HTML(f"""<div style='background:#f5f5f5;padding:12px;border-radius:8px'>
                        <div>🎮 {GPU_NAME}</div>
                        <div>🎙️ Kokoro: {'✅' if KOK else '❌'}</div>
                        <div>📝 Whisper: {'✅' if WHIS else '❌'}</div>
                        <div>🎨 AI Images: {'✅' if (GROQ_AVAILABLE and TOGETHER_AVAILABLE) else '⚠️ Install APIs'}</div>
                    </div>""")
            
            with gr.Row():
                vid = gr.Video(label="Video")
                sta = gr.Textbox(label="Status", lines=5)
            
            # One-Click Title → Video: Auto-generate script when in title mode
            def gen_title_to_video(mode, manual_scr, title, duration, style, groq_key, *args):
                if mode == "Generate from Title":
                    # Validate
                    if not title:
                        return "", "❌ Please enter a video title"
                    if not groq_key:
                        return "", "❌ Please enter your Groq API key for script generation"
                    
                    # Auto-generate script with smart research
                    try:
                        print("\n🎬 TITLE → VIDEO: Auto-generating script...")
                        final_script = generate_script_from_title(title, duration, style, groq_key, "auto")
                        if not final_script or final_script.startswith("❌"):
                            return "", f"❌ Script generation failed: {final_script}"
                    except Exception as e:
                        return "", f"❌ Script generation error: {str(e)}"
                else:
                    final_script = manual_scr
                
                # Now generate video with the script
                return gen(final_script, *args)
            
            # Scene Preview handlers (integrated from Scene Editor)
            preview_images_cache = []  # Store generated images for use in video
            preview_prompts_cache = []  # Store prompts for editing
            
            def generate_scene_preview(script, groq_key, together_key, model, resolution, 
                                       scene_mode, custom_count, image_style, prog=gr.Progress()):
                """Generate preview of all scenes before video creation"""
                global preview_images_cache, preview_prompts_cache
                
                if not script:
                    return [], "❌ Please enter a script first"
                if not groq_key or not together_key:
                    return [], "❌ Please enter both API keys"
                
                try:
                    from groq import Groq
                    
                    # Get scene count
                    if scene_mode == "Custom":
                        num_images = int(custom_count)
                    elif scene_mode == "Scene-by-Scene":
                        num_images = len([s for s in script.split('.') if len(s.strip()) > 10])
                    else:
                        num_images = int(scene_mode.split()[0])
                    
                    # Get prompts from Groq
                    print(f"\n🎨 Generating {num_images} scene previews...")
                    groq_client = Groq(api_key=groq_key)
                    
                    prompt_request = f"""Generate exactly {num_images} image prompts for this video script.
Each prompt should describe a visual scene that matches the narration.

Script: {script[:3000]}

Return ONLY the prompts, one per line, no numbers or prefixes."""
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt_request}],
                        temperature=0.7
                    )
                    
                    raw_prompts = [p.strip() for p in response.choices[0].message.content.strip().split('\n') if p.strip()][:num_images]
                    preview_prompts_cache = raw_prompts
                    
                    # Apply style preset to all prompts
                    style_prompt = IMAGE_STYLES.get(image_style, "")
                    styled_prompts = []
                    for p in raw_prompts:
                        if style_prompt:
                            full_prompt = f"{STYLE_HEADER}, {style_prompt}, SUBJECT: {p}"
                        else:
                            full_prompt = f"{STYLE_HEADER}, SUBJECT: {p}"
                        styled_prompts.append(full_prompt)
                    
                    # Generate images
                    images = []
                    model_id = IMAGE_MODELS.get(model, IMAGE_MODELS["Flux Schnell (Free)"])
                    w, h = IMAGE_RESOLUTIONS.get(resolution, (768, 512))
                    
                    from openai import OpenAI
                    client = OpenAI(api_key=together_key, base_url="https://api.together.xyz/v1")
                    
                    for i, prompt in enumerate(styled_prompts):
                        prog((i+1)/len(styled_prompts), f"Generating scene {i+1}/{len(styled_prompts)}...")
                        try:
                            resp = client.images.generate(
                                model=model_id, prompt=prompt,
                                n=1, size=f"{w}x{h}"
                            )
                            import requests
                            img_data = requests.get(resp.data[0].url).content
                            img_path = os.path.join(DIRS['temp'], f"preview_{i}.png")
                            with open(img_path, 'wb') as f:
                                f.write(img_data)
                            images.append(img_path)
                        except Exception as e:
                            print(f"   ⚠️ Scene {i+1} failed: {e}")
                    
                    preview_images_cache = images
                    return [(img, f"Scene {i+1}") for i, img in enumerate(images)], f"✅ Generated {len(images)} scenes"
                
                except Exception as e:
                    return [], f"❌ Error: {str(e)}"
            
            def clear_scene_preview():
                global preview_images_cache, preview_prompts_cache
                preview_images_cache = []
                preview_prompts_cache = []
                return [], "Cleared"
            
            def regenerate_single_scene(scene_num, new_prompt, together_key, model, resolution, image_style):
                """Regenerate a single scene with new prompt"""
                global preview_images_cache
                
                try:
                    idx = int(scene_num) - 1
                    if idx < 0 or idx >= len(preview_images_cache):
                        return [(img, f"Scene {i+1}") for i, img in enumerate(preview_images_cache)], "❌ Invalid scene number"
                    
                    model_id = IMAGE_MODELS.get(model, IMAGE_MODELS["Flux Schnell (Free)"])
                    w, h = IMAGE_RESOLUTIONS.get(resolution, (768, 512))
                    
                    # Apply style preset
                    style_prompt = IMAGE_STYLES.get(image_style, "")
                    if style_prompt:
                        full_prompt = f"{STYLE_HEADER}, {style_prompt}, SUBJECT: {new_prompt}"
                    else:
                        full_prompt = f"{STYLE_HEADER}, SUBJECT: {new_prompt}"
                    
                    from openai import OpenAI
                    import requests
                    
                    client = OpenAI(api_key=together_key, base_url="https://api.together.xyz/v1")
                    resp = client.images.generate(model=model_id, prompt=full_prompt, n=1, size=f"{w}x{h}")
                    
                    img_data = requests.get(resp.data[0].url).content
                    img_path = os.path.join(DIRS['temp'], f"preview_{idx}.png")
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    preview_images_cache[idx] = img_path
                    return [(img, f"Scene {i+1}") for i, img in enumerate(preview_images_cache)], f"✅ Regenerated scene {scene_num}"
                
                except Exception as e:
                    return [(img, f"Scene {i+1}") for i, img in enumerate(preview_images_cache)], f"❌ Error: {str(e)}"
            
            # Scene preview event handlers
            preview_scenes_btn.click(
                generate_scene_preview,
                [scr, groq_api_key, together_api_key, image_model, image_resolution,
                 scene_count_mode, custom_scene_count, image_style],
                [scene_gallery, scene_preview_status]
            )
            
            clear_preview_btn.click(
                clear_scene_preview, [], [scene_gallery, scene_preview_status]
            )
            
            regen_btn.click(
                regenerate_single_scene,
                [selected_scene, new_prompt, together_api_key, image_model, image_resolution, image_style],
                [scene_gallery, scene_preview_status]
            )
            
            btn.click(gen_title_to_video, [
                script_mode, scr,  # Mode and manual script
                script_title, script_duration, script_style, script_groq_key,  # Title params (no research checkbox)
                voi, nam, img,
                sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
                text_color, highlight_color, outline_color, outline_size,
                shadow_on, shadow_color, shadow_depth, bg_on, bg_color, bg_opacity,
                sub_pos, margin_v, animation, eff, mus, vol,
                # AI Image parameters
                image_source, groq_api_key, together_api_key,
                image_model, image_resolution,
                scene_count_mode, custom_scene_count,
                image_style
            ], [vid, sta])
        
        # =====================================================================
        # TAB 2: BULK GENERATION (Same features as Single Gen)
        # =====================================================================
        with gr.TabItem("📦 Bulk Generation"):
            gr.Markdown("### Add multiple videos to queue, then process all at once")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### ➕ Add to Queue")
                    bulk_name = gr.Textbox(label="📝 Video Name", placeholder="Video 1")
                    
                    # Script Mode - Same as Single Gen
                    with gr.Accordion("📝 Script", open=True):
                        bulk_script_mode = gr.Radio(
                            ["Write Script", "Generate from Title"],
                            value="Write Script",
                            label="Input Mode"
                        )
                        
                        # Manual script
                        with gr.Group(visible=True) as bulk_manual_group:
                            bulk_script = gr.Textbox(label="Script", lines=4, placeholder="Enter script for this video...")
                        
                        # Title-to-Video
                        with gr.Group(visible=False) as bulk_title_group:
                            bulk_title = gr.Textbox(
                                label="📌 Video Title", 
                                placeholder="e.g., '10 Shocking Facts About the Ocean'"
                            )
                            bulk_duration = gr.Dropdown(
                                SCRIPT_DURATIONS,
                                value="2 minutes",
                                label="⏱️ Video Duration"
                            )
                            bulk_style_prompt = gr.Textbox(
                                label="✨ Style Prompt",
                                placeholder="e.g., 'dramatic with suspense', 'fun and energetic'",
                                lines=2
                            )
                            with gr.Row():
                                bulk_script_groq_key = gr.Textbox(
                                    label="🔑 Groq API Key",
                                    type="password",
                                    placeholder="gsk_..."
                                )
                            gr.Markdown("*🔍 Research: Auto-enabled for news, facts, stats, tech, events*")
                        
                        # Toggle
                        def toggle_bulk_script_mode(mode):
                            return (
                                gr.update(visible=(mode == "Write Script")),
                                gr.update(visible=(mode == "Generate from Title"))
                            )
                        
                        bulk_script_mode.change(
                            toggle_bulk_script_mode,
                            [bulk_script_mode],
                            [bulk_manual_group, bulk_title_group]
                        )
                    
                    bulk_voice = gr.Dropdown(list(VOICES.keys()), value='Adam (Male US)', label="🎙️ Voice")
                    
                    # Image Source for Bulk
                    with gr.Accordion("🖼️ Images", open=True):
                        bulk_image_source = gr.Radio(
                            ["Custom Images", "AI Generated"],
                            value="Custom Images",
                            label="Image Source"
                        )
                        
                        with gr.Group(visible=True) as bulk_custom_group:
                            bulk_images = gr.File(label="📁 Upload Images", file_count="multiple", file_types=["image"], type="filepath")
                        
                        with gr.Group(visible=False) as bulk_ai_group:
                            gr.Markdown("##### 🔑 API Keys")
                            with gr.Row():
                                bulk_groq_key = gr.Textbox(label="Groq Key", type="password", placeholder="gsk_...")
                                bulk_together_key = gr.Textbox(label="Together AI Key", type="password", placeholder="...")
                            
                            with gr.Row():
                                bulk_img_model = gr.Dropdown(list(IMAGE_MODELS.keys()), value="Flux Schnell (Free)", label="Model")
                                bulk_img_res = gr.Dropdown(list(IMAGE_RESOLUTIONS.keys()), value="512x768 (Landscape)", label="Resolution")
                            
                            with gr.Row():
                                bulk_scene_mode = gr.Dropdown(SCENE_COUNT_MODES, value="30 images", label="Scene Count")
                                bulk_custom_count = gr.Number(value=30, label="Custom", visible=False, minimum=1, maximum=200)
                            
                            gr.Markdown("##### 🎨 Image Style")
                            bulk_image_style = gr.Dropdown(
                                IMAGE_STYLE_NAMES,
                                value="No Style",
                                label="Style Preset"
                            )
                        
                        def toggle_bulk_source(source):
                            return (
                                gr.update(visible=(source == "Custom Images")),
                                gr.update(visible=(source == "AI Generated"))
                            )
                        
                        bulk_image_source.change(toggle_bulk_source, [bulk_image_source], [bulk_custom_group, bulk_ai_group])
                        
                        def toggle_bulk_custom_count(mode):
                            return gr.update(visible=(mode == "Custom"))
                        
                        bulk_scene_mode.change(toggle_bulk_custom_count, [bulk_scene_mode], [bulk_custom_count])
                    
                    with gr.Row():
                        add_btn = gr.Button("➕ Add to Queue", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Queue", variant="stop")
                    
                    add_status = gr.Textbox(label="Status", lines=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 Queue")
                    queue_display = gr.HTML(value="<div style='padding:20px;text-align:center;color:#888'>Queue empty</div>")
            
            gr.Markdown("---")
            gr.Markdown("#### ⚙️ Generation Settings (applies to all videos)")
            
            with gr.Row():
                # Captions - Full settings like Single Gen
                with gr.Column():
                    with gr.Accordion("📝 Captions", open=True):
                        bulk_sub_on = gr.Checkbox(label="Enable Captions", value=True)
                        bulk_caption_mode = gr.Radio(["single", "word", "line"], value="single", label="Mode")
                        bulk_words_per_group = gr.Slider(2, 5, value=3, step=1, label="Words/Group")
                        
                        gr.Markdown("##### 🔤 Font")
                        bulk_font = gr.Dropdown(FONTS_FLAT, value="⬛ Bold | BebasNeue-Regular", label="Font")
                        bulk_font_preview = gr.HTML(value=generate_font_preview("⬛ Bold | BebasNeue-Regular"))
                        bulk_font.change(generate_font_preview, [bulk_font], [bulk_font_preview])
                        
                        with gr.Row():
                            bulk_fontsize = gr.Slider(20, 80, value=48, step=2, label="Size")
                            bulk_bold = gr.Checkbox(label="Bold", value=True)
                            bulk_uppercase = gr.Checkbox(label="UPPERCASE", value=True)
                
                with gr.Column():
                    with gr.Accordion("🎨 Caption Styling", open=True):
                        with gr.Row():
                            bulk_text_color = gr.ColorPicker(value="#FFFFFF", label="Text")
                            bulk_highlight_color = gr.ColorPicker(value="#FFFF00", label="Highlight")
                        with gr.Row():
                            bulk_outline_color = gr.ColorPicker(value="#000000", label="Outline")
                            bulk_outline_size = gr.Slider(0, 8, value=3, step=1, label="Outline Size")
                        with gr.Row():
                            bulk_shadow_on = gr.Checkbox(label="Shadow", value=True)
                            bulk_shadow_color = gr.ColorPicker(value="#000000", label="Shadow Color")
                            bulk_shadow_depth = gr.Slider(0, 6, value=2, step=1, label="Depth")
                        with gr.Row():
                            bulk_bg_on = gr.Checkbox(label="BG Box", value=False)
                            bulk_bg_color = gr.ColorPicker(value="#000000", label="BG Color")
                            bulk_bg_opacity = gr.Slider(0, 100, value=70, step=10, label="Opacity")
                        with gr.Row():
                            bulk_sub_pos = gr.Dropdown(["bottom", "middle", "top"], value="bottom", label="Position")
                            bulk_margin_v = gr.Slider(20, 150, value=50, step=10, label="Margin")
                        bulk_animation = gr.Dropdown(list(ANIMATIONS.keys()), value="Pop", label="Animation")
                
                with gr.Column():
                    with gr.Accordion("🎨 Transitions & Music", open=True):
                        bulk_effect = gr.Dropdown(EFFECTS, value='Zoom Combo', label="Transition Effect")
                        bulk_mus = gr.File(label="🎵 Background Music", file_types=['audio'])
                        bulk_vol = gr.Slider(0, 100, value=20, label="Music Volume %")
            
            gr.Markdown("---")
            
            with gr.Row():
                process_btn = gr.Button("🚀 Process All Videos", variant="primary", size="lg")
                download_btn = gr.Button("⬇️ Download All (ZIP)", variant="secondary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=2):
                    bulk_video = gr.Video(label="Preview Video (Click gallery item)")
                    bulk_status = gr.Textbox(label="Bulk Status", lines=6)
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 🎬 Generated Videos")
                    bulk_gallery = gr.Gallery(
                        label="Click to preview",
                        columns=2,
                        height=200,
                        object_fit="cover",
                        allow_preview=True
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    bulk_files = gr.File(label="📁 Download ZIP", file_count="single", visible=True)
                with gr.Column(scale=1):
                    gr.Markdown("#### 🗑️ Delete from Queue")
                    delete_index = gr.Number(label="Item # to Delete", value=1, minimum=1, precision=0)
                    delete_btn = gr.Button("🗑️ Delete Item", variant="stop")
            
            # Updated add_to_queue with script mode support
            def add_to_queue_v2(name, script_mode, script, title, duration, style_prompt, script_groq_key,
                               voice, images, image_source, groq_key, together_key,
                               img_model, img_res, scene_mode, custom_count, image_style):
                """Add video to bulk queue with Title-to-Video support"""
                
                # Determine script
                if script_mode == "Generate from Title":
                    if not title:
                        return "❌ Please enter a video title", update_queue_display()
                    final_script = f"[GENERATE_FROM_TITLE]{title}"  # Special marker
                    script_info = f"Title: {title[:30]}..."
                else:
                    if not script or len(script) < 10:
                        return "❌ Script too short", update_queue_display()
                    final_script = script
                    script_info = f"{script[:40]}..."
                
                use_ai = image_source == "AI Generated"
                if not use_ai and not images:
                    return "❌ No images uploaded", update_queue_display()
                if use_ai and (not groq_key or not together_key):
                    return "❌ API keys required for AI images", update_queue_display()
                
                bulk_queue.append({
                    'name': name or f"Video_{len(bulk_queue)+1}",
                    'script': final_script,
                    'script_mode': script_mode,
                    'title': title,
                    'duration': duration,
                    'style_prompt': style_prompt,
                    'script_groq_key': script_groq_key,
                    'use_research': 'auto',  # Auto-detect based on topic
                    'voice': voice,
                    'images': images,
                    'status': '⏳ Queued',
                    'image_source': image_source,
                    'groq_key': groq_key,
                    'together_key': together_key,
                    'img_model': img_model,
                    'img_res': img_res,
                    'scene_mode': scene_mode,
                    'custom_count': custom_count,
                    'image_style': image_style
                })
                img_info = "🎨 AI" if use_ai else f"📁 {len(images) if images else 0} images"
                mode_info = "📌 Title" if script_mode == "Generate from Title" else "📝 Script"
                return f"✅ Added: {name} ({mode_info}, {img_info})", update_queue_display()
            
            # Updated process with full settings
            def process_bulk_queue_v2(sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
                                      text_color, highlight_color, outline_color, outline_size,
                                      shadow_on, shadow_color, shadow_depth, bg_on, bg_color, bg_opacity,
                                      sub_pos, margin_v, animation, eff, mus, vol, prog=gr.Progress()):
                """Process all videos in queue with full settings"""
                
                if not bulk_queue:
                    return None, "❌ Queue is empty", update_queue_display()
                
                total = len(bulk_queue)
                completed_files = []
                
                for idx, item in enumerate(bulk_queue):
                    try:
                        item['status'] = f"🔄 Processing ({idx+1}/{total})"
                        prog((idx) / total, f"Processing {item['name']}...")
                        
                        # Handle Title-to-Video script generation
                        script = item['script']
                        if item.get('script_mode') == "Generate from Title":
                            print(f"\n🎬 Generating script for: {item['title']}")
                            try:
                                script = generate_script_from_title(
                                    item['title'], item['duration'], 
                                    item['style_prompt'], item['script_groq_key'],
                                    item.get('use_research', True)
                                )
                            except Exception as e:
                                item['status'] = f"❌ Script error: {str(e)[:30]}"
                                continue
                        
                        result_path, status = gen(
                            script, item['voice'], item['name'], item['images'],
                            sub_on, caption_mode, words_per_group, font, fontsize, bold, uppercase,
                            text_color, highlight_color, outline_color, outline_size,
                            shadow_on, shadow_color, shadow_depth, bg_on, bg_color, bg_opacity,
                            sub_pos, margin_v, animation, eff, mus, vol,
                            image_source=item.get('image_source', 'Custom Images'),
                            groq_api_key=item.get('groq_key', ''),
                            together_api_key=item.get('together_key', ''),
                            image_model=item.get('img_model', 'Flux Schnell (Free)'),
                            image_resolution=item.get('img_res', '512x768 (Landscape)'),
                            scene_count_mode=item.get('scene_mode', '30 images'),
                            custom_scene_count=item.get('custom_count', 30),
                            image_style=item.get('image_style', 'No Style'),
                            prog=gr.Progress()
                        )
                        
                        if result_path:
                            item['status'] = f"✅ Done: {os.path.basename(result_path)}"
                            completed_files.append(result_path)
                            bulk_results.append(result_path)
                        else:
                            item['status'] = f"❌ Failed"
                            
                    except Exception as e:
                        item['status'] = f"❌ Error: {str(e)[:30]}"
                
                prog(1.0, "All done!")
                last_video = completed_files[-1] if completed_files else None
                
                summary = f"✅ Completed {len(completed_files)}/{total} videos\n\n"
                summary += "📁 Output files:\n"
                for f in completed_files:
                    summary += f"• {os.path.basename(f)}\n"
                
                # Return with gallery data
                return last_video, summary, update_queue_display(), get_video_gallery()
            
            # Event handlers
            add_btn.click(
                add_to_queue_v2,
                [bulk_name, bulk_script_mode, bulk_script, bulk_title, bulk_duration, 
                 bulk_style_prompt, bulk_script_groq_key,
                 bulk_voice, bulk_images, bulk_image_source, bulk_groq_key, bulk_together_key,
                 bulk_img_model, bulk_img_res, bulk_scene_mode, bulk_custom_count,
                 bulk_image_style],
                [add_status, queue_display]
            )
            clear_btn.click(clear_queue, [], [add_status, queue_display, bulk_video, bulk_status, bulk_gallery])
            
            process_btn.click(
                process_bulk_queue_v2,
                [bulk_sub_on, bulk_caption_mode, bulk_words_per_group, bulk_font, bulk_fontsize, bulk_bold, bulk_uppercase,
                 bulk_text_color, bulk_highlight_color, bulk_outline_color, bulk_outline_size,
                 bulk_shadow_on, bulk_shadow_color, bulk_shadow_depth, bulk_bg_on, bulk_bg_color, bulk_bg_opacity,
                 bulk_sub_pos, bulk_margin_v, bulk_animation, bulk_effect, bulk_mus, bulk_vol],
                [bulk_video, bulk_status, queue_display, bulk_gallery]
            )
            
            download_btn.click(download_all, [], [bulk_files, bulk_status])
            
            # Delete button handler
            delete_btn.click(
                delete_from_queue,
                [delete_index],
                [add_status, queue_display]
            )

# Launch
def port():
    import socket
    for p in range(7860, 7880):
        try:
            with socket.socket() as s:
                s.bind(('', p))
                return p
        except:
            pass
    return 7860

P = port()
print(f"\n🌐 Port {P}")
app.launch(share=True, server_port=P, allowed_paths=[DIRS['output'], DIRS['temp']])

