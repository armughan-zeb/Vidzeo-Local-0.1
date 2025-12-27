# =============================================================================
# IMAGE SERVICE - AI Image Generation with Safety Pipeline
# =============================================================================

import os
import json
import hashlib
import time
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEMP_DIR, IMAGE_MODELS, IMAGE_RESOLUTIONS, IMAGE_STYLES, IMAGE_STYLE_NAMES,
    STYLE_HEADER, BLOCK_WORDS, SAFE_CONTENT_PREFIX, NSFW_THRESHOLD, MAX_REGEN_TRIES
)

# Global NSFW classifier
_nsfw_classifier = None


def init_nsfw():
    """Initialize NSFW classifier"""
    global _nsfw_classifier
    
    if _nsfw_classifier is not None:
        return True
    
    try:
        from opennsfw2 import predict_image
        _nsfw_classifier = predict_image
        print("✅ OpenNSFW2 loaded - Image safety enabled")
        return True
    except ImportError:
        print("⚠️ OpenNSFW2 not installed - Image safety disabled")
        _nsfw_classifier = None
        return False


def is_prompt_safe(prompt: str) -> bool:
    """Check if prompt contains blocked NSFW words"""
    p = prompt.lower()
    return not any(word in p for word in BLOCK_WORDS)


def sanitize_prompt(prompt: str) -> str:
    """Remove blocked words from prompt"""
    p = prompt.lower()
    for word in BLOCK_WORDS:
        p = p.replace(word, "")
    return p.strip()


def classify_nsfw(image_path: str) -> float:
    """Get NSFW score for an image (0.0 = safe, 1.0 = explicit)"""
    global _nsfw_classifier
    
    if _nsfw_classifier is None:
        return 0.0  # Assume safe if classifier not available
    
    try:
        score = _nsfw_classifier(image_path)
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


def extract_scenes_from_script(
    script: str,
    num_scenes: int,
    groq_api_key: str,
    style_description: str = ""
) -> list:
    """
    Use Groq Llama 3.1 8B to split script into scenes with image prompts.
    
    Returns:
        list: List of image prompt strings
    """
    try:
        from groq import Groq
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
        # Fallback: split by sentences
        lines = [l.strip() for l in script.replace('!', '.').replace('?', '.').split('.') if l.strip()]
        step = max(1, len(lines) // num_scenes)
        prompts = []
        for i in range(0, len(lines), step):
            chunk = '. '.join(lines[i:i+step])
            prompts.append(f"Illustration of: {chunk[:200]}")
        return prompts[:num_scenes]
    except Exception as e:
        raise ValueError(f"Scene extraction failed: {str(e)}")


def generate_pollination_image(
    prompt: str,
    width: int = 768,
    height: int = 512,
    model: str = "flux",
    api_key: str = ""
) -> str:
    """
    Generate image using Pollinations AI (FREE - API key optional).
    
    Uses the Flux model via simple GET request.
    Endpoint: https://image.pollinations.ai/prompt/{encoded_prompt}
    
    Args:
        prompt: Image description
        width: Image width (default 768)
        height: Image height (default 512)
        model: Model to use (default 'flux')
        api_key: Optional Pollinations API key for higher limits/no watermarks
    
    Returns:
        str: Path to saved image, or None if failed
    """
    import urllib.parse
    
    # Step 1: Check and sanitize prompt
    if not is_prompt_safe(prompt):
        print(f"   ⚠️ Unsafe prompt detected - sanitizing...")
        prompt = sanitize_prompt(prompt)
        if not prompt:
            print(f"   ❌ Prompt completely unsafe - skipping")
            return None
    
    # Step 2: Build safe prompt
    safe_prompt = f"{SAFE_CONTENT_PREFIX}, {prompt}"
    
    # URL encode the prompt
    encoded_prompt = urllib.parse.quote(safe_prompt)
    
    # Build URL with correct gen.pollinations.ai endpoint
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
    params = f"?model={model}&width={width}&height={height}&nologo=true"
    
    # Add API key if provided (for higher limits and no watermarks)
    if api_key:
        params += f"&key={api_key}"
    
    full_url = url + params
    
    # Build headers with Bearer token if API key provided
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    for attempt in range(MAX_REGEN_TRIES):
        try:
            # Request image with longer timeout
            response = requests.get(full_url, headers=headers, timeout=120)
            
            if response.status_code == 200:
                # Save image
                jid = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8]
                img_path = str(TEMP_DIR / f"pollination_{jid}_{int(time.time())}.png")
                
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                # Step 3: NSFW check
                if is_image_safe(img_path):
                    print(f"   ✅ Safe image saved: {os.path.basename(img_path)}")
                    return img_path
                else:
                    os.remove(img_path)
                    print(f"   ⚠️ NSFW detected, regenerating... (attempt {attempt + 1})")
                    # Add random seed for variety
                    full_url = url + params + f"&seed={int(time.time())}"
            else:
                print(f"   ⚠️ Pollinations API error: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️ Generation attempt {attempt + 1} failed: {e}")
    
    print(f"   ❌ Could not generate safe image after {MAX_REGEN_TRIES} attempts")
    return None


def generate_ai_image(
    prompt: str,
    model: str,
    resolution: str,
    together_api_key: str
) -> str:
    """
    Generate image using Together AI with full safety pipeline.
    
    Returns:
        str: Path to safe image, or None if generation fails
    """
    # Step 1: Check and sanitize prompt
    if not is_prompt_safe(prompt):
        print(f"   ⚠️ Unsafe prompt detected - sanitizing...")
        prompt = sanitize_prompt(prompt)
        if not prompt:
            print(f"   ❌ Prompt completely unsafe - skipping")
            return None
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=together_api_key,
            base_url="https://api.together.xyz/v1"
        )
        
        res = IMAGE_RESOLUTIONS.get(resolution, {'width': 768, 'height': 512})
        model_id = IMAGE_MODELS.get(model, 'black-forest-labs/FLUX.1-schnell')
        
        print(f"   🔍 DEBUG: model param='{model}', model_id='{model_id}'")
        
        # Build safe prompt
        safe_prompt = f"{SAFE_CONTENT_PREFIX}, {prompt}"
        
        print(f"   🎨 Generating: {model} ({res['width']}x{res['height']})...")
        
        # Generate with auto-regeneration loop
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
                img_path = TEMP_DIR / f"ai_img_{img_hash}_{int(time.time())}.png"
                
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                
                # NSFW Detection
                if is_image_safe(str(img_path)):
                    print(f"   ✅ Safe image saved: {img_path.name}")
                    return str(img_path)
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
                    time.sleep(1)
        
        print(f"   ❌ Could not generate safe image after {MAX_REGEN_TRIES} attempts")
        return None
        
    except Exception as e:
        raise ValueError(f"Image generation failed: {str(e)}")


def generate_all_images(
    script: str,
    num_scenes: int,
    scene_mode: str,
    groq_api_key: str,
    together_api_key: str,
    model: str,
    resolution: str,
    image_style: str = "No Style",
    progress_callback=None
) -> tuple:
    """
    Full pipeline: Script → Scenes → Styled Images
    
    Returns:
        tuple: (list of prompts, list of image paths)
    """
    # Determine number of scenes
    if scene_mode == 'Scene-by-Scene (1 per sentence)':
        sentences = [s.strip() for s in script.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        num_scenes = len(sentences)
        num_scenes = max(3, min(num_scenes, 100))
    elif 'images' in scene_mode:
        num_scenes = int(scene_mode.split()[0])
    
    print(f"\n🎨 AI Image Generation: {num_scenes} images")
    print(f"   🎭 Style: {image_style}")
    
    # Step 1: Extract basic scene prompts
    if progress_callback:
        progress_callback(0.1, "Extracting scenes from script...")
    print("   📝 Extracting scenes...")
    raw_prompts = extract_scenes_from_script(script, num_scenes, groq_api_key, "")
    
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
        if progress_callback:
            progress_callback(0.1 + 0.8 * (i / total), f"Generating image {i+1}/{total}...")
        try:
            img_path = generate_ai_image(prompt, model, resolution, together_api_key)
            images.append(img_path)
        except Exception as e:
            print(f"   ⚠️ Failed image {i+1}: {e}")
            images.append(None)
            continue
    
    print(f"   ✅ Generated {len([i for i in images if i])}/{total} images")
    return raw_prompts, images


def get_style_names() -> list:
    """Get list of available style names"""
    return IMAGE_STYLE_NAMES


def get_models() -> list:
    """Get list of available image models"""
    return list(IMAGE_MODELS.keys())


def get_resolutions() -> list:
    """Get list of available resolutions"""
    return list(IMAGE_RESOLUTIONS.keys())
