# =============================================================================
# VIDEO SERVICE - FFmpeg Video Processing
# =============================================================================

import os
import subprocess
import hashlib
import time
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEMP_DIR, OUTPUT_DIR, FONTS_DIR, FPS, CROSSFADE_DURATION, EFFECTS, ANIMATIONS, FONT_NAME_MAP

# Whisper model reference (lazy loaded)
_whisper_model = None

# Global state
_gpu_name = "CPU"
_encoder = "libx264"
_has_gpu = False


def init_video_service():
    """Initialize video service and detect GPU"""
    global _gpu_name, _encoder, _has_gpu
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            _gpu_name = result.stdout.strip().split('\n')[0]
            
            # Check for NVENC support
            encoder_check = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True
            )
            if 'h264_nvenc' in encoder_check.stdout:
                _encoder = 'h264_nvenc'
                _has_gpu = True
                print(f"✅ GPU: {_gpu_name} (NVENC enabled)")
                return
    except:
        pass
    
    print("⚠️ CPU mode (no GPU acceleration)")


def get_gpu_info() -> dict:
    """Get GPU information"""
    return {
        "name": _gpu_name,
        "encoder": _encoder,
        "has_gpu": _has_gpu
    }


def validate_image(path: str) -> bool:
    """Validate image file"""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


# =============================================================================
# CAPTION GENERATION
# =============================================================================

def get_font_name(dropdown_value: str) -> str:
    """Extract font family name from dropdown value like '⬛ Bold | BebasNeue-Regular'"""
    if " | " in dropdown_value:
        filename = dropdown_value.split(" | ")[1]
    else:
        filename = dropdown_value
    return FONT_NAME_MAP.get(filename, filename)


def to_ass_time(t: float) -> str:
    """Convert seconds to ASS time format"""
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def hex_to_ass(c: str) -> str:
    """Convert hex color to ASS format"""
    c = c.lstrip("#")
    return f"&H00{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"


def hex_bgr(c: str) -> str:
    """Convert hex color to BGR format for ASS"""
    c = c.lstrip("#")
    return f"{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"


def esc(s: str) -> str:
    """Escape special characters for ASS"""
    return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def load_whisper_model():
    """Load Whisper model lazily"""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            device = "cuda" if _has_gpu else "cpu"
            print(f"   Loading Whisper ({device})...")
            _whisper_model = whisper.load_model("base", device=device)
            print("   ✅ Whisper loaded")
        except Exception as e:
            print(f"   ⚠️ Whisper load failed: {e}")
            return None
    return _whisper_model


def transcribe_audio(audio_path: str) -> tuple:
    """Transcribe audio using Whisper, return (words, segments)"""
    model = load_whisper_model()
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
    
    print(f"   ✅ {len(words)} words transcribed")
    return words, segments


def generate_captions(
    audio_path: str,
    caption_mode: str = "single",
    words_per_group: int = 3,
    font: str = "Arial",
    fontsize: int = 48,
    bold: bool = True,
    uppercase: bool = True,
    text_color: str = "#FFFFFF",
    highlight_color: str = "#FFFF00",
    outline_color: str = "#000000",
    outline_size: int = 3,
    shadow_on: bool = True,
    shadow_color: str = "#000000",
    shadow_depth: int = 2,
    bg_on: bool = False,
    bg_color: str = "#000000",
    bg_opacity: int = 80,
    position: str = "bottom",
    margin_v: int = 50,
    animation: str = "Pop"
) -> str:
    """
    Generate ASS subtitle file from audio using Whisper.
    
    Args:
        audio_path: Path to audio file
        caption_mode: 'single' (one word), 'word' (word groups), 'line' (full lines)
        words_per_group: Words per group in 'word' mode
        font: Font name or dropdown value
        fontsize: Font size
        bold: Bold text
        uppercase: Uppercase text
        text_color: Primary text color (hex)
        highlight_color: Highlighted word color (hex)
        outline_color: Text outline color (hex)
        outline_size: Outline thickness
        shadow_on: Enable shadow
        shadow_color: Shadow color (hex)
        shadow_depth: Shadow distance
        bg_on: Enable background box
        bg_color: Background color (hex)
        bg_opacity: Background opacity (0-100)
        position: Text position ('top', 'middle', 'bottom')
        margin_v: Vertical margin
        animation: Animation style ('None', 'Fade', 'Pop')
    
    Returns:
        Path to generated ASS file, or None if failed
    """
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
    
    else:  # word mode (groups)
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
    
    ass_path = str(TEMP_DIR / f"cap_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.ass")
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    print(f"   ✅ Captions saved: {ass_path}")
    return ass_path


def get_smooth_zoom_filter(effect: str, duration: float, index: int = 0) -> str:
    """
    Ultra-smooth zoom/pan with sine-curve easing.
    """
    frames = int(duration * FPS)
    frames = max(frames, FPS)  # At least 1 second worth of frames
    
    if effect == 'Zoom Combo':
        effect = 'Zoom In' if index % 2 == 0 else 'Zoom Out'
    
    if effect == 'Zoom In':
        zoom_expr = f"1+0.08*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect == 'Zoom Out':
        zoom_expr = f"1.08-0.08*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif effect == 'Pan Left':
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*(1-(1-cos(PI*on/{frames}))/2)"
        y_expr = "(ih-ih/zoom)/2"
    elif effect == 'Pan Right':
        zoom_expr = "1.05"
        x_expr = f"(iw-iw/zoom)*(1-cos(PI*on/{frames}))/2"
        y_expr = "(ih-ih/zoom)/2"
    else:  # None - subtle slow zoom
        zoom_expr = f"1+0.03*(1-cos(PI*on/{frames}))/2"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    
    return f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=1920x1080:fps={FPS}"


def create_image_clip(
    img_path: str,
    duration: float,
    output_path: str,
    effect: str,
    index: int
) -> str:
    """
    Create video clip from image with smooth zoom/pan effect.
    """
    # High quality scaling with lanczos
    vf_parts = [
        "scale=2560:1440:flags=lanczos:force_original_aspect_ratio=increase",
        "crop=2560:1440",
        "setsar=1"
    ]
    
    # Smooth zoom/pan
    zoom_filter = get_smooth_zoom_filter(effect, duration, index)
    vf_parts.append(zoom_filter)
    
    # Longer fade for smoother transitions
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
    
    # Encoding settings
    if _encoder == 'h264_nvenc':
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


def concatenate_videos_smooth(clips: list, output_path: str, crossfade_duration: float = 0.4) -> str:
    """Concatenate videos with smooth crossfade transitions"""
    
    if len(clips) == 1:
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-i', clips[0], '-c', 'copy', output_path], check=True)
        return output_path
    
    n = len(clips)
    
    # Get durations
    durations = []
    for clip in clips:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', clip
        ], capture_output=True, text=True)
        durations.append(float(result.stdout.strip()))
    
    # Build filter complex with 'fade' transition
    filter_parts = []
    current = "[0:v]"
    offset = durations[0] - crossfade_duration
    
    for i in range(1, n):
        next_in = f"[{i}:v]"
        out = f"[v{i}]" if i < n - 1 else "[vout]"
        filter_parts.append(f"{current}{next_in}xfade=transition=fade:duration={crossfade_duration}:offset={offset:.3f}{out}")
        current = out
        if i < n - 1:
            offset += durations[i] - crossfade_duration
    
    filter_complex = ";".join(filter_parts)
    
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
    for c in clips:
        cmd += ['-i', c]
    
    cmd += ['-filter_complex', filter_complex, '-map', '[vout]']
    
    if _encoder == 'h264_nvenc':
        cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '6M']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    
    cmd += ['-pix_fmt', 'yuv420p', '-an', output_path]
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        return output_path
    except subprocess.SubprocessError:
        # Fallback to simple concat
        print("   ⚠️ Crossfade failed, using simple concat")
        concat_file = output_path + ".txt"
        with open(concat_file, 'w') as f:
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n")
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0', '-i', concat_file,
                       '-c', 'copy', output_path], check=True)
        return output_path


def assemble_final_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str = None,
    music_path: str = None,
    music_volume: int = 20,
    total_duration: float = None,
    output_name: str = "Video"
) -> str:
    """
    Assemble final video with audio, subtitles, and optional music.
    """
    jid = hashlib.md5(f"{output_name}{time.time()}".encode()).hexdigest()[:8]
    final_output = OUTPUT_DIR / f"{output_name}_{jid}.mp4"
    
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-i', video_path, '-i', audio_path]
    
    # Music mixing
    if music_path and os.path.exists(music_path):
        cmd += ['-i', music_path, '-filter_complex',
               f"[1:a]volume=1.0[v];[2:a]volume={music_volume/100}[m];[v][m]amix=inputs=2:duration=first[a]",
               '-map', '0:v', '-map', '[a]']
    else:
        cmd += ['-map', '0:v', '-map', '1:a']
    
    # Subtitles with custom fonts directory
    # Use relative paths by running FFmpeg from temp directory
    use_subtitles = subtitle_path and os.path.exists(subtitle_path)
    if use_subtitles:
        # Get just the filename for relative path usage
        sub_filename = os.path.basename(subtitle_path)
        # Use forward slashes for fontsdir (FFmpeg on Windows needs this)
        fonts_rel = os.path.relpath(FONTS_DIR, TEMP_DIR).replace("\\", "/")
        cmd += ['-vf', f"subtitles='{sub_filename}':fontsdir='{fonts_rel}'"]
    
    # Trim to exact audio duration
    if total_duration:
        cmd += ['-t', f"{total_duration:.3f}"]
    
    # Encoding
    if _encoder == 'h264_nvenc':
        cmd += ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '8M']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
    
    cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(final_output)]
    
    # Run from temp directory if using subtitles (for relative path resolution)
    cwd = str(TEMP_DIR) if use_subtitles else None
    subprocess.run(cmd, check=True, timeout=300, cwd=cwd)
    
    return str(final_output)


def generate_video(
    images: list,
    audio_path: str,
    total_duration: float,
    effect: str = "Zoom Combo",
    subtitle_path: str = None,
    music_path: str = None,
    music_volume: int = 20,
    output_name: str = "Video",
    progress_callback=None
) -> str:
    """
    Full video generation pipeline.
    
    Args:
        images: List of image paths
        audio_path: Path to audio file
        total_duration: Total video duration in seconds
        effect: Transition effect
        subtitle_path: Path to ASS subtitle file
        music_path: Path to background music
        music_volume: Music volume percentage
        output_name: Output video name
        progress_callback: Progress callback function
    
    Returns:
        str: Path to generated video
    """
    jid = hashlib.md5(f"{output_name}{time.time()}".encode()).hexdigest()[:8]
    
    # Validate images
    valid_images = [img for img in images if img and validate_image(img)]
    if not valid_images:
        raise ValueError("No valid images provided")
    
    n = len(valid_images)
    
    # Calculate per-image duration
    crossfade = CROSSFADE_DURATION
    total_crossfade_time = crossfade * (n - 1) if n > 1 else 0
    per_image_duration = (total_duration + total_crossfade_time) / n
    per_image_duration = max(1.5, per_image_duration)
    
    print(f"\n🖼️ Processing {n} images...")
    print(f"   Duration: {per_image_duration:.2f}s each")
    
    # Create clips
    if progress_callback:
        progress_callback(0.4, "Creating clips...")
    
    clips = []
    for i, img_path in enumerate(valid_images):
        output = str(TEMP_DIR / f"clip_{jid}_{i}.mp4")
        result = create_image_clip(img_path, per_image_duration, output, effect, i)
        if result:
            clips.append(result)
            print(f"   ✅ {i+1}/{n}")
        if progress_callback:
            progress_callback(0.4 + 0.25 * (i + 1) / n, f"Clip {i+1}/{n}...")
    
    if not clips:
        raise ValueError("Failed to create any clips")
    
    # Concatenate with smooth crossfade
    if progress_callback:
        progress_callback(0.7, "Smooth transitions...")
    print("\n🎬 Concatenating with crossfade...")
    concat_output = str(TEMP_DIR / f"concat_{jid}.mp4")
    concatenate_videos_smooth(clips, concat_output, crossfade)
    print("   ✅ Done")
    
    # Final assembly
    if progress_callback:
        progress_callback(0.85, "Final assembly...")
    print("\n🎵 Final assembly...")
    
    final_path = assemble_final_video(
        video_path=concat_output,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        music_path=music_path,
        music_volume=music_volume,
        total_duration=total_duration,
        output_name=output_name
    )
    
    if progress_callback:
        progress_callback(1.0, "Done!")
    
    print(f"\n✅ {final_path}")
    return final_path


def get_effects() -> list:
    """Get list of available effects"""
    return EFFECTS


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False
