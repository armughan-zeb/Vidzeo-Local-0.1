# =============================================================================
# TRANSCRIPTION SERVICE - Whisper + ASS Subtitles
# =============================================================================

import hashlib
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEMP_DIR, FONTS_DIR, FONT_NAME_MAP, ANIMATIONS
)

# Global state
_whisper_model = None
_available = False


def init_whisper(device: str = "cpu"):
    """Initialize Whisper model"""
    global _whisper_model, _available
    
    if _whisper_model is not None:
        return _available
    
    try:
        import whisper
        print(f"   Loading Whisper ({device})...")
        _whisper_model = whisper.load_model("base", device=device)
        _available = True
        print("✅ Whisper loaded")
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
        _whisper_model = None
        _available = False
    
    return _available


def is_available():
    """Check if transcription is available"""
    return _available


def transcribe_audio(audio_path: str) -> tuple:
    """
    Transcribe audio to get word-level timestamps.
    
    Returns:
        tuple: (words, segments)
    """
    global _whisper_model
    
    if not _available or _whisper_model is None:
        return [], []
    
    print("   Transcribing...")
    result = _whisper_model.transcribe(audio_path, word_timestamps=True, language="en")
    
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


def to_ass_time(t: float) -> str:
    """Convert seconds to ASS timestamp format"""
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def hex_to_ass(c: str) -> str:
    """Convert hex color to ASS format"""
    c = c.lstrip("#")
    return f"&H00{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"


def hex_bgr(c: str) -> str:
    """Convert hex to BGR format"""
    c = c.lstrip("#")
    return f"{int(c[4:6], 16):02X}{int(c[2:4], 16):02X}{int(c[0:2], 16):02X}"


def escape_ass(s: str) -> str:
    """Escape special characters for ASS"""
    return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def get_font_name(dropdown_value: str) -> str:
    """Extract font family name from dropdown value"""
    if "|" in dropdown_value:
        # Robust split: take last part and strip whitespace
        filename = dropdown_value.split("|")[-1].strip()
    else:
        filename = dropdown_value
    return FONT_NAME_MAP.get(filename, filename)


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
    Generate ASS subtitle file from audio.
    
    Args:
        audio_path: Path to audio file
        caption_mode: "single" (one word), "word" (grouped), "line" (full sentence)
        ... styling parameters ...
    
    Returns:
        str: Path to generated ASS file
    """
    # Get font family name
    font_raw = font
    font = get_font_name(font)
    print(f"   🔡 Font Mapping: '{font_raw}' -> '{font}'")
    
    # Transcribe audio
    words, segments = transcribe_audio(audio_path)
    if not words:
        return None
    
    # Position alignment
    align = {"top": 8, "middle": 5, "bottom": 2}.get(position, 2)
    shadow = shadow_depth if shadow_on else 0
    shadow_ass = hex_to_ass(shadow_color).replace("&H00", "&H80")
    border_style = 3 if bg_on else 1
    bg_ass = hex_to_ass(bg_color).replace("&H00", f"&H{255 - int(bg_opacity * 2.55):02X}")
    bold_val = -1 if bold else 0
    
    # Custom style adjustments for karaoke_box
    if caption_mode == "karaoke_box":
        # Force box style for Highlight, keep Default as is (likely outline)
        hl_border_style = 3
        # Ensure distinct background for box (using highlight color as bg? or text color? usually text is white, box is color)
        # User said "background Goes to each word". 
        # Typically: White Text on Purple Box.
        # So Highlight Style: Text=White, Back=HighlightColor, BorderStyle=3.
        # But our Highlight Style def uses Primary=HighlightColor.
        # Let's swap: Text=TextColor, Back=HighlightColor.
        hl_primary = hex_to_ass(text_color)
        hl_back = hex_to_ass(highlight_color).replace("&H00", f"&H{255 - int(bg_opacity * 2.55):02X}")
    else:
        hl_border_style = border_style
        hl_primary = hex_to_ass(highlight_color)
        hl_back = shadow_ass if shadow_on else bg_ass

    # ASS header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},{hex_to_ass(text_color)},{hex_to_ass(text_color)},{hex_to_ass(outline_color)},{shadow_ass if shadow_on else bg_ass},{bold_val},0,0,0,100,100,0,0,{border_style},{outline_size},{shadow},{align},50,50,{margin_v},1
Style: Highlight,{font},{fontsize},{hl_primary},{hl_primary},{hex_to_ass(outline_color)},{hl_back},{bold_val},0,0,0,100,100,0,0,{hl_border_style},{outline_size + 1},{shadow},{align},50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    lines = [header]
    anim = ANIMATIONS.get(animation, ANIMATIONS["None"])
    anim_tag = "{" + anim.get("in", "") + "}" if anim.get("in") else ""
    
    hl_bgr = hex_bgr(highlight_color)
    txt_bgr = hex_bgr(text_color)
    
    if caption_mode == "single":
        # One word at a time
        for w in words:
            txt = w["word"].upper() if uppercase else w["word"]
            start, end = w["start"], w["end"]
            if end <= start:
                end = start + 0.25
            lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Highlight,,0,0,0,,{anim_tag}{escape_ass(txt)}\n")
    
    elif caption_mode == "line":
        # Full line at a time
        for seg in segments:
            txt = seg["text"].upper() if uppercase else seg["text"]
            lines.append(f"Dialogue: 0,{to_ass_time(seg['start'])},{to_ass_time(seg['end'])},Default,,0,0,0,,{anim_tag}{escape_ass(txt)}\n")

    elif caption_mode in ["karaoke", "karaoke_box"]:
        # Karaoke mode: Full line visible, active word highlighted
        for seg in segments:
            # Find words in this segment
            # We filter words that fall within segment time range
            seg_words = [w for w in words if w["start"] >= seg["start"] - 0.1 and w["end"] <= seg["end"] + 0.1]
            
            if not seg_words:
                continue

            # Iterate through each word to create a frame
            for k, active_word in enumerate(seg_words):
                start = active_word["start"]
                end = active_word["end"]
                
                # Fill gaps if words aren't contiguous (optional, but good for smoothness)
                if k < len(seg_words) - 1:
                    next_start = seg_words[k+1]["start"]
                    if next_start > end:
                        end = next_start # Extend to next word
                
                parts = []
                for m, w in enumerate(seg_words):
                    txt = w["word"].upper() if uppercase else w["word"]
                    if m == k:
                        # Active word
                        parts.append("{\\rHighlight}" + escape_ass(txt) + "{\\rDefault}")
                    else:
                        # Inactive word
                        parts.append(escape_ass(txt))
                    if m < len(seg_words) - 1:
                        parts.append(" ")
                
                # Use Default style for base, but join parts which switch to Highlight
                lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Default,,0,0,0,,{anim_tag}{''.join(parts)}\n")
    
    else:  # "word" mode - grouped words (legacy/standard)
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
                        parts.append("{\\rHighlight\\1c&H" + hl_bgr + "&}" + escape_ass(txt))
                    else:
                        parts.append("{\\rDefault\\1c&H" + txt_bgr + "&}" + escape_ass(txt))
                    if m < len(block) - 1:
                        parts.append(" ")
                lines.append(f"Dialogue: 0,{to_ass_time(start)},{to_ass_time(end)},Default,,0,0,0,,{''.join(parts)}\n")
            i = j
    
    # Save ASS file
    ass_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    ass_path = TEMP_DIR / f"captions_{ass_hash}.ass"
    
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write("".join(lines))
    
    print(f"   ✅ Captions generated: {ass_path.name}")
    return str(ass_path)
