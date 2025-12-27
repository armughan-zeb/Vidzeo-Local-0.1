# =============================================================================
# VIDZEO LOCAL - Configuration Constants
# =============================================================================

import os
from pathlib import Path

# =============================================================================
# DIRECTORIES
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
SERVER_DIR = Path(__file__).parent
FONTS_DIR = BASE_DIR / "fonts"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
for d in [FONTS_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VOICES (Kokoro TTS)
# =============================================================================

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
# FONTS
# =============================================================================

# Font URLs from Google Fonts GitHub
FONT_URLS = {
    "BebasNeue-Regular": "https://raw.githubusercontent.com/dharmatype/Bebas-Neue/master/fonts/BebasNeue(2018)ByDhamraType/ttf/BebasNeue-Regular.ttf",
    "Anton-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Montserrat-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Poppins-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Bold.ttf",
    "Poppins-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
    "Roboto-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "LilitaOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/lilitaone/LilitaOne-Regular.ttf",
    "FredokaOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/fredokaone/FredokaOne-Regular.ttf",
    "Creepster-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/creepster/Creepster-Regular.ttf",
    "Nosifer-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/nosifer/Nosifer-Regular.ttf",
    "Caveat-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "Pacifico-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "DancingScript-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "PatrickHand-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "PlayfairDisplay-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Cinzel-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",
    "TitanOne-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/titanone/TitanOne-Regular.ttf",
    "Bungee-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/bungee/Bungee-Regular.ttf",
    "Righteous-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/righteous/Righteous-Regular.ttf",
    "Audiowide-Regular": "https://raw.githubusercontent.com/google/fonts/main/ofl/audiowide/Audiowide-Regular.ttf",
    "Orbitron-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf",
    "ComicNeue-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
    "OpenSans-Bold": "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
}

# Font categories for UI
FONT_CATEGORIES = {
    "⬛ Bold": ["BebasNeue-Regular", "Anton-Regular", "Roboto-Bold", "TitanOne-Regular", "Bungee-Regular"],
    "🔷 Modern": ["Montserrat-Bold", "Poppins-Bold", "Poppins-Regular"],
    "🟣 Bubbly": ["LilitaOne-Regular", "FredokaOne-Regular"],
    "👻 Horror": ["Creepster-Regular", "Nosifer-Regular"],
    "✍️ Handwriting": ["Caveat-Bold", "Pacifico-Regular", "DancingScript-Bold", "PatrickHand-Regular"],
    "🎩 Formal": ["PlayfairDisplay-Bold", "Cinzel-Bold"],
    "🎨 Creative": ["Righteous-Regular", "Audiowide-Regular", "Orbitron-Bold"],
    "😊 Informal": ["ComicNeue-Bold", "OpenSans-Bold", "Arial"],
}

# Font filename → Font family name (for ASS subtitles)
FONT_NAME_MAP = {
    "BebasNeue-Regular": "Bebas Neue",
    "Anton-Regular": "Anton",
    "Roboto-Bold": "Roboto",
    "TitanOne-Regular": "Titan One",
    "Bungee-Regular": "Bungee",
    "Montserrat-Bold": "Montserrat",
    "Poppins-Bold": "Poppins",
    "Poppins-Regular": "Poppins",
    "LilitaOne-Regular": "Lilita One",
    "FredokaOne-Regular": "Fredoka One",
    "Creepster-Regular": "Creepster",
    "Nosifer-Regular": "Nosifer",
    "Caveat-Bold": "Caveat",
    "Pacifico-Regular": "Pacifico",
    "DancingScript-Bold": "Dancing Script",
    "PatrickHand-Regular": "Patrick Hand",
    "PlayfairDisplay-Bold": "Playfair Display",
    "Cinzel-Bold": "Cinzel",
    "Righteous-Regular": "Righteous",
    "Audiowide-Regular": "Audiowide",
    "Orbitron-Bold": "Orbitron",
    "ComicNeue-Bold": "Comic Neue",
    "OpenSans-Bold": "Open Sans",
    "Arial": "Arial",
}

# Flatten fonts for dropdown
FONTS_FLAT = []
for category, fonts in FONT_CATEGORIES.items():
    for font in fonts:
        FONTS_FLAT.append(f"{category} | {font}")

# =============================================================================
# IMAGE GENERATION
# =============================================================================

# Image providers - Pollinations is FREE (no API key needed)
IMAGE_PROVIDERS = ['Together AI', 'Pollinations AI (Free)']

# Models by provider
IMAGE_MODELS = {
    'Together AI': {
        'Flux Schnell': 'black-forest-labs/FLUX.1-schnell',
    },
    'Pollinations AI (Free)': {
        'Flux (Free)': 'flux',
        'Turbo (Free)': 'turbo',
    }
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

# Style lock header
STYLE_HEADER = """MASTERPIECE, HIGH QUALITY,
SINGLE CLEAR SUBJECT,
CENTERED COMPOSITION,
NO TEXT, NO LOGOS, NO SYMBOLS,
CLEAN EDGES, STABLE STRUCTURE,
CONSISTENT STYLE THROUGHOUT"""

# 20 Image Style Presets
IMAGE_STYLES = {
    "No Style": "",
    
    "2D Cartoon (Flat)": """2D CARTOON ILLUSTRATION,
FLAT VECTOR LOOK, SOLID COLOR AREAS ONLY,
THICK BLACK OUTLINES, NO DEPTH CUES, NO GRADIENTS,
SIMPLE GEOMETRIC FORMS, BRIGHT CARTOON PALETTE,
EXAGGERATED EXPRESSIONS, PLAIN BACKGROUND""",
    
    "3D Cartoon (Toy-like)": """3D CARTOON CHARACTER,
SMOOTH PLASTIC SURFACE, ROUNDED SHAPES,
STYLIZED NON-REALISTIC PROPORTIONS,
SOFT STUDIO LIGHTING, GLOBAL ILLUMINATION,
SATURATED COLORS, CLEAN STUDIO BACKGROUND""",
    
    "2D Stickman Pencil": """2D STICKMAN DRAWING,
BLACK PENCIL ON WHITE PAPER, CIRCULAR HEAD,
STRAIGHT LINE LIMBS, NO FILL COLOR, NO SHADING,
HAND-DRAWN SKETCH STYLE""",
    
    "2D Stickman Cartoon": """2D STICKMAN CARTOON,
THICK BLACK OUTLINES, SIMPLE CIRCLE HEAD,
STRAIGHT LINE ARMS AND LEGS, FLAT SOLID COLORS,
MINIMAL CARTOON FACE, CLEAN SHAPES, SIMPLE BACKGROUND""",
    
    "3D Stickman Cartoon": """3D STICKMAN CARTOON CHARACTER,
SPHERE HEAD, CYLINDER LIMBS, SMOOTH MATTE MATERIAL,
BRIGHT PLAYFUL COLORS, NON-HUMAN PROPORTIONS,
SOFT STUDIO LIGHTING""",
    
    "Pencil Sketch (Fine Art)": """REALISTIC PENCIL SKETCH,
BLACK AND WHITE ONLY, GRAPHITE LINE WORK,
CROSS-HATCH SHADING, VISIBLE PAPER TEXTURE,
HAND-DRAWN ART STYLE""",
    
    "Photorealistic": """PHOTOREALISTIC IMAGE,
REAL CAMERA PHOTOGRAPH, NATURAL LIGHT INTERACTION,
REALISTIC SURFACE TEXTURE, HIGH DYNAMIC RANGE,
SHALLOW DEPTH OF FIELD, 85MM LENS LOOK""",
    
    "Cinematic Realism": """CINEMATIC REALISM,
MOVIE STILL FRAME, DRAMATIC LIGHT DIRECTION,
CONTROLLED COLOR GRADING, VOLUMETRIC LIGHT RAYS,
ANAMORPHIC LENS FEEL, EPIC FRAMING""",
    
    "1950s Vintage Photo": """1950s VINTAGE PHOTOGRAPH,
MONOCHROME OR FADED COLOR, VISIBLE FILM GRAIN,
SOFT FOCUS LENS, LOW CONTRAST, AGED PHOTO CHARACTER""",
    
    "Comic Book": """COMIC BOOK ILLUSTRATION,
BOLD INKED OUTLINES, HALFTONE DOT SHADING,
HIGH CONTRAST COLORS, DRAMATIC POSE, DYNAMIC COMPOSITION""",
    
    "Anime (Cel Shading)": """ANIME STYLE ILLUSTRATION,
CLEAN LINE ART, CEL SHADING, FLAT COLOR ZONES,
LARGE EXPRESSIVE EYES, STYLIZED PROPORTIONS""",
    
    "Studio Ghibli": """STUDIO GHIBLI STYLE,
HAND-PAINTED LOOK, SOFT PASTEL COLORS,
GENTLE LIGHTING, WATERCOLOR-LIKE TEXTURE, DREAMY ATMOSPHERE""",
    
    "Horror": """HORROR ART STYLE,
DARK TONAL PALETTE, LOW-KEY LIGHTING, HEAVY SHADOWS,
FOG AND GRAIN, TENSE ATMOSPHERE""",
    
    "Medieval Art": """MEDIEVAL ILLUSTRATION,
ILLUMINATED MANUSCRIPT STYLE, FLAT PERSPECTIVE,
MUTED EARTH TONES, PARCHMENT TEXTURE, DECORATIVE DETAILS""",
    
    "Furry Art": """FURRY ART STYLE,
ANTHROPOMORPHIC ANIMAL CHARACTER, SOFT FUR DETAIL,
EXPRESSIVE FACE, DIGITAL CHARACTER ILLUSTRATION""",
    
    "Digital Illustration": """PROFESSIONAL DIGITAL ILLUSTRATION,
CLEAR SUBJECT SEPARATION, STYLIZED DESIGN LANGUAGE,
BALANCED COLOR PALETTE, CLEAN COMPOSITION""",
    
    "Watercolor Painting": """WATERCOLOR PAINTING,
SOFT COLOR BLEEDING, VISIBLE PAPER GRAIN, LIGHT WASHES,
ORGANIC EDGES, HAND-PAINTED LOOK""",
    
    "B&W Historical Photo": """BLACK AND WHITE HISTORICAL PHOTOGRAPH,
DOCUMENTARY REALISM, NATURAL LIGHT, VISIBLE FILM GRAIN,
CLASSIC CONTRAST""",
    
    "B&W Pencil Sketch": """BLACK AND WHITE PENCIL SKETCH,
FINE GRAPHITE LINES, REALISTIC SHADING,
HAND-DRAWN STYLE, WHITE PAPER BACKGROUND""",
    
    "Oil Painting": """OIL PAINTING,
SOFT DISTORTED BRUSH STROKES, IMPRESSIONISTIC STYLE,
BLENDED COLORS, VISIBLE CANVAS TEXTURE, ARTISTIC COMPOSITION""",
}

IMAGE_STYLE_NAMES = list(IMAGE_STYLES.keys())

# =============================================================================
# SCRIPT GENERATION
# =============================================================================

SCRIPT_DURATIONS = [
    "30 seconds", "1 minute", "2 minutes", "3 minutes", "5 minutes",
    "10 minutes", "15 minutes", "30 minutes", "45 minutes",
    "1 hour", "1.5 hours", "2 hours", "3 hours"
]

# Kokoro TTS speaks at ~170 words per minute (measured rate)
WORDS_PER_MINUTE = 170

# Keywords that trigger automatic web research
RESEARCH_TRIGGERS = [
    "news", "latest", "recent", "current", "today", "2024", "2025", "update",
    "breaking", "new", "trending", "viral",
    "facts", "statistics", "stats", "data", "numbers", "study", "research",
    "percent", "million", "billion", "how many", "how much",
    "ai", "technology", "tech", "science", "discovery", "invention", "innovation",
    "crypto", "bitcoin", "blockchain", "smartphone", "app", "software",
    "market", "economy", "stock", "price", "cost", "investment", "business",
    "company", "startup", "billionaire", "richest",
    "country", "countries", "world", "global", "war", "election", "president",
    "government", "politics", "climate", "environment",
    "movie", "film", "celebrity", "famous", "star", "singer", "actor",
    "sport", "championship", "record", "winner"
]

# =============================================================================
# NSFW SAFETY
# =============================================================================

BLOCK_WORDS = [
    "nude", "naked", "sex", "sexual", "erotic", "porn", "pornographic",
    "fetish", "boobs", "breasts", "ass", "lingerie", "bikini", "underwear",
    "nsfw", "xxx", "adult", "explicit", "sensual", "seductive", "intimate",
    "provocative", "revealing", "topless", "bottomless", "stripper", "escort"
]

SAFE_CONTENT_PREFIX = (
    "professional photography, family-friendly, PG-13, modest clothing, "
    "no suggestive content, appropriate for all ages, tasteful composition, "
    "high quality stock photo style"
)

NSFW_THRESHOLD = 0.3
MAX_REGEN_TRIES = 3

# =============================================================================
# VIDEO PROCESSING
# =============================================================================

FPS = 30
CROSSFADE_DURATION = 0.4

EFFECTS = ['Zoom Combo', 'Zoom In', 'Zoom Out', 'Pan Left', 'Pan Right', 'None']

ANIMATIONS = {
    "None": {"in": ""},
    "Fade": {"in": "\\fad(150,100)"},
    "Pop": {"in": "\\t(0,80,\\fscx110\\fscy110)\\t(80,160,\\fscx100\\fscy100)"},
}
