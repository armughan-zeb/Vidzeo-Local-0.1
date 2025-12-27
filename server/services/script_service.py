# =============================================================================
# SCRIPT SERVICE - AI Script Generation from Title
# =============================================================================

import re
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SCRIPT_DURATIONS, WORDS_PER_MINUTE, RESEARCH_TRIGGERS


def get_duration_minutes(duration_str: str) -> float:
    """Convert duration string to minutes"""
    if "second" in duration_str:
        return int(duration_str.split()[0]) / 60
    elif "hour" in duration_str:
        hours = float(duration_str.split()[0])
        return hours * 60
    else:
        return int(duration_str.split()[0])


def needs_research(title: str, style_prompt: str = "") -> bool:
    """
    Auto-detect if topic needs current/fresh data from web search.
    Returns True if the topic likely needs recent facts, news, or statistics.
    """
    text = f"{title} {style_prompt}".lower()
    
    # Check for year references (2023+)
    if re.search(r'\b20[2-9][3-9]\b', text):
        return True
    
    # Check trigger keywords
    for trigger in RESEARCH_TRIGGERS:
        if trigger in text:
            return True
    
    return False


def clean_script_output(script: str) -> str:
    """
    Post-process LLM output to strip any meta-text that slipped through.
    Removes common AI response patterns that shouldn't be in the final script.
    """
    lines = script.strip().split('\n')
    cleaned_lines = []
    
    # Patterns to remove at start of script
    skip_patterns = [
        "here is", "here's", "sure,", "sure!", "certainly",
        "of course", "absolutely", "great!", "no problem",
        "here are", "i'd be happy", "let me", "i'll",
        "(start of script)", "script:", "video script:"
    ]
    
    # Remove header lines that match skip patterns
    started_content = False
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines at the start
        if not started_content and not line_lower:
            continue
        
        # Skip lines matching skip patterns
        if not started_content and any(line_lower.startswith(p) for p in skip_patterns):
            continue
        
        # Skip lines that look like headers (e.g., "## Introduction")
        if line_lower.startswith('#') or line_lower.startswith('**'):
            continue
        
        # Skip lines with brackets [like this]
        if '[' in line and ']' in line:
            line = re.sub(r'\[.*?\]', '', line).strip()
            if not line:
                continue
        
        started_content = True
        cleaned_lines.append(line)
    
    # Join and clean up extra whitespace
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 newlines in a row
    
    return result.strip()


def search_web_for_facts(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for relevant facts and figures"""
    try:
        from duckduckgo_search import DDGS
        
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


def generate_script_chunk(
    groq_client,
    title: str,
    style_prompt: str,
    chunk_num: int,
    total_chunks: int,
    previous_summary: str,
    target_words: int,
    research_context: str = ""
) -> str:
    """Generate a single chunk of the script with context continuity"""
    
    chunk_words = target_words // total_chunks
    current_date = datetime.now().strftime("%B %Y")
    current_year = datetime.now().year
    
    # Position context
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
    
    system_prompt = f"""You are an elite viral video scriptwriter. Today is {current_date}. You create scripts for {current_year} audiences.

CRITICAL FORMATTING RULES (MUST FOLLOW):
1. NEVER start with meta-text like "Here is a X minute script" or "Sure, here's..."
2. NEVER include any commentary, headers, or explanations
3. START IMMEDIATELY with an attention-grabbing hook - first 3 words must hook the viewer
4. Output ONLY the raw spoken words - exactly what the narrator will say
5. NO stage directions, NO [brackets], NO markdown, NO formatting

SCRIPT STRUCTURE (Build this flow):
- HOOK (0-3 sec): Shock statement, bold claim, or irresistible question
- CURIOSITY LOOP: Promise value, tease what's coming ("But here's where it gets crazy...")
- CONTENT BODY: Deliver insights with "rollercoaster pacing" - highs (revelations) and lows (tension)
- EMOTIONAL DEPTH: Make it personal, relatable, use "you" language
- MICRO-HOOKS: Every 15-20 seconds, add a cliffhanger or curiosity pull
- HUMOR/INTERACTION: Sprinkle conversational asides ("I know, crazy right?", "Wait for this...")
- CTA: End with clear action ("Subscribe", "Comment X", "Share this")
- OUTRO: Satisfying closer that loops back to the hook

PACING RULES:
- Short punchy sentences (5-12 words each)
- Vary rhythm: short-short-medium-short-long-short
- Build tension then release with insights
- Use power words: "secret", "dangerous", "discover", "shocking", "hidden"

VOICE/TONE:
- Speak like a knowledgeable friend, not a lecturer
- Confident but conversational
- Use current {current_year} slang sparingly but naturally
- Reference current events/trends when relevant

{f'STYLE PREFERENCE: {style_prompt}' if style_prompt else ''}

{position_context}
{research_section}"""

    user_prompt = f"""Write EXACTLY {chunk_words} words for this video script:

TITLE: {title}

STRICT REQUIREMENTS:
- Word count MUST be between {int(chunk_words * 0.9)}-{int(chunk_words * 1.1)} words
- First sentence MUST be an attention-grabbing hook (NO intro text)
- Start speaking IMMEDIATELY - your first word is the first word the narrator speaks
- Include at least 2 curiosity hooks scattered throughout
- End with a call-to-action and satisfying closer
- Write ONLY the spoken words - nothing else

FORBIDDEN (will be rejected):
- "Here is a X minute script..."
- "Sure, here's..."
- Any brackets [like this]
- Any headers or sections
- Any meta-commentary about the script itself

BEGIN YOUR SCRIPT NOW (start with the hook):"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=min(chunk_words * 2, 8000)
        )
        
        script_text = response.choices[0].message.content.strip()
        
        # Post-processing: Strip any meta-text that slipped through
        script_text = clean_script_output(script_text)
        
        return script_text
    except Exception as e:
        raise ValueError(f"Script generation failed: {str(e)}")


def summarize_chunk(groq_client, script_chunk: str) -> str:
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


def generate_script_from_title(
    title: str,
    duration: str,
    style_prompt: str,
    groq_api_key: str,
    use_research: str = "auto"
) -> str:
    """
    Generate a complete video script from a title.
    
    Features:
    - Duration-based length calculation
    - AUTO web research (detects when needed based on topic)
    - Current date injection
    - Smart chunking for long scripts
    - Quality script elements (hooks, suspense, CTAs)
    
    Args:
        title: Video title
        duration: Duration string (e.g., "2 minutes")
        style_prompt: Style description
        groq_api_key: Groq API key
        use_research: "auto" (smart detection), True (always), False (never)
    
    Returns:
        str: Generated script text
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
    except ImportError:
        raise ValueError("Groq package not installed")
    
    if not groq_api_key:
        raise ValueError("Groq API key required for script generation")
    
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
    MAX_WORDS_PER_CHUNK = 800
    
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
    
    # Safety truncation: If script is too long, truncate at sentence boundary
    max_allowed = int(target_words * 1.15)  # Allow 15% buffer
    if word_count > max_allowed:
        print(f"   ⚠️ Script too long ({word_count} words), truncating to ~{target_words} words...")
        words = full_script.split()
        truncated_text = ' '.join(words[:target_words])
        
        # Find last sentence boundary (. ! ?)
        for i in range(len(truncated_text) - 1, max(len(truncated_text) - 100, 0), -1):
            if truncated_text[i] in '.!?':
                full_script = truncated_text[:i+1]
                break
        else:
            full_script = truncated_text + '.'
        
        word_count = len(full_script.split())
    
    print(f"   ✅ Script complete: {word_count} words (target: {target_words})")
    
    return full_script


def get_durations() -> list:
    """Get list of available duration options"""
    return SCRIPT_DURATIONS
