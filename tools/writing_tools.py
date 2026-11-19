import re
from typing import List

from langchain_core.tools import tool

# narrow no-break / no-break / thin / figure spaces are used as thousands separators by many models
_SPACES = {ord(c): " " for c in "\u202f\u00a0\u2009\u2007\u2008\u200a"}
# 1,520  |  1 520  |  1520  |  20.42  - all read as the same kind of number
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w)")


def _numbers(text: str) -> set:
    found = set()
    for token in _NUMBER.findall((text or "").translate(_SPACES)):
        value = float(token.replace(",", "").replace(" ", ""))
        # ignore list numbers / tiny counts and plain years - they are rarely "facts"
        if value.is_integer() and (value <= 10 or 1900 <= value <= 2100):
            continue
        found.add(value)
    return found


def find_unsupported_numbers(source_text: str, draft_text: str) -> List[str]:
    """Numbers that appear in the draft but nowhere in the source material."""
    missing = _numbers(draft_text) - _numbers(source_text)
    return sorted(f"{m:g}" for m in missing)


@tool
def format_artical(title: str, content: str) -> str:
    """Format a finished draft as a clean Markdown article: adds the title as an H1
    (if missing), normalises blank lines and puts a blank line around every heading."""
    lines = [line.rstrip() for line in (content or "").strip().splitlines()]
    out = []
    for line in lines:
        is_heading = line.lstrip().startswith("#")
        if is_heading and out and out[-1] != "":
            out.append("")
        out.append(line)
        if is_heading:
            out.append("")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if not text.startswith("# "):
        text = f"# {title.strip()}\n\n{text}"
    return text + "\n"


format_article = format_artical        # correctly spelled alias


@tool
def word_count(text: str) -> dict:
    """Count words, characters, sentences and paragraphs and estimate reading time."""
    words = re.findall(r"\b[\w'-]+\b", text or "")
    sentences = [s for s in re.split(r"[.!?]+\s", (text or "").strip()) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    return {
        "words": len(words),
        "characters": len(text or ""),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "reading_time_minutes": round(len(words) / 200, 1),
    }


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(1, count)


@tool
def readability_score(text: str) -> dict:
    """Flesch Reading Ease and Flesch-Kincaid grade level (higher ease = easier to read;
    60-70 is plain English, 80+ is beginner friendly)."""
    words = re.findall(r"[A-Za-z']+", text or "")
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    if not words or not sentences:
        return {"error": "not enough text"}
    syllables = sum(_syllables(w) for w in words)
    wps, spw = len(words) / len(sentences), syllables / len(words)
    ease = 206.835 - 1.015 * wps - 84.6 * spw
    grade = 0.39 * wps + 11.8 * spw - 15.59
    label = ("very easy" if ease >= 80 else "easy" if ease >= 70 else "standard" if ease >= 60
             else "fairly difficult" if ease >= 50 else "difficult")
    return {"flesch_reading_ease": round(ease, 1), "grade_level": round(grade, 1),
            "label": label, "avg_words_per_sentence": round(wps, 1)}


@tool
def verify_numbers(source_text: str, draft_text: str) -> dict:
    """Fact-preservation check: lists numbers in the draft that do NOT appear in the
    source material (research / data results). Small integers (<=10) and years are ignored.
    Run it on your draft; fix or remove every unsupported number."""
    unsupported = find_unsupported_numbers(source_text, draft_text)
    return {"ok": not unsupported, "unsupported_numbers": unsupported}


@tool
def build_references(sources: List[str]) -> str:
    """Build a numbered Markdown 'Sources' section from a list of URLs / citations
    (duplicates removed, order kept). Only pass sources that really came from the research results."""
    seen, items = set(), []
    for s in sources or []:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            items.append(s)
    if not items:
        return ""
    return "## Sources\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(items, 1)) + "\n"


# source lines written by the research agent:  [S1-2] Title - https://url
SOURCE_LINE = re.compile(r"^\[(S\d+-\d+)\]\s+(.*?)\s+-\s+(https?://\S+)\s*$", re.M)
CITATION = re.compile(r"\[([^\[\]]*S\d+-\d+[^\[\]]*)\]")


def available_source_ids(source_text: str) -> set:
    return {m.group(1) for m in SOURCE_LINE.finditer(source_text)}


def cited_source_ids(text: str) -> list:
    ids = []
    for group in CITATION.findall(text):
        ids.extend(re.findall(r"S\d+-\d+", group))
    return ids


def renumber_citations(text: str, source_text: str):
    """Turn [S1-3] style ids into [1], [2] ... in order of first use, drop invented ids and
    build a Sources list with ONLY the sources the article really cites."""
    sources = {m.group(1): (m.group(2), m.group(3)) for m in SOURCE_LINE.finditer(source_text)}
    order = []

    def replace(match):
        numbers = set()
        for sid in re.findall(r"S\d+-\d+", match.group(1)):
            if sid not in sources:
                continue                                 # invented id -> removed
            if sid not in order:
                order.append(sid)
            numbers.add(order.index(sid) + 1)
        return "[" + ", ".join(str(n) for n in sorted(numbers)) + "]" if numbers else ""

    text = CITATION.sub(replace, text)
    references = "\n".join(f"[{i}] {sources[sid][0]} - {sources[sid][1]}"
                           for i, sid in enumerate(order, 1))
    return text, references


def max_table_rows(text: str) -> int:
    """Largest number of DATA rows (header and separator excluded) in any Markdown table."""
    best, current, header_seen = 0, 0, False
    for line in (text or "").splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|"):
            if re.match(r"^\|[\s:|-]+\|?$", stripped):
                continue                                   # separator row
            if not header_seen:
                header_seen = True                         # header row
                continue
            current += 1
        else:
            best, current, header_seen = max(best, current), 0, False
    return best


def repeated_table_labels(text: str) -> List[str]:
    """Row labels that appear in 2+ different Markdown tables (the same data shown twice)."""
    tables, current = [], None
    for line in (text or "").splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|"):
            if re.match(r"^\|[\s:|-]+\|?$", stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if current is None:
                current = set()
                tables.append(current)
                continue                                   # header row
            current.add(re.sub(r"[*_`]", "", cells[0]).strip().lower())
        else:
            current = None
    counts = {}
    for labels in tables:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return sorted(l for l, c in counts.items() if c >= 2 and l)


def mentions_hypothetical(text: str) -> bool:
    return bool(re.search(r"\b(hypothetical|fictional|made-up|imaginary|fake)\b", text or "", re.I))


WRITING_TOOLS = [format_artical, word_count, readability_score,
                 verify_numbers, build_references]
