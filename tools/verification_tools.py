"""Claim-by-claim verification: every important sentence is checked against the evidence it actually cites
(pure code, no LLM - fast, free, deterministic). It is a lexical-overlap heuristic, not logical entailment,
so it is a backstop next to the LLM fact check. Can also REMOVE the sentences it cannot verify."""
import re
from typing import Dict, List, Optional, Tuple

CITATION_GROUP = re.compile(r"\[([^\[\]]*S\d+-\d+[^\[\]]*)\]")
SOURCE_HEADER = re.compile(r"^\[(S\d+-\d+)\]\s+(.*?)\s+-\s+(https?://\S+)\s*$")
OFFICIAL_TYPES = {"official documentation", "research paper"}
# claims about these properties must rest on documentation or a paper, not on a blog
SCOPE_TERMS = ("schedul", "cron", "production-ready", "production-grade", "scalab", "durab", "enterprise-grade")
UNSUPPORTED_PROBLEMS = {"no_citation", "unknown_source", "weak_support", "scope"}

# models write non-breaking hyphens / dashes / narrow spaces; the sources use plain ones
_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad"}
_SPACES = {ord(c): " " for c in "\u202f\u00a0\u2009\u2007\u2008\u200a\u200b"}

_STOP = set("""this that with from have has had been were was are for and the its their they them which while when
where what who whom whose also into onto over under than then there these those such can could would should will may
might must not but about between through during each other more most some any all both either neither across around
based using use used uses make makes made allow allows allowed enabling enables enable provide provides provided
including includes include like well very many much often just only even still thus however""".split())

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z\[])")
_CITE_GROUP_RE = r"\[[^\[\]]*S\d+-\d+[^\[\]]*\]"
_FOREIGN_TAG = re.compile(r"\s?\[(?:[A-Za-z]{1,3}\d{1,3}|\d{1,2})\]")      # [T1] [D2] [3] - not a SOURCES id


def normalize_text(text: str) -> str:
    return (text or "").translate(_DASHES).translate(_SPACES)


def parse_sources(text: str) -> Dict[str, dict]:
    """Read the numbered SOURCES block written by the research agent:
       [S1-2] Title - https://url / type: ... / > snippet"""
    sources, current = {}, None
    for line in (text or "").splitlines():
        header = SOURCE_HEADER.match(line.rstrip())
        if header:
            current = {"title": header.group(2), "url": header.group(3), "type": "", "snippet": ""}
            sources[header.group(1)] = current
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("type:"):
            current["type"] = stripped[5:].strip().lower()
        elif stripped.startswith(">"):
            current["snippet"] = (current["snippet"] + " " + stripped[1:].strip()).strip()
        elif stripped:
            current = None                                  # the sources block ended
    return sources


def compact_evidence(sources: Dict[str, dict]) -> str:
    """The sources in the smallest useful form for the LLM fact checker (id, type, title, snippet)."""
    return "\n".join(f"[{sid}] ({s['type'] or 'source'}) {s['title']}: {s['snippet']}" for sid, s in sources.items())


def cited_ids(text: str) -> List[str]:
    ids = []
    for group in CITATION_GROUP.findall(text or ""):
        ids.extend(re.findall(r"S\d+-\d+", group))
    return ids


def foreign_citation_tags(text: str) -> List[str]:
    """Citation-like tags that are not SOURCES ids ([T1], [D1], [3]) - the model must not invent them."""
    return sorted({m.strip() for m in _FOREIGN_TAG.findall(text or "")})


def strip_foreign_tags(text: str) -> str:
    return _FOREIGN_TAG.sub("", text or "")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def content_words(text: str) -> set:
    """Meaningful words, stems, hyphen-insensitive ("long-running" = "long" + "running", any dash character)."""
    text = CITATION_GROUP.sub(" ", normalize_text(text)).lower()
    words = set()
    for token in re.findall(r"[a-z][a-z0-9+#'-]*", text):
        for part in token.split("-"):
            if len(part) >= 4 and part not in _STOP:
                words.add(_stem(part))
    return words


def _prepare_line(line: str) -> str:
    line = re.sub(r"^[-*\u2022\d.)\s]+", "", line).replace("**", "").replace("*", "")
    # a citation that follows the full stop ("... behavior. [S0-1]") is moved back to its sentence
    return re.sub(rf"([.!?])\s+((?:{_CITE_GROUP_RE}\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", line)


def split_claims(text: str) -> List[str]:
    """Sentences / bullets of a text. Headings and code are skipped; table rows only count when they carry a citation."""
    claims = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line[0] in "#`":
            continue
        if line.startswith("|"):
            if re.match(r"^\|[\s:|-]+\|?$", line) or not re.search(r"S\d+-\d+", line):
                continue
            claims.append(" ".join(c.strip() for c in line.strip("|").split("|") if c.strip()))
            continue
        prepared = _prepare_line(line)
        if re.match(r"(?i)\s*(illustration|gaps)\s*:", prepared):
            continue                                      # labelled illustrations and gap notes are not claims
        for sentence in _SENTENCE_SPLIT.split(prepared):
            sentence = sentence.strip()
            if len(sentence) >= 30:
                claims.append(sentence)
    return claims


def verify_claims(text: str, sources: Dict[str, dict], topic_terms: Optional[List[str]] = None,
                  min_support: float = 0.30) -> List[dict]:
    """Problems found, each {claim, ids, problem, detail}:
       unknown_source - cites an id that does not exist
       no_citation    - talks about the topic but cites nothing
       weak_support   - the cited snippets hardly contain the claim's content words
       scope          - claims scheduling / scalability / production-readiness without documentation backing"""
    issues = []
    for claim in split_claims(text):
        low = normalize_text(claim).lower()
        if low.startswith(("illustration:", "gaps:", "note:", "sources")):
            continue
        ids = cited_ids(claim)
        words = content_words(claim)
        if not ids:
            if any(t in low for t in (topic_terms or [])) and len(words) >= 6:
                issues.append({"claim": claim, "ids": [], "problem": "no_citation",
                               "detail": "a statement about the topic must cite a source id, be deleted, or sit in a "
                                         "clearly labelled 'Illustration:' paragraph"})
            continue
        unknown = [i for i in ids if i not in sources]
        if unknown:
            issues.append({"claim": claim, "ids": ids, "problem": "unknown_source",
                           "detail": "unknown source id: " + ", ".join(unknown)})
            continue
        evidence = " ".join(f"{sources[i]['title']} {sources[i]['snippet']}" for i in ids)
        if len(words) >= 5:                                   # very short list items are too thin to score
            support = len(words & content_words(evidence)) / len(words)
            if support < min_support:
                issues.append({"claim": claim, "ids": ids, "problem": "weak_support",
                               "detail": f"only {support:.0%} of its key words appear in the cited sources"})
        for term in SCOPE_TERMS:
            if term in low:
                backed = any(sources[i]["type"] in OFFICIAL_TYPES and term in sources[i]["snippet"].lower()
                             for i in ids)
                if not backed:
                    issues.append({"claim": claim, "ids": ids, "problem": "scope",
                                   "detail": (f"'{term}...' needs a documentation or paper source that says so. "
                                              "Running a graph is not the same as an external scheduler.")})
                break
    return issues


def format_issues(issues: List[dict], limit: int = 6) -> List[str]:
    return [f"[{i['problem']}] \"{i['claim'][:140]}\" - {i['detail']}" for i in issues[:limit]]


def issue_key(issues: List[dict]) -> str:
    """Stable fingerprint of a set of problems: the same key twice in a row means the rewrite did not fix them."""
    return "|".join(sorted(_key(i["claim"]) for i in issues))


def _key(sentence: str) -> str:
    sentence = re.sub(r"^[-*\u2022\d.)\s]+", "", sentence)
    sentence = CITATION_GROUP.sub("", normalize_text(sentence)).lower()
    return re.sub(r"[^a-z0-9]+", "", sentence)


def remove_claims(text: str, issues: List[dict]) -> Tuple[str, List[str]]:
    """Delete the sentences named in `issues` from `text` (the deterministic last resort when a rewrite keeps
    repeating an unsupported sentence). Returns (new text, removed sentences)."""
    keys = {_key(i["claim"]) for i in issues if i["problem"] in UNSUPPORTED_PROBLEMS}
    out, removed = [], []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#`":
            out.append(line)
            continue
        if stripped.startswith("|"):
            if _key(" ".join(c for c in stripped.strip("|").split("|"))) in keys:
                removed.append(stripped)
            else:
                out.append(line)
            continue
        kept, dropped_previous = [], False
        for piece in _SENTENCE_SPLIT.split(line):
            key = _key(piece)
            if key and key in keys:
                removed.append(piece.strip())
                dropped_previous = True
                continue
            if not key and dropped_previous:               # a dangling "[S0-1]" that belonged to the removed sentence
                continue
            dropped_previous = False
            kept.append(piece)
        if kept:
            out.append(" ".join(kept).rstrip())
    return "\n".join(out), removed
