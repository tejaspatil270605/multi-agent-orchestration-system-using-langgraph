import os
import re
import ipaddress
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import List
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

HEADERS = {"User-Agent": "multi-agent-research/1.0"}


def _error(message: str) -> List[dict]:
    return [{"error": message}]


@tool
def tavily_search(query: str) -> List[dict]:
    """Search the web with Tavily. Returns a list of results with title, url,
    content (snippet), published_date and score. Use for current facts and general research."""
    try:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return _error("TAVILY_API_KEY is not set")
        response = TavilyClient(api_key=api_key).search(
            query=query, max_results=5, search_depth="advanced"
        )
        return [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "published_date": r.get("published_date"),
                "score": r.get("score"),
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        return _error(f"tavily_search failed: {e}")


@tool
def wikipedia_search(query: str) -> List[dict]:
    """Search Wikipedia and return up to 3 article introductions (title, url, extract).
    Best for definitions, background and well-established concepts."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": query, "gsrlimit": 3, "prop": "extracts",
                "exintro": 1, "explaintext": 1, "exchars": 1200,
            },
            headers=HEADERS, timeout=10,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        return [
            {
                "title": p.get("title"),
                "url": f"https://en.wikipedia.org/?curid={p.get('pageid')}",
                "content": p.get("extract"),
            }
            for p in sorted(pages.values(), key=lambda p: p.get("index", 0))
        ]
    except Exception as e:
        return _error(f"wikipedia_search failed: {e}")


@tool
def arxiv_search(query: str) -> List[dict]:
    """Search arXiv for research papers. Returns up to 3 papers with title, url,
    published date and abstract. Use for technical / scientific topics."""
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3,
                    "sortBy": "relevance"},
            headers=HEADERS, timeout=8,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in ET.fromstring(r.text).findall("a:entry", ns):
            out.append({
                "title": " ".join((entry.findtext("a:title", "", ns)).split()),
                "url": entry.findtext("a:id", "", ns),
                "published_date": entry.findtext("a:published", "", ns),
                "content": " ".join((entry.findtext("a:summary", "", ns)).split())[:1200],
            })
        return out
    except Exception as e:
        return _error(f"arxiv_search failed: {e}")


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _is_public_http_url(url: str) -> bool:
    """Basic SSRF guard: only http(s) and no localhost / private IP literals."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host == "localhost" or host.endswith(".local"):
            return False
        try:
            ip = ipaddress.ip_address(host)
            return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
        except ValueError:
            return True                    # a normal domain name
    except Exception:
        return False


@tool
def fetch_webpage(url: str, max_chars: int = 3000) -> List[dict]:
    """Download a public web page and return its readable text (truncated to
    max_chars). Use it to read the full content of a promising search result."""
    if not _is_public_http_url(url):
        return _error("URL rejected: only public http(s) URLs are allowed")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        parser = _TextExtractor()
        parser.feed(r.text)
        text = " ".join(parser.parts)
        return [{"url": url, "content": text[:max(200, min(int(max_chars), 8000))]}]
    except Exception as e:
        return _error(f"fetch_webpage failed: {e}")


@tool
def get_current_datetime() -> str:
    """Return today's date and time (ISO format). Use to judge how recent a source is."""
    return datetime.now().isoformat(timespec="seconds")


_JUNK_LINE = re.compile(
    r"^(copy path|copy raw file|download raw file|raw|outline|edit and raw actions|"
    r"file metadata and controls|all scaffoldings|skip to (main )?content|menu|share|"
    r"subscribe|sign in|log ?in|cookies?|accept all|image \d+|last updated:?|table of contents)$"
    r"|^\d+ lines? .*bytes|^\u2190",
    re.I,
)


_PROMO = re.compile(
    r"(get tickets|register now|sign up|sign in|subscribe|buy now|book a demo|join (us|the waitlist)|"
    r"early[- ]bird|limited time|free trial|download now|read more|learn more|click here|"
    r"llms\.txt|fetch the complete documentation index|use this file to discover|"
    r"accept (all )?cookies|all rights reserved|"
    r"for (individuals|businesses|universities|governments)|\blog in\b|join for free|"
    r"skip to main content|open navigation menu|found this document useful|\b\d+ views\b|"
    r"get hands-on experience|written by .{0,40} staff|join one of the largest)",
    re.I,
)
_MOJIBAKE = re.compile(r"[\u00c2-\u00f4][\u0080-\u00bf]{1,3}")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 ("\u00e2\u0080\u00a2" instead of a bullet)."""
    def repair(match):
        try:
            return match.group(0).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return match.group(0)
    return _MOJIBAKE.sub(repair, text or "")
_EVENT = re.compile(r"\b(is|are) coming to [A-Z]")           # "... is coming to London on October 13!"
_LEADING_FRAGMENT = re.compile(r"^[^.!?]{0,40}?[\u2014\u2013]\s*(?=[A-Z])")   # "Morgan, and more-LangGraph is ..."


def _strip_fragment(sentence: str) -> str:
    """Remove a broken lead-in cut from the middle of the page ("...Morgan, and more-LangGraph is...")."""
    return _LEADING_FRAGMENT.sub("", sentence, count=1).strip()


_PRODUCT_NAME = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+$")   # LangSmith, LangChain, PostgreSQL


def clean_snippet(text, limit: int = 300) -> str:
    """Remove web-page junk (menus, GitHub/YouTube UI text, markdown images, adverts and
    boilerplate sentences, very short fragments) and cut at a sentence boundary.
    Short lines are kept when they are product names (LangSmith) or introduce a list ("pair X with:")."""
    if not text:
        return ""
    text = fix_mojibake(str(text))
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)               # markdown images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # [label](url) -> label
    kept = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" #*>|-_/\t")
        if not line or _JUNK_LINE.match(line):
            continue
        if len(line) < 30 and not re.search(r"[.!?]$", line):
            is_product = bool(_PRODUCT_NAME.match(line))
            is_list_intro = line.endswith(":") and " " in line
            if not (is_product or is_list_intro):
                continue                               # headings / button labels
        kept.append(line)
    cleaned = " ".join(kept)
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", cleaned)
                 if x and not _PROMO.search(x) and not _EVENT.search(x)]
    if sentences:
        sentences[0] = _strip_fragment(sentences[0])
    if len(sentences) > 1 and sentences[0][:1].islower():
        sentences = sentences[1:]                      # text starts in the middle of a sentence
    cleaned = " ".join(x for x in sentences if x)
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    last_stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:last_stop + 1] if last_stop > limit * 0.6 else cut.rstrip() + "..."


# ---- relevance: which words are the TOPIC of the research? ----
_GENERIC = set("""the and for with what how why are is of in on to a an its their from into about using use used
uses official documentation docs overview core concepts concept architecture components component features
feature modules typical cases case examples example applications application recent developments development
releases release integration integrations integrate tutorials tutorial beginners beginner community contributions
contribution github repository guide introduction explained explain best practices comparison versus vs python
main purpose list provide new latest 2024 2025 2026 pipelines data llm llms ai""".split())


def topic_terms(queries: List[str], max_terms: int = 3) -> List[str]:
    """Words that appear in most search queries and are not generic research words
    (for 'LangGraph purpose', 'LangGraph architecture' ... -> ['langgraph'])."""
    if not queries:
        return []
    counts = {}
    for query in queries:
        for word in set(re.findall(r"[a-z0-9][a-z0-9+#.-]{2,}", query.lower())):
            if word not in _GENERIC:
                counts[word] = counts.get(word, 0) + 1
    need = max(2, int(0.6 * len(queries) + 0.5))
    terms = [w for w, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= need]
    return terms[:max_terms]


def is_relevant(hit: dict, terms: List[str]) -> bool:
    """A hit is relevant if it mentions at least one topic term (no terms -> keep everything)."""
    if not terms:
        return True
    haystack = " ".join(str(hit.get(k) or "") for k in ("title", "url", "content")).lower()
    return any(term in haystack for term in terms)


def best_excerpt(text: str, terms: List[str], limit: int = 400) -> str:
    """Pick the most informative sentences of a long page text: those that mention the topic."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text or "") if 40 <= len(x.strip()) <= 400]
    sentences = [_strip_fragment(x) for x in sentences if not _PROMO.search(x) and not _EVENT.search(x)]
    sentences = [x for x in sentences if len(x) >= 40 and not x[:1].islower()]
    if not sentences:
        return ""
    scored = [(sum(t in x.lower() for t in terms), -i, x) for i, x in enumerate(sentences)]
    top = [x for score, _, x in sorted(scored, reverse=True)[:3] if score > 0] or sentences[:2]
    out, size = [], 0
    for sentence in sorted(top, key=sentences.index):        # keep original order
        if size + len(sentence) > limit and out:
            break
        out.append(sentence)
        size += len(sentence) + 1
    return " ".join(out)[:limit + 40]


# Domains of a project's OWN sites ("official"). Extend with a JSON file: OFFICIAL_SOURCES_FILE=mapping.json
# {"topic keyword": ["domain.com", "github.com/org"]}
OFFICIAL_DOMAINS = {
    "langgraph": ["langchain.com", "langchain-ai.github.io", "github.com/langchain-ai"],
    "langchain": ["langchain.com", "langchain-ai.github.io", "github.com/langchain-ai"],
    "langsmith": ["langchain.com", "smith.langchain.com"],
}
try:
    import json as _json
    if os.getenv("OFFICIAL_SOURCES_FILE"):
        with open(os.environ["OFFICIAL_SOURCES_FILE"], encoding="utf-8") as _f:
            for _k, _v in _json.load(_f).items():
                OFFICIAL_DOMAINS.setdefault(_k.lower(), []).extend(_v)
except Exception:
    pass

_EXCLUDED_HOSTS = ("scribd.com", "linkedin.com", "pinterest.com", "facebook.com", "twitter.com",
                   "x.com", "tiktok.com", "quora.com")          # never used as evidence
_LOW_QUALITY_HOSTS = ("youtube.com", "youtu.be", "reddit.com") + _EXCLUDED_HOSTS


def _host_path(url: str):
    try:
        parsed = urlparse(url or "")
        return (parsed.hostname or "").lower(), (parsed.path or "").lower()
    except Exception:
        return "", ""


def _on(host: str, *domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_official(url: str, terms: List[str] = None) -> bool:
    """The project's own site: listed domains, or (for unknown projects) the topic word in the host name."""
    host, path = _host_path(url)
    keys = [t for t in (terms or []) if t in OFFICIAL_DOMAINS] or ([] if terms else list(OFFICIAL_DOMAINS))
    for key in keys:
        for entry in OFFICIAL_DOMAINS[key]:
            if "/" in entry:
                if (host + path).startswith(entry):
                    return True
            elif _on(host, entry):
                return True
    return any(t in host for t in (terms or []) if len(t) >= 4)


def source_tier(url: str, terms: List[str] = None) -> int:
    """0 = the project's own docs / primary & reference sources, 1 = normal (incl. third-party docs and blogs),
    2 = video / forum, 3 = excluded (Scribd, LinkedIn, ... never used as evidence)."""
    host, path = _host_path(url)
    if _on(host, *_EXCLUDED_HOSTS):
        return 3
    if _on(host, "youtube.com", "youtu.be", "reddit.com"):
        return 2
    if (is_official(url, terms) or _on(host, "arxiv.org", "wikipedia.org", "python.org")
            or host.endswith((".gov", ".edu"))):
        return 0
    return 1


def source_type(url: str, terms: List[str] = None) -> str:
    """What kind of source is this? The writer must attribute claims from single-author, single-project
    or third-party sources instead of presenting them as general facts."""
    host, path = _host_path(url)
    if _on(host, *_EXCLUDED_HOSTS):
        return "low-quality page (excluded)"
    if _on(host, "arxiv.org"):
        return "research paper"
    if _on(host, "wikipedia.org"):
        return "encyclopedia"
    if _on(host, "github.com", "gitlab.com", "bitbucket.org"):
        return "official code repository" if is_official(url, terms) else "code repository (ONE project)"
    if _on(host, *_LOW_QUALITY_HOSTS):
        return "video / social post"
    if _on(host, "medium.com", "substack.com", "dev.to", "hashnode.dev", "towardsdatascience.com") \
            or host.startswith("blog.") or "/blog" in path:
        return "blog post (one author's view)"
    if is_official(url, terms) or _on(host, "python.org") or host.endswith((".gov", ".edu")):
        return "official documentation"
    if host.startswith(("docs.", "developer.", "developers.")) or path.startswith("/docs") \
            or "/docs/" in path or _on(host, "readthedocs.io", "github.io"):
        return "third-party / vendor documentation (NOT the project's own)"
    return "web article"


def evidence_from_hits(search_results: List[dict], limit: int = 8, terms: List[str] = None) -> List[dict]:
    """Build evidence items straight from raw search hits (no LLM) - used as a fallback
    when the LLM evidence step returns nothing although the searches found results.
    Snippets are cleaned, junk-only hits dropped, official sources first."""
    hits = []
    for entry in search_results or []:
        for hit in entry.get("result", []) or []:
            if isinstance(hit, dict) and hit.get("url"):
                hits.append((entry.get("source", ""), hit))
    hits = [pair for pair in hits if source_tier(pair[1]["url"], terms) < 3]
    hits.sort(key=lambda pair: (source_tier(pair[1]["url"], terms), -(pair[1].get("score") or 0)))
    items, seen = [], set()
    terms = terms or []
    for source, hit in hits:
        if hit["url"] in seen:
            continue
        snippet = clean_snippet(hit.get("page_excerpt"), 400)
        if len(snippet) < 60:
            snippet = clean_snippet(hit.get("content"), 300)
        if len(snippet) < 40:
            continue                                   # nothing useful left after cleaning
        seen.add(hit["url"])
        items.append({
            "title": " ".join((hit.get("title") or hit["url"]).split()),
            "url": hit["url"],
            "published_at": hit.get("published_date"),
            "snippets": snippet,
            "sources": source,
        })
    # within the same quality tier, sources whose TEXT (not just title) mentions the topic come first
    if terms:
        items.sort(key=lambda i: (source_tier(i["url"], terms),
                                  0 if any(t in i["snippets"].lower() for t in terms) else 1))
        # a source whose TEXT never mentions the topic (a generic paper abstract) is noise: drop it when enough remain
        relevant = [i for i in items if any(t in i["snippets"].lower() for t in terms)]
        if len(relevant) >= 3:
            items = relevant
    return items[:limit]


RESEARCH_TOOLS = [tavily_search, wikipedia_search, arxiv_search,
                  fetch_webpage, get_current_datetime]
