import re
from collections import Counter


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "their", "them", "they",
    "he", "she", "his", "her", "you", "your", "we", "our", "i", "me", "my",
    "about", "into", "over", "after", "before", "during", "up", "down", "out",
    "off", "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "too", "very", "can",
    "will", "just", "don", "should", "now", "bbc",
    "says", "said", "has", "have", "had", "new", "first", "who"
}


def generate_summary(text: str, max_sentences: int = 2) -> str:
    """Return text as summary."""

    if not isinstance(text, str) or not text.strip():
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:max_sentences]).strip()


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top frequent non-stopword terms."""

    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)

    filtered = []
    for w in words:
        w = w.strip("-'")
        if w and w not in STOPWORDS and len(w) > 2:
            filtered.append(w)

    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def extract_entities(text: str, top_n: int = 5) -> list[str]:
    """Heuristic entity extraction using capitalized words/phrases."""

    if not isinstance(text, str) or not text.strip():
        return []

    matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)

    cleaned = []
    for m in matches:
        entity = m.strip()
        COMMON_NON_ENTITIES = {"Within", "After", "Before", "During", "When", "While"}

        if entity in COMMON_NON_ENTITIES:
            continue

        entity = re.sub(r"^(The|In|On|At|For|From|To|Of)\s+", "", entity)

        if not entity or len(entity) <= 2:
            continue

        if entity.lower() in STOPWORDS:
            continue

        cleaned.append(entity)

    counts = Counter(cleaned)
    return [entity for entity, _ in counts.most_common(top_n)]
