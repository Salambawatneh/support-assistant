"""Bilingual (Arabic/English) retrieval-augmented support assistant core.

Pipeline: load knowledge-base articles -> chunk -> normalize (Arabic-aware)
-> TF-IDF index -> retrieve top-k chunks -> classify the request -> compose
an answer with citations. Standard library only.

The generation step is pluggable: by default answers are extractive (the best
matching passages, quoted with their source). Set a `generator` callable on
`SupportAssistant` to pass the retrieved context to an LLM instead.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------- Arabic NLP
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)

# Light prefixes/suffixes (conservative: keeps stems of 3+ letters).
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف")
_AR_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "هم", "ه", "ة", "ي")

_STOPWORDS = {
    # English
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "is", "are", "how",
    "do", "i", "my", "can", "for", "it", "with", "what", "me", "you", "be",
    # Arabic
    "في", "من", "على", "الى", "إلى", "عن", "هل", "كيف", "ما", "ماذا", "هذا",
    "هذه", "انا", "أنا", "لا", "مع", "او", "أو", "ان", "أن", "كان", "لم",
}


def normalize_arabic(text: str) -> str:
    """Unify Arabic letter variants and strip diacritics/tatweel."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return text.translate(_AR_DIGITS)


def _stem_ar(token: str) -> str:
    for p in _AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    for s in _AR_SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str) -> list[str]:
    text = normalize_arabic(text).lower()
    tokens = []
    for raw in _TOKEN.findall(text):
        if raw in _STOPWORDS:
            continue
        tokens.append(_stem_ar(raw) if re.search(r"[؀-ۿ]", raw) else raw)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def detect_language(text: str) -> str:
    ar = len(re.findall(r"[؀-ۿ]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return "ar" if ar >= en else "en"


# ------------------------------------------------------------------ Indexing
@dataclass
class Chunk:
    source: str
    category: str
    title: str
    text: str


def load_chunks(kb_dir: Path) -> list[Chunk]:
    """Each .md file: first line `category: X`, then `# Title`, then body.
    The body is split into chunks at blank lines."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        category, title, body = "general", path.stem, []
        for line in lines:
            if line.lower().startswith("category:"):
                category = line.split(":", 1)[1].strip().lower()
            elif line.startswith("# ") and title == path.stem:
                title = line[2:].strip()
            else:
                body.append(line)
        for block in re.split(r"\n\s*\n", "\n".join(body)):
            block = block.strip()
            if block:
                chunks.append(Chunk(path.name, category, title, block))
    return chunks


class TfidfIndex:
    def __init__(self, docs: list[str]):
        self.tokens = [tokenize(d) for d in docs]
        n = len(docs)
        df = Counter(t for toks in self.tokens for t in set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vec(toks) for toks in self.tokens]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q = self._vec([t for t in tokenize(query) if t in self.idf])
        scored = []
        for i, vec in enumerate(self.vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((i, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


# ----------------------------------------------------------------- Assistant
MIN_SCORE = 0.12  # below this the assistant declines instead of guessing

_MESSAGES = {
    "en": {
        "no_answer": "I could not find this in the knowledge base. "
                     "Please open a ticket with the IT service desk.",
        "lead": "Based on the knowledge base ({category}):",
    },
    "ar": {
        "no_answer": "لم أجد إجابة لهذا السؤال في قاعدة المعرفة. "
                     "يرجى فتح تذكرة لدى مكتب الدعم الفني.",
        "lead": "بناءً على قاعدة المعرفة ({category}):",
    },
}


class SupportAssistant:
    def __init__(self, kb_dir: Path,
                 generator: Optional[Callable[[str, list[Chunk]], str]] = None):
        self.chunks = load_chunks(kb_dir)
        if not self.chunks:
            raise ValueError(f"No knowledge-base articles found in {kb_dir}")
        self.index = TfidfIndex([f"{c.title} {c.text}" for c in self.chunks])
        self.categories = sorted({c.category for c in self.chunks})
        self.generator = generator

    def classify(self, text: str) -> dict:
        """Score each category by the retrieval mass of its chunks."""
        totals: Counter = Counter()
        for i, score in self.index.search(text, k=5):
            totals[self.chunks[i].category] += score
        if not totals:
            return {"category": "unknown", "confidence": 0.0}
        cat, top = totals.most_common(1)[0]
        return {"category": cat, "confidence": round(top / sum(totals.values()), 2)}

    def ask(self, question: str, k: int = 2) -> dict:
        lang = detect_language(question)
        hits = [(self.chunks[i], s) for i, s in self.index.search(question, k)]
        category = self.classify(question)
        if not hits or hits[0][1] < MIN_SCORE:
            return {"language": lang, "category": category, "answer":
                    _MESSAGES[lang]["no_answer"], "sources": [], "answered": False}
        # Keep only passages close to the best match (drops weak extras).
        hits = [(c, s) for c, s in hits if s >= 0.5 * hits[0][1]]
        context = [c for c, _ in hits]
        answer, generated_by = None, "extractive"
        if self.generator:
            try:  # any LLM failure falls back to the extractive answer
                answer = self.generator(question, context) or None
                generated_by = "llm" if answer else generated_by
            except Exception:
                answer = None
        if not answer:
            lead = _MESSAGES[lang]["lead"].format(category=hits[0][0].category)
            answer = lead + "\n\n" + "\n\n".join(c.text for c in context)
        return {
            "language": lang,
            "category": category,
            "answer": answer,
            "generated_by": generated_by,
            "answered": True,
            "sources": [{"file": c.source, "title": c.title, "score": round(s, 2)}
                        for c, s in hits],
        }
