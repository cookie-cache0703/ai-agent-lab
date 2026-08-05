"""Tool: produces a short extractive summary of a block of text.

Deterministic and fully offline (no LLM call from inside a tool — that would
risk unbounded recursive cost/latency). Picks the highest word-frequency
sentences, a classic simple extractive-summarization technique.
"""

import re
from collections import Counter

from pydantic import BaseModel

from tools.base import Tool, tool_error

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "these",
    "this", "those", "to", "was", "were", "which", "with",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_MAX_SUMMARY_SENTENCES = 3


class SummarizeTextArgs(BaseModel):
    text: str


def _sentence_words(sentence: str) -> list[str]:
    return [w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS]


def _select_top_sentences(sentences: list[str]) -> list[str]:
    word_freq = Counter(word for sentence in sentences for word in _sentence_words(sentence))

    def score(sentence: str) -> float:
        words = _sentence_words(sentence)
        return sum(word_freq[w] for w in words) / len(words) if words else 0.0

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    top_indices = sorted(ranked[:_MAX_SUMMARY_SENTENCES])
    return [sentences[i] for i in top_indices]


def _summarize_text(args: SummarizeTextArgs) -> dict:
    text = args.text.strip()
    if not text:
        return tool_error("empty_text", "text must not be empty.")

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return tool_error("empty_text", "text must not be empty.")

    if len(sentences) <= _MAX_SUMMARY_SENTENCES:
        summary = " ".join(sentences)
    else:
        summary = " ".join(_select_top_sentences(sentences))

    return {
        "summary": summary,
        "sentence_count": len(sentences),
        "original_length": len(text),
        "summary_length": len(summary),
    }


summarize_text_tool = Tool(
    name="summarize_text",
    description="Produce a short extractive summary of a block of text.",
    args_model=SummarizeTextArgs,
    handler=_summarize_text,
)
