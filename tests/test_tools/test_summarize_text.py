import re

from tools.summarize_text_tool import summarize_text_tool

_LONG_TEXT = (
    "S1 The city council approved a new budget for public transit. "
    "S2 The budget increases funding for bus routes across downtown. "
    "S3 A local bakery announced a new seasonal menu for autumn. "
    "S4 Transit officials say the funding will reduce bus wait times significantly. "
    "S5 The mayor praised the transit budget as a major investment in the city. "
    "S6 Weather this week is expected to be mild with occasional rain."
)


def test_summarize_text_returns_short_text_unchanged():
    text = "This is short. Only two sentences."

    result = summarize_text_tool.run({"text": text})

    assert result["summary"] == text
    assert result["sentence_count"] == 2


def test_summarize_text_picks_the_most_relevant_sentences_from_long_text():
    result = summarize_text_tool.run({"text": _LONG_TEXT})

    assert result["sentence_count"] == 6
    # The budget/transit theme repeats across sentences, so it dominates the
    # summary over the one-off bakery and weather sentences.
    markers = re.findall(r"S\d", result["summary"])
    assert markers == ["S1", "S2", "S5"]
    assert result["original_length"] == len(_LONG_TEXT)
    assert result["summary_length"] == len(result["summary"])


def test_summarize_text_preserves_original_sentence_order():
    result = summarize_text_tool.run({"text": _LONG_TEXT})

    marker_numbers = [int(m) for m in re.findall(r"S(\d)", result["summary"])]

    assert marker_numbers == sorted(marker_numbers)


def test_summarize_text_returns_structured_error_on_empty_text():
    result = summarize_text_tool.run({"text": "   "})

    assert result["error"] == "empty_text"
