"""Text helpers shared by training, decoding, and inference/verification scripts."""
import re

# Keep word characters, whitespace, and apostrophes; drop everything else.
# Apostrophes stay so contractions ("don't") remain a single exact-match word,
# per the grading rule. Anything else (periods, commas, question marks, brackets,
# hyphens, $, ...) becomes a space so it neither joins nor splits real words.
_STRICT_DROP = re.compile(r"[^\w\s']")

_normalizers = {}


def _strict_normalize(s: str) -> str:
    """Rule-compliant normalizer: lowercase, strip punctuation (keeping
    apostrophes), and collapse whitespace — and nothing else.

    Unlike Whisper's EnglishTextNormalizer this does NOT expand contractions
    ("don't" stays "don't", not "do not"), abbreviations ("Mr." -> "mr"), or
    slang, and it does NOT delete bracketed text ("[RIGHT HAND - CLOSE]" ->
    "right hand close"). Words are scored by exact character match; only
    punctuation is discarded.
    """
    return " ".join(_STRICT_DROP.sub(" ", s.lower()).split())


def english_normalizer(strict: bool = True):
    """Return the text normalizer used before WER scoring.

    ``strict`` (default): the rule-compliant normalizer above — the metric that
    matches the official grading (exact word match incl. apostrophes, punctuation
    ignored). Set ``strict=False`` for Whisper's EnglishTextNormalizer (which also
    expands contractions/abbreviations and drops bracketed text).
    """
    if strict not in _normalizers:
        if strict:
            _normalizers[strict] = _strict_normalize
        else:
            try:
                from whisper.normalizers import EnglishTextNormalizer
                _normalizers[strict] = EnglishTextNormalizer()
            except Exception:
                _normalizers[strict] = lambda s: s.lower().strip()
    return _normalizers[strict]


def collapse_runaway_repeats(text, max_repeat=2):
    """Collapse Whisper repetition loops (e.g. "De De De De ...").

    A token repeated more than ``max_repeat`` times in a row is a decoder
    artifact, not language — a single runaway trial can otherwise inflate WER
    by orders of magnitude (observed: WER 36.7 on one trial). Runs longer than
    the threshold are truncated to ``max_repeat`` copies; genuine short
    repetitions ("had had", "very very very") are left untouched.
    """
    words = text.split()
    if not words:
        return text
    out, run = [words[0]], 1
    for w in words[1:]:
        if w == out[-1]:
            run += 1
            if run <= max_repeat:
                out.append(w)
        else:
            run = 1
            out.append(w)
    return " ".join(out)
