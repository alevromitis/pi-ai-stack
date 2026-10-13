"""Offline machine translation with Argos Translate's models.

Each Argos package is a CTranslate2 model plus a tokenizer for one language
pair: SentencePiece, or in some older packages (es, cs, da, fi, hu, id, pl,
ro, uk) Moses tokenization and BPE, as argostranslate does it, X->en or en->X; other pairs pivot through English. Only
ctranslate2 and sentencepiece are used, not the argostranslate package:
that pulls in stanza and torch (~1 GB) just to split sentences, done here
with a regex instead. Language detection is py3langid, limited to the
installed languages.

Models live in MODELS_DIR/<from>_<to>/ (fetch_models.py writes them); a few
recently used pairs stay loaded, the rest are read from disk on demand.
"""

import collections
import ctypes
import os
import re
import threading

import ctranslate2
import py3langid
import sacremoses
import sentencepiece
from subword_nmt.apply_bpe import BPE

NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sv": "Swedish", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese",
    "zh": "Chinese",
}
# What a model may write instead of a code.
ALIASES = {name.lower(): code for code, name in NAMES.items()} | {
    "iw": "he", "greek": "el", "farsi": "fa", "mandarin": "zh", "zh-cn": "zh", "pt-br": "pt", "pt-pt": "pt",
}

# Sentence ends: Latin/Greek/Cyrillic punctuation before a space, or CJK and
# Devanagari full stops anywhere. Greek's question mark is ";".
_SENTENCE_END = re.compile(r"(?<=[.!?;:…])\s+|(?<=[。！？；।])")


_LIBC = ctypes.CDLL("libc.so.6")


class TranslateError(Exception):
    pass


class _SentencePiece:
    def __init__(self, path: str):
        self.sp = sentencepiece.SentencePieceProcessor(model_file=path)

    def encode(self, sentence: str) -> list[str]:
        return self.sp.encode(sentence, out_type=str)

    def decode(self, tokens: list[str]) -> str:
        # As argostranslate: the model's pieces don't always round-trip
        # through sp.decode, which leaves "▁" in the text.
        return "".join(tokens).replace("▁", " ").strip()


class _MosesBPE:
    def __init__(self, path: str, src: str, dst: str):
        self.normalizer = sacremoses.MosesPunctNormalizer(src)
        self.tokenizer = sacremoses.MosesTokenizer(src)
        self.detokenizer = sacremoses.MosesDetokenizer(dst)
        with open(path, encoding="utf-8") as f:
            self.bpe = BPE(f)

    def encode(self, sentence: str) -> list[str]:
        tokens = self.tokenizer.tokenize(self.normalizer.normalize(sentence), escape=False)
        return self.bpe.segment_tokens(tokens)

    def decode(self, tokens: list[str]) -> str:
        return self.detokenizer.detokenize(tokens).replace("@@ ", "")


class _Pair:
    def __init__(self, path: str, src: str, dst: str, threads: int):
        self.translator = ctranslate2.Translator(
            os.path.join(path, "model"), device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=threads,
        )
        sp = os.path.join(path, "sentencepiece.model")
        self.tok = _SentencePiece(sp) if os.path.isfile(sp) else _MosesBPE(os.path.join(path, "bpe.model"), src, dst)

    def translate(self, sentences: list[str]) -> list[str]:
        results = self.translator.translate_batch(
            [self.tok.encode(s) for s in sentences], beam_size=4, num_hypotheses=1,
            replace_unknowns=True, max_decoding_length=256, max_batch_size=16,
        )
        return [self.tok.decode(r.hypotheses[0]) for r in results]


class Translator:
    def __init__(self, models_dir: str, keep_loaded: int = 2, threads: int = 2):
        self.models_dir = models_dir
        self.keep_loaded = keep_loaded
        self.threads = threads
        self.pairs = {
            tuple(name.split("_", 1)) for name in os.listdir(models_dir)
            if "_" in name and os.path.isfile(os.path.join(models_dir, name, "model", "model.bin"))
        }
        if not self.pairs:
            raise SystemExit(f"no models in {models_dir}; run fetch_models.py")
        self.languages = sorted({c for p in self.pairs for c in p})
        self._loaded: collections.OrderedDict[tuple[str, str], _Pair] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._identifier = py3langid.langid.LanguageIdentifier.from_model_file(
            py3langid.langid.MODEL_FILE, norm_probs=True)
        self._identifier.set_languages([c for c in self.languages if c in self._identifier.labels])

    def code(self, language: str) -> str:
        lang = language.strip().lower()
        lang = ALIASES.get(lang, lang)
        lang = re.split(r"[-_]", lang)[0]  # "de-AT" -> "de"
        if lang not in self.languages:
            raise TranslateError(f"unsupported language {language!r}; supported: {', '.join(self.languages)}")
        return lang

    def detect(self, text: str) -> tuple[str, float]:
        lang, prob = self._identifier.classify(text)
        return lang, float(prob)

    def _pair(self, src: str, dst: str) -> _Pair:
        key = (src, dst)
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key]
        if key not in self.pairs:
            raise TranslateError(f"no {src}->{dst} model")
        pair = _Pair(os.path.join(self.models_dir, f"{src}_{dst}"), src, dst, self.threads)
        self._loaded[key] = pair
        while len(self._loaded) > self.keep_loaded:
            _, old = self._loaded.popitem(last=False)
            old.translator.unload_model()
            # glibc keeps freed model memory in its arenas otherwise: RSS
            # climbed to ~740 MB over 12 pairs with only 2 kept loaded.
            _LIBC.malloc_trim(0)
        return pair

    def _batch(self, src: str, dst: str, pieces: list[str]) -> list[str]:
        """Each piece translated, sentence by sentence, in one batch."""
        pair = self._pair(src, dst)
        split = [[s for s in _SENTENCE_END.split(p.strip()) if s.strip()] for p in pieces]
        flat = iter(pair.translate([s for ss in split for s in ss]))
        return [" ".join(next(flat) for _ in ss) for ss in split]

    def _run(self, src: str, dst: str, pieces: list[str]) -> list[str]:
        if src == "en" or dst == "en":
            return self._batch(src, dst, pieces)
        return self._batch("en", dst, self._batch(src, "en", pieces))

    def translate(self, text: str, target: str = "en", source: str = "",
                  min_confidence: float = 0.0, only_from: str = "") -> dict:
        """Translates the prose of a Markdown text; code, links, URLs and
        Sources lines stay as they are. Without `source` the language is
        detected; below `min_confidence`, or when it isn't `only_from`, the
        text comes back untranslated. Returns {source, target, confidence,
        translated, text}."""
        if not text.strip():
            raise TranslateError("empty text")
        dst = self.code(target or "en")
        parts = _parts(text)
        prose = " ".join(p for kind, p in parts if kind == "tr")
        if source.strip():
            src, confidence = self.code(source), 1.0
        else:
            src, confidence = self.detect(prose or text)
        result = {"source": src, "target": dst, "confidence": round(confidence, 3), "translated": False, "text": text}
        if (src == dst or not prose.strip() or confidence < min_confidence
                or only_from and src != self.code(only_from)):
            return result
        with self._lock:
            done = iter(self._run(src, dst, [p for kind, p in parts if kind == "tr"]))
        out = []
        for kind, p in parts:
            if kind == "keep":
                out.append(p)
            else:
                # Keep the piece's surrounding spaces: "see [link](url) here".
                lead, trail = p[:len(p) - len(p.lstrip())], p[len(p.rstrip()):]
                out.append(lead + next(done) + trail)
        return result | {"translated": True, "text": "".join(out)}


# Kept as they are: fenced code blocks; Sources lines and lines of nothing
# but links (citations.py writes them); list, heading and quote markers;
# links, inline code, URLs, maths and table bars inside a line. The
# translator would garble them ("[Παρίσι](https://el.wikipedia...").
_FENCE = re.compile(r"^[ \t]*```.*?^[ \t]*```[^\n]*$", re.S | re.M)
_KEEP_LINE = re.compile(r"^\s*(?:[*_]*Sources?[*_]*:.*|(?:\[[^\]\n]+\]\([^)\n]+\)[\s,;.]*)+)$", re.I)
_PREFIX = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)]|#{1,6}|>)\s+)*")
_PROTECT = re.compile(r"\[[^\]\n]+\]\((?:[^\s()]|\([^\s()]*\))+\)|`[^`\n]+`|https?://\S+|\$[^$\n]+\$|\|")
# Bold and italics markers: dropped from translated prose, which would
# move or split them.
_EMPHASIS = re.compile(r"\*\*|__")


def _parts(text: str) -> list[tuple[str, str]]:
    """The text as ("keep", s) and ("tr", s) pieces, in order."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for block in [*_FENCE.finditer(text), None]:
        prose = text[pos:block.start() if block else len(text)]
        for line in re.split(r"(?<=\n)", prose):
            body = line.rstrip("\n")
            if not body.strip() or _KEEP_LINE.match(body):
                parts.append(("keep", line))
                continue
            prefix = _PREFIX.match(body).group(0)
            parts.append(("keep", prefix))
            at = len(prefix)
            for kept in _PROTECT.finditer(body, at):
                parts.append(("tr", _EMPHASIS.sub("", body[at:kept.start()])))
                parts.append(("keep", kept.group(0)))
                at = kept.end()
            parts.append(("tr", _EMPHASIS.sub("", body[at:])))
            parts.append(("keep", line[len(body):]))
        if block:
            parts.append(("keep", block.group(0)))
            pos = block.end()
    # Pieces with no letters (" ", ", ", "2.") need no model.
    return [("keep" if kind == "tr" and not re.search(r"[^\W\d_]", p) else kind, p) for kind, p in parts if p]
