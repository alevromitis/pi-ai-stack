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

    def _run(self, src: str, dst: str, text: str) -> str:
        pair = self._pair(src, dst)
        out = []
        for line in text.split("\n"):
            sentences = [s for s in _SENTENCE_END.split(line.strip()) if s.strip()]
            out.append(" ".join(pair.translate(sentences)) if sentences else "")
        return "\n".join(out)

    def translate(self, text: str, target: str = "en", source: str = "") -> tuple[str, str, str]:
        """Returns (source code, target code, translation)."""
        if not text.strip():
            raise TranslateError("empty text")
        dst = self.code(target or "en")
        src = self.code(source) if source.strip() else self.detect(text)[0]
        if src == dst:
            return src, dst, text
        with self._lock:
            if src == "en" or dst == "en":
                return src, dst, self._run(src, dst, text)
            return src, dst, self._run("en", dst, self._run(src, "en", text))
