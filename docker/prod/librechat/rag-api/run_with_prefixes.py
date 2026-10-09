"""Runs rag_api's main.py with EmbeddingGemma 2's task prefixes added.

EmbeddingGemma 2 is trained on prefixed input - a query as
"task: search result | query: <q>", a document as "title: <t> | text: <d>" -
and rag_api sends plain text. This patches langchain's OpenAIEmbeddings (the
class rag_api builds for EMBEDDINGS_PROVIDER=openai) before main.py imports
app.config, which creates the embedder at import time.

Upstream embed_query is embed_documents([text])[0], so patching both naively
would give queries both prefixes; the query methods call the unpatched
document methods instead. Fails at startup, not silently, if the methods or
the prefix variables are missing.

LiteRT-LM's EmbeddingGemma 2 takes at most 1024 tokens per input, and one
longer input fails the whole request (500 "Failed to compute embedding
batch"), which fails the upload. rag_api's 1500-character chunks fit prose,
but not number-dense text: Gemma's tokenizer splits digits, so a chunk that
is a list of years is ~1200 tokens. A failing request is therefore retried
one input at a time, and an input that still fails is cut to 80% until it
embeds. Only its vector comes from the shorter text; the stored chunk, and
what file_search returns, is whole.
"""

import os
import runpy
import sys

import openai
from langchain_openai import OpenAIEmbeddings

QUERY_PREFIX = os.environ["EMBEDDINGS_QUERY_PREFIX"]
DOCUMENT_PREFIX = os.environ["EMBEDDINGS_DOCUMENT_PREFIX"]
# Inputs are not cut below this; a failure that persists there is raised.
MIN_CHARS = 200

_embed_documents = OpenAIEmbeddings.embed_documents
_aembed_documents = OpenAIEmbeddings.aembed_documents


def _log(message):
    print(f"rag-api prefixes: {message}", file=sys.stderr, flush=True)


def _shorter(text):
    """The next, shorter attempt at an input the server rejected, or None."""
    if len(text) <= MIN_CHARS:
        return None
    return text[: max(MIN_CHARS, int(len(text) * 0.8))]


def _logged(vector, text, used):
    if len(used) < len(text):
        _log(f"input over the embedding model's limit; embedded the first "
             f"{len(used)} of {len(text)} characters")
    return vector


def _embed_one(self, prefix, text, **kwargs):
    used = text
    while True:
        try:
            return _logged(_embed_documents(self, [prefix + used], **kwargs)[0], text, used)
        except openai.InternalServerError:
            used = _shorter(used)
            if used is None:
                raise


async def _aembed_one(self, prefix, text, **kwargs):
    used = text
    while True:
        try:
            return _logged((await _aembed_documents(self, [prefix + used], **kwargs))[0], text, used)
        except openai.InternalServerError:
            used = _shorter(used)
            if used is None:
                raise


def _batches(self, texts, chunk_size):
    # Upstream's own batching, so a request that succeeds is unchanged.
    size = chunk_size or self.chunk_size
    return [texts[i:i + size] for i in range(0, len(texts), size)]


def _embed(self, prefix, texts, chunk_size=None, **kwargs):
    vectors = []
    for batch in _batches(self, texts, chunk_size):
        try:
            vectors.extend(_embed_documents(self, [prefix + t for t in batch], chunk_size, **kwargs))
        except openai.InternalServerError:
            vectors.extend(_embed_one(self, prefix, t, **kwargs) for t in batch)
    return vectors


async def _aembed(self, prefix, texts, chunk_size=None, **kwargs):
    vectors = []
    for batch in _batches(self, texts, chunk_size):
        try:
            vectors.extend(await _aembed_documents(self, [prefix + t for t in batch], chunk_size, **kwargs))
        except openai.InternalServerError:
            for t in batch:
                vectors.append(await _aembed_one(self, prefix, t, **kwargs))
    return vectors


def embed_documents(self, texts, chunk_size=None, **kwargs):
    return _embed(self, DOCUMENT_PREFIX, texts, chunk_size, **kwargs)


async def aembed_documents(self, texts, chunk_size=None, **kwargs):
    return await _aembed(self, DOCUMENT_PREFIX, texts, chunk_size, **kwargs)


def embed_query(self, text, **kwargs):
    return _embed(self, QUERY_PREFIX, [text], **kwargs)[0]


async def aembed_query(self, text, **kwargs):
    return (await _aembed(self, QUERY_PREFIX, [text], **kwargs))[0]


OpenAIEmbeddings.embed_documents = embed_documents
OpenAIEmbeddings.aembed_documents = aembed_documents
OpenAIEmbeddings.embed_query = embed_query
OpenAIEmbeddings.aembed_query = aembed_query

if __name__ == "__main__":
    _log(f"query {QUERY_PREFIX!r}, document {DOCUMENT_PREFIX!r}")
    # The image's CMD is `python main.py` from WORKDIR /app.
    sys.path.insert(0, "/app")
    runpy.run_path("/app/main.py", run_name="__main__")
