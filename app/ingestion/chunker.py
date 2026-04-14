"""
MODULE: app/ingestion/chunker.py

Adaptive text chunker that is aware of:
- Sentence boundaries (via NLTK)
- Code blocks (fenced ``` blocks kept intact)
- Paragraph boundaries (double newlines)

Replaces the naive character-slicer to improve retrieval quality.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Lazy-load NLTK sentence tokenizer
_sent_tokenize = None

def _get_sent_tokenize():
    global _sent_tokenize
    if _sent_tokenize is None:
        try:
            from nltk.tokenize import sent_tokenize
            # Trigger a test call to ensure punkt data is available
            sent_tokenize("Hello. World.")
            _sent_tokenize = sent_tokenize
        except LookupError:
            import nltk
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            _sent_tokenize = sent_tokenize
    return _sent_tokenize


def _extract_code_blocks(text: str) -> tuple[list[str], str]:
    """
    Pull fenced code blocks out of the text, replacing them with placeholders.
    Returns (list_of_code_blocks, text_with_placeholders).
    """
    code_blocks = []
    pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    def replacer(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    cleaned = pattern.sub(replacer, text)
    return code_blocks, cleaned


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK, falling back to regex."""
    tokenize = _get_sent_tokenize()
    try:
        return tokenize(text)
    except Exception:
        # Regex fallback
        return re.split(r'(?<=[.!?])\s+', text)


def _restore_code_blocks(chunk: str, code_blocks: list[str]) -> str:
    """Replace code block placeholders back with original code."""
    for i, block in enumerate(code_blocks):
        chunk = chunk.replace(f"__CODE_BLOCK_{i}__", block)
    return chunk


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Split text into overlapping chunks that respect sentence and paragraph boundaries.

    Args:
        text: The full document text.
        chunk_size: Target maximum characters per chunk.
        overlap_sentences: Number of trailing sentences to repeat in the next chunk.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Step 1: Extract code blocks so they don't get split
    code_blocks, cleaned_text = _extract_code_blocks(text)

    # Step 2: Split into paragraphs first (double newline boundaries)
    paragraphs = re.split(r'\n\s*\n', cleaned_text)

    # Step 3: Break each paragraph into sentences
    all_sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if this paragraph is actually a code block placeholder
        if re.match(r'^__CODE_BLOCK_\d+__$', para):
            all_sentences.append(para)
            continue

        sentences = _split_into_sentences(para)
        all_sentences.extend(sentences)

    if not all_sentences:
        return []

    # Step 4: Group sentences into chunks respecting chunk_size
    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in all_sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, it becomes its own chunk
        if sentence_len > chunk_size:
            # Flush current buffer first
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_length = 0
            chunks.append(sentence)
            continue

        # Would adding this sentence exceed the target?
        if current_length + sentence_len + 1 > chunk_size and current_chunk_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Carry over the last N sentences as overlap
            overlap = current_chunk_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk_sentences = list(overlap)
            current_length = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) - 1 if current_chunk_sentences else 0

        current_chunk_sentences.append(sentence)
        current_length += sentence_len + 1  # +1 for the space join

    # Flush remaining
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # Step 5: Restore code blocks in all chunks
    chunks = [_restore_code_blocks(c, code_blocks) for c in chunks]

    # Step 6: Clean up whitespace
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(f"Chunked text into {len(chunks)} adaptive chunks (target={chunk_size} chars)")
    return chunks
