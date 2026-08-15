import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Lightweight processor for parsing and chunking Q&A text.
    Deletes the old PDF, image, table, and vision code paths.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}")

    def chunk_text(self, text: str, max_chars: int = 1500) -> List[str]:
        """
        Splits text by paragraph boundaries. If a paragraph exceeds max_chars,
        splits it into smaller segments.
        """
        if not text:
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If a single paragraph is too large, split it by sentence or simple length
            if len(para) > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph by punctuation or simple character limits
                sub_paras = re.split(r'(?<=[.!?])\s+', para)
                for sub in sub_paras:
                    if len(sub) > max_chars:
                        # Hard character split if still too big
                        for i in range(0, len(sub), max_chars):
                            chunks.append(sub[i:i + max_chars])
                    else:
                        chunks.append(sub)
            else:
                if current_length + len(para) > max_chars:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    current_chunk.append(para)
                    current_length += len(para) + 2  # account for \n\n

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Carries the tail of each chunk into the start of the next one, so a
        boundary landing mid-argument doesn't fully sever context (e.g.
        "here's why X is wrong" in one chunk, "therefore do Y instead" in
        the next, with nothing connecting them). ~29% of answers in this
        corpus produce more than one chunk, so this isn't a rare edge case.

        Kept separate from chunk_text() so the plain, non-overlapping chunks
        stay available too -- process_answer() uses those (via "display_text")
        for citation snippets shown to users, so a snippet doesn't start
        mid-sentence with leftover text carried over from the previous chunk.
        Only the embedded/generation text gets the overlap.
        """
        if not self.chunk_overlap or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append(prev_tail + "\n\n" + chunks[i])
        return overlapped

    def process_answer(self, answer_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks answer text and returns list of chunks ready to be embedded.

        Each result carries two versions of the text:
          - "text": question_title + overlap-tail + chunk. This is what gets
            embedded and what the LLM sees as context. Many StackExchange
            answers are direct, deictic replies ("Yes.", "That is correct.",
            "Such X is...") that only make sense next to their question --
            embedded alone they carry almost no semantic signal matching what
            a user actually searches for. Confirmed as the main cause of poor
            recall on straightforward queries (see evaluation/results.md).
          - "display_text": the plain chunk, no title, no overlap tail. Used
            for citation snippets shown to users, so they don't see the
            question title repeated or a snippet that starts mid-sentence
            with text carried over from the previous chunk.
        """
        if not answer_text:
            return []

        question_title = metadata.get("question_title", "")
        raw_chunks = self.chunk_text(answer_text)
        embed_chunks = self._add_overlap(raw_chunks)

        result = []
        for i, (raw, embed) in enumerate(zip(raw_chunks, embed_chunks)):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            embedded_text = f"{question_title}\n\n{embed}" if question_title else embed
            result.append({
                "text": embedded_text,
                "display_text": raw,
                "metadata": chunk_metadata
            })
        return result