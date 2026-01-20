"""Smart chunking for semi-structured project reports."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.schema import Document


class SemanticChunker:
    """Section-aware chunking that respects document structure."""

    SECTION_PATTERNS = [
        r"^(?:Project\s+)?(?:ID|Name|Summary|Overview)",
        r"^(?:Budget|TIV|Investment|Cost)",
        r"^(?:Schedule|Timeline|Milestones?)",
        r"^(?:Location|Site|Address)",
        r"^(?:Status|Progress|Update)",
        r"^(?:Details?|Description|Scope)",
        r"^(?:Challenge|Risk|Issue|Constraint)",
        r"^(?:Engineering|Construction|Procurement)",
        r"^(?:Environmental|Regulatory|Permit)",
    ]

    DENSE_INDICATORS = [
        r'\$[\d,]+',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\b[A-Z]{2,}\b',
        r'\d+\s*(?:MW|GW|tons?|MT|units?|km|miles?)',
    ]

    def __init__(
        self,
        max_chunk_size: int = 1200,
        min_chunk_size: int = 200,
        overlap_sentences: int = 2,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
        self._section_pattern = re.compile(
            "|".join(f"({p})" for p in self.SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )

    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify section boundaries in document."""
        sections: List[Dict[str, Any]] = []
        matches = list(self._section_pattern.finditer(text))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append({
                "header": match.group().strip(),
                "start": start,
                "end": end,
                "content": text[start:end].strip()
            })

        if not sections:
            sections.append({
                "header": "Document",
                "start": 0,
                "end": len(text),
                "content": text.strip()
            })

        return sections

    def _calculate_density(self, text: str) -> float:
        """Calculate information density of text (matches per 100 chars)."""
        total_matches = sum(len(re.findall(p, text)) for p in self.DENSE_INDICATORS)
        return (total_matches / max(len(text), 1)) * 100

    def _optimal_chunk_size(self, text: str) -> int:
        """Determine optimal chunk size based on content density."""
        density = self._calculate_density(text)
        if density > 5:
            return 600
        elif density > 2:
            return 900
        return 1200

    def _split_section(
        self,
        section: Dict[str, Any],
        source: str,
        chunk_size: Optional[int] = None
    ) -> List[Document]:
        """Split a section into appropriately sized chunks."""
        content = section["content"]
        header = section["header"]
        effective_chunk_size = chunk_size or self.max_chunk_size

        if len(content) <= effective_chunk_size:
            return [Document(
                page_content=f"[{header}] {content}",
                metadata={
                    "source": source,
                    "section": header,
                    "chunk_size": len(content),
                    "density": self._calculate_density(content),
                }
            )]

        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks: List[Document] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > effective_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=f"[{header}] {chunk_text}",
                    metadata={
                        "source": source,
                        "section": header,
                        "chunk_size": len(chunk_text),
                        "density": self._calculate_density(chunk_text),
                    }
                ))
                current_chunk = current_chunk[-self.overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(Document(
                    page_content=f"[{header}] {chunk_text}",
                    metadata={
                        "source": source,
                        "section": header,
                        "chunk_size": len(chunk_text),
                        "density": self._calculate_density(chunk_text),
                    }
                ))

        return chunks

    def chunk_document(self, text: str, source: str, adaptive: bool = True) -> List[Document]:
        """Chunk document respecting section boundaries."""
        sections = self._detect_sections(text)
        all_chunks: List[Document] = []

        for section in sections:
            chunk_size = self._optimal_chunk_size(section["content"]) if adaptive else self.max_chunk_size
            chunks = self._split_section(section, source, chunk_size)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_pages(self, pages: List[Document], adaptive: bool = True) -> List[Document]:
        """Chunk a list of page Documents."""
        if not pages:
            return []

        source = pages[0].metadata.get("source", "document.pdf")
        full_text = ""
        page_boundaries: List[int] = []

        for page in pages:
            page_boundaries.append(len(full_text))
            full_text += page.page_content + "\n\n"

        chunks = self.chunk_document(full_text, source, adaptive)

        for chunk in chunks:
            chunk_start = full_text.find(
                chunk.page_content.replace(f"[{chunk.metadata.get('section', '')}] ", "")[:50]
            )
            if chunk_start >= 0:
                page_num = 1
                for i, boundary in enumerate(page_boundaries):
                    if chunk_start >= boundary:
                        page_num = i + 1
                chunk.metadata["page"] = page_num

        return chunks


_default_chunker: Optional[SemanticChunker] = None


def get_chunker() -> SemanticChunker:
    """Get the default chunker instance (singleton)."""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = SemanticChunker()
    return _default_chunker
