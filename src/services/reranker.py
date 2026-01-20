"""Cross-encoder reranker for document retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

from langchain.schema import Document

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model at import time
_cross_encoder = None
_cross_encoder_model_name = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy load the cross-encoder model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        CrossEncoder instance
    """
    global _cross_encoder, _cross_encoder_model_name

    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {model_name}")
            _cross_encoder = CrossEncoder(model_name, max_length=512)
            _cross_encoder_model_name = model_name
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            return None

    return _cross_encoder


class FastCrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Runs locally and is faster than LLM-based reranking.
    """

    MODEL_OPTIONS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "tiny": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    }

    def __init__(
        self,
        model_name: str = "fast",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: One of "fast", "balanced", "tiny", or a HuggingFace model ID
            max_length: Maximum sequence length for encoding
            batch_size: Batch size for scoring (higher = faster but more memory)
        """
        # Resolve model name alias
        self.model_name = self.MODEL_OPTIONS.get(model_name, model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self) -> bool:
        """Ensure model is loaded.

        Returns:
            True if model is available, False otherwise
        """
        if self._model is None:
            self._model = _get_cross_encoder(self.model_name)
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Document]:
        """Rerank documents by relevance to query.

        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked documents (most relevant first)
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return documents

        if not self._ensure_model():
            logger.warning("Cross-encoder not available, returning original order")
            return documents[:top_k]

        try:
            # Prepare query-document pairs
            pairs = [
                (query, self._get_text(doc)[:self.max_length])
                for doc in documents
            ]

            # Score all pairs (batched for efficiency)
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # Sort by score descending
            scored_docs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            return [doc for doc, _ in scored_docs[:top_k]]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original order")
            return documents[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Tuple[Document, float]]:
        """Rerank documents and return with scores.

        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return [(doc, 1.0) for doc in documents]

        if not self._ensure_model():
            return [(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents[:top_k])]

        try:
            pairs = [
                (query, self._get_text(doc)[:self.max_length])
                for doc in documents
            ]

            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            scored_docs = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return [(doc, 1.0 - i * 0.1) for i, doc in enumerate(documents[:top_k])]

    def _get_text(self, doc: Document) -> str:
        """Extract text content from document.

        Args:
            doc: LangChain Document

        Returns:
            Text content
        """
        if hasattr(doc, 'page_content'):
            return doc.page_content
        return str(doc)


class NoOpReranker:
    """No-op reranker that returns documents in original order.

    Use this as a fallback when cross-encoder is not available.
    """

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Document]:
        """Return documents without reranking."""
        return documents[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 6,
    ) -> List[Tuple[Document, float]]:
        """Return documents with dummy scores."""
        return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(documents[:top_k])]


def get_reranker(
    model_name: str = "fast",
    fallback_to_noop: bool = True,
) -> FastCrossEncoderReranker:
    """Factory function to get a reranker instance.

    Args:
        model_name: Model name or alias
        fallback_to_noop: If True, return NoOpReranker when cross-encoder fails

    Returns:
        Reranker instance
    """
    try:
        reranker = FastCrossEncoderReranker(model_name)
        # Test model loading
        if reranker._ensure_model():
            return reranker
    except Exception as e:
        logger.warning(f"Failed to create cross-encoder reranker: {e}")

    if fallback_to_noop:
        logger.info("Using no-op reranker as fallback")
        return NoOpReranker()

    raise RuntimeError("Cross-encoder reranker not available")
