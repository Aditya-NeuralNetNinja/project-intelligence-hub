"""Optimized retriever with pattern-based expansion and cross-encoder reranking."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import Document

from src.config import get_logger, log_step

logger = get_logger(__name__)


class OptimizedRetriever:
    """Fast retriever without LLM calls for expansion/reranking.

    Uses pattern-based query expansion and cross-encoder reranking
    instead of LLM calls for faster retrieval.
    """

    EXPANSION_PATTERNS = {
        "budget": ["cost", "investment", "TIV", "capex", "funding", "allocation", "financial"],
        "location": ["site", "address", "city", "country", "region", "plant", "facility"],
        "timeline": ["schedule", "milestone", "deadline", "completion", "duration", "phase"],
        "challenge": ["risk", "issue", "constraint", "problem", "delay", "obstacle", "barrier"],
        "project": ["plant", "facility", "refinery", "station", "development"],
        "status": ["progress", "state", "condition", "update"],
    }

    def __init__(
        self,
        vector_store: Any,
        reranker: Optional[Any] = None,
        k_initial: int = 12,
        k_final: int = 6,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.k_initial = k_initial
        self.k_final = k_final
        self.use_expansion = use_expansion
        self.use_reranking = use_reranking
        self.use_cache = use_cache
        self._cache: Dict[str, List[Document]] = {}
        self._reranker = reranker
        self._reranker_loaded = reranker is not None

    def _get_reranker(self) -> Optional[Any]:
        if self._reranker_loaded:
            return self._reranker

        try:
            from src.services.reranker import get_reranker
            self._reranker = get_reranker("fast")
            self._reranker_loaded = True
            logger.info("Loaded cross-encoder reranker")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self._reranker = None
            self._reranker_loaded = True

        return self._reranker

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _expand_query_fast(self, query: str) -> List[str]:
        queries = [query]
        query_lower = query.lower()

        for keyword, expansions in self.EXPANSION_PATTERNS.items():
            if keyword in query_lower:
                for exp in expansions[:2]:
                    if exp.lower() not in query_lower:
                        variation = re.sub(
                            rf'\b{keyword}\b',
                            exp,
                            query,
                            flags=re.IGNORECASE
                        )
                        if variation != query and variation not in queries:
                            queries.append(variation)
                break

        return queries[:3]

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[Document, float]]],
        k: int = 60,
    ) -> List[Document]:
        doc_scores: Dict[str, Dict[str, Any]] = {}

        for results in result_lists:
            for rank, (doc, _) in enumerate(results):
                doc_id = hashlib.md5(doc.page_content[:200].encode()).hexdigest()

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}

                doc_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        sorted_items = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return [item["doc"] for item in sorted_items]

    def retrieve(self, question: str) -> List[Document]:
        with log_step(logger, "Optimized retrieval"):
            if self.use_cache:
                cache_key = self._cache_key(question)
                if cache_key in self._cache:
                    logger.info("Cache hit - returning cached results")
                    return self._cache[cache_key]

            if self.use_expansion:
                queries = self._expand_query_fast(question)
                logger.substep(f"Expanded to {len(queries)} queries")
            else:
                queries = [question]

            all_results: List[List[Tuple[Document, float]]] = []

            for i, query in enumerate(queries):
                try:
                    if hasattr(self.vector_store, 'similarity_search_with_score'):
                        results = self.vector_store.similarity_search_with_score(
                            query, k=self.k_initial
                        )
                    else:
                        docs = self.vector_store.similarity_search(
                            query, k=self.k_initial
                        )
                        results = [(doc, 1.0 - j * 0.01) for j, doc in enumerate(docs)]

                    all_results.append(results)
                except Exception as e:
                    logger.warning(f"Query {i+1} failed: {e}")

            if not all_results:
                logger.warning("No results from any query")
                return []

            if len(all_results) > 1:
                fused_docs = self._reciprocal_rank_fusion(all_results)
            else:
                fused_docs = [doc for doc, _ in all_results[0]]

            fused_docs = fused_docs[:self.k_initial]
            logger.substep(f"Fused to {len(fused_docs)} documents")

            if self.use_reranking and len(fused_docs) > self.k_final:
                reranker = self._get_reranker()
                if reranker:
                    with log_step(logger, "Cross-encoder reranking"):
                        fused_docs = reranker.rerank(question, fused_docs, self.k_final)

            final_docs = fused_docs[:self.k_final]

            if self.use_cache:
                self._cache[cache_key] = final_docs

            logger.info(f"Returning {len(final_docs)} documents")
            return final_docs

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        return {"cached_queries": len(self._cache)}
