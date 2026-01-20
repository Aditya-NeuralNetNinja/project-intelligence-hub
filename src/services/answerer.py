"""Query answering service with hybrid strategy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from langchain.schema import Document

from src.config import get_logger, trace_flow, log_step
from src.models.state import AppState
from src.services.retriever import OptimizedRetriever
from src.services.cache import AnswerCache, get_answer_cache
from src.services.cypher_templates import (
    CypherTemplateRouter,
    TemplateResultFormatter,
    QueryIntent,
)

# Module logger
logger = get_logger(__name__)


class QueryAnswerer:
    """Answers user questions using an optimized hybrid strategy.

    Strategy:
        1) Template-first routing: Pattern matching classifies intent and
           executes pre-validated Cypher templates for most queries.
           This is deterministic, fast, and reliable.

        2) For general queries: GraphRAG with optimized retrieval:
           - Pattern-based query expansion (no LLM)
           - Cross-encoder reranking (faster than LLM)
           - Single LLM call for synthesis only
    """

    # Default retrieval settings
    DEFAULT_K = 6

    # Optimized synthesis prompt (simpler, more focused)
    SYNTHESIS_PROMPT = """You are an expert analyst for industrial project reports.

## Question
{question}

## Retrieved Document Excerpts
{context}

## Graph Database Context
{graph_context}

## Instructions
1. Answer directly and concisely based on the evidence
2. If information is incomplete, acknowledge what's missing
3. For comparison questions, structure answer by project
4. Use citations like [1], [2] to reference sources
5. For challenges/risks, consider: cancellation reasons, delays, funding issues, permitting

Answer:""".strip()

    def __init__(
        self,
        k: int = DEFAULT_K,
        use_optimized_retrieval: bool = True,
        use_caching: bool = True,
        cache_ttl: float = 3600,
        use_reranking: bool = True,
    ) -> None:
        """Initialize query answerer.

        Args:
            k: Number of chunks to retrieve for similarity search.
            use_optimized_retrieval: If True, uses fast pattern-based expansion
                and cross-encoder reranking. If False, uses original LLM-based.
            use_caching: If True, caches answers for repeated queries.
            cache_ttl: Cache time-to-live.
            use_reranking: If True, uses cross-encoder reranking.
        """
        self.k = k
        self.use_optimized_retrieval = use_optimized_retrieval
        self.use_caching = use_caching
        self.use_reranking = use_reranking
        self._retriever: Optional[OptimizedRetriever] = None
        self._cache: Optional[AnswerCache] = None

        # Initialize template router for fast intent classification
        self._template_router = CypherTemplateRouter()

        if use_caching:
            self._cache = get_answer_cache(default_ttl=cache_ttl)

    def _format_citations(self, docs: List[Document]) -> str:
        """Format unique citations from retrieved chunk documents.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted citation string.
        """
        seen: Set[Tuple[str, Optional[int]]] = set()
        lines: List[str] = []

        for doc in docs:
            src = doc.metadata.get("source", "")
            page = doc.metadata.get("page", None)
            key = (src, page)

            if key in seen:
                continue
            seen.add(key)

            if page is not None:
                lines.append(f"- {src} p.{page}")
            else:
                lines.append(f"- {src}")

        return "\n".join(lines)

    def _format_budget_value(
        self,
        budget: Optional[Any],
        currency: Optional[str]
    ) -> str:
        """Format budget value for display.

        Args:
            budget: Budget amount (may be None or numeric).
            currency: Currency code.

        Returns:
            Formatted budget string.
        """
        if isinstance(budget, (int, float)) and currency:
            return f"{budget:,.0f} {currency}"
        elif budget:
            return str(budget)
        return "—"

    def _format_location(self, row: Dict[str, Any]) -> str:
        """Format location components into a string.

        Args:
            row: Query result row with location fields.

        Returns:
            Formatted location string.
        """
        loc_parts = [
            x for x in [
                row.get("address"),
                row.get("city"),
                row.get("state"),
                row.get("postal"),
                row.get("country"),
            ] if x
        ]
        return ", ".join(loc_parts) if loc_parts else "—"

    def _budget_location(self, graph: Any) -> str:
        """Deterministic answer for budget allocation and location.

        Args:
            graph: Neo4jGraph instance.

        Returns:
            Formatted budget and location answer.
        """
        rows = graph.query(self.CYPHER_BUDGET_LOCATION)

        if not rows:
            return "No structured budget/location data found in the graph yet."

        out = ["**Budget allocation (TIV) and location**"]
        for row in rows:
            budget_str = self._format_budget_value(
                row.get("budget"),
                row.get("currency"),
            )
            loc = self._format_location(row)
            out.append(f"- **{row.get('project')}**: {budget_str}; {loc}")

        return "\n".join(out)

    def _timelines(self, graph: Any) -> str:
        """Deterministic timeline comparison using extracted milestones.

        Args:
            graph: Neo4jGraph instance.

        Returns:
            Formatted timeline answer.
        """
        rows = graph.query(self.CYPHER_TIMELINES)
        logger.info(f"Timeline query returned {len(rows) if rows else 0} rows")

        if not rows:
            return "No structured timeline data found in the graph yet."

        out = ["**Timelines (milestones extracted from Schedule)**"]
        for row in rows:
            project_name = row.get('project') or 'Unknown Project'
            out.append(f"\n### {project_name}")
            milestones = row.get("milestones") or []
            logger.info(f"Project '{project_name}': {len(milestones)} milestones raw")

            # Filter out null milestones (from OPTIONAL MATCH returning nulls)
            valid_milestones = [m for m in milestones if m and m.get("name")]
            logger.info(f"Project '{project_name}': {len(valid_milestones)} valid milestones")

            if not valid_milestones:
                out.append("- No milestones extracted")
            else:
                for m in valid_milestones[:14]:  # Limit display
                    dt = (m.get("dateText") or "").strip()
                    nm = (m.get("name") or "Milestone").strip()
                    if dt:
                        out.append(f"- {nm}: {dt}")
                    else:
                        sent = m.get('sentence') or ''
                        out.append(f"- {nm}: {sent[:100]}")

        result = "\n".join(out)
        logger.info(f"Timeline result: {len(result)} chars")
        return result

    def _challenges(self, graph: Any) -> str:
        """Deterministic challenges listing from structured Challenge nodes.

        Args:
            graph: Neo4jGraph instance.

        Returns:
            Formatted challenges answer.
        """
        rows = graph.query(self.CYPHER_CHALLENGES)

        if not rows:
            return "No structured challenges found yet."

        out = [
            "**Potential challenges / constraints "
            "(from Status reason + Details + schedule heuristics)**"
        ]
        for row in rows:
            out.append(f"\n### {row['project']}")
            challenges = [x for x in (row.get("challenges") or []) if x]

            if not challenges:
                out.append("- —")
            else:
                for ch in challenges[:14]:  # Limit display
                    out.append(f"- {ch}")

        return "\n".join(out)

    def _get_graph_context(self, question: str, graph: Any) -> str:
        """Get relevant graph context without LLM Cypher generation.

        Uses simple pattern matching to find related entities.

        Args:
            question: User question
            graph: Neo4j graph instance

        Returns:
            Formatted graph context string
        """
        import re

        # Extract potential project names from question
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)

        if not potential_names:
            return ""

        context_parts = []

        for name in potential_names[:2]:
            try:
                results = graph.query("""
                    MATCH (p:Project)
                    WHERE toLower(p.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                    OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                    RETURN p.name AS project,
                           p.status AS status,
                           b.amount AS budget,
                           b.currency AS currency,
                           l.city AS city,
                           l.country AS country
                    LIMIT 3
                """, {"name": name.lower()})

                for r in results:
                    parts = [f"**{r['project']}**"]
                    if r.get('status'):
                        parts.append(f"Status: {r['status']}")
                    if r.get('budget'):
                        parts.append(f"Budget: {r['budget']:,.0f} {r.get('currency', '')}")
                    if r.get('city'):
                        parts.append(f"Location: {r['city']}, {r.get('country', '')}")
                    context_parts.append(" | ".join(parts))

            except Exception:
                pass

        return "\n".join(context_parts) if context_parts else ""

    def _get_retriever(self, state: AppState) -> OptimizedRetriever:
        """Get or create the optimized retriever.

        Args:
            state: Application state with vector store.

        Returns:
            OptimizedRetriever instance (fast pattern-based + cross-encoder).
        """
        if self._retriever is None:
            self._retriever = OptimizedRetriever(
                vector_store=state.vector,
                k_initial=self.k * 2,  # Retrieve more initially for reranking
                k_final=self.k,
                use_expansion=True,
                use_reranking=self.use_reranking,
                use_cache=True,
            )
        return self._retriever

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string with source attribution.
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            section = doc.metadata.get('section', '')

            header = f"[{i}] Source: {source}, Page {page}"
            if section:
                header += f", Section: {section}"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def _graphrag_answer(
        self,
        question: str,
        state: AppState,
    ) -> str:
        """Generate answer using optimized GraphRAG approach.

        Optimized flow:
        1. Retrieve with optimized retriever (pattern expansion + cross-encoder)
        2. Get graph context (no LLM Cypher generation)
        3. Single LLM call for synthesis

        Args:
            question: User question.
            state: Application state.

        Returns:
            Synthesized answer with citations.
        """
        with log_step(logger, "GraphRAG answer generation"):
            # Retrieve relevant chunks with optimized retriever
            with log_step(logger, "Retrieve relevant chunks"):
                if self.use_optimized_retrieval:
                    logger.substep("Using optimized retrieval (pattern expansion + cross-encoder)")
                    retriever = self._get_retriever(state)
                    docs = retriever.retrieve(question)
                else:
                    logger.substep("Using simple similarity search")
                    docs = state.vector.similarity_search(question, k=self.k)
                logger.info(f"Retrieved {len(docs)} chunks")

            # Get graph context (fast, no LLM)
            with log_step(logger, "Get graph context"):
                graph = state.get_graph()
                graph_context = self._get_graph_context(question, graph)
                if graph_context:
                    logger.substep(f"Found graph context")
                else:
                    logger.substep("No direct graph context found")

            # Format context
            context = self._format_context(docs)

            # Single LLM call for synthesis
            with log_step(logger, "Synthesize answer"):
                logger.substep("Invoking LLM for synthesis")
                synthesis_prompt = self.SYNTHESIS_PROMPT.format(
                    question=question,
                    context=context,
                    graph_context=graph_context if graph_context else "(No structured data found)",
                )

                resp = state.llm.invoke(synthesis_prompt)
                answer = getattr(resp, "content", str(resp))

            # Cache the answer
            if self._cache and self.use_caching:
                logger.substep("Caching answer")
                self._cache.set_answer(
                    query=question,
                    answer=answer,
                    documents=docs,
                    cypher_result=graph_context,
                )

        return answer

    def clear_cache(self) -> int:
        """Clear the answer cache.

        Returns:
            Number of cached entries cleared.
        """
        if self._cache:
            return self._cache.invalidate_all()
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics.
        """
        if self._cache:
            return self._cache.get_stats()
        return {"caching_enabled": False}

    @trace_flow("Query Processing")
    def answer(self, question: str, state: AppState) -> str:
        """Answer a user question using optimized hybrid approach.

        Flow:
        1. Check answer cache
        2. Template routing with pattern classification
        3. For structured queries: Execute template + format
        4. For general queries: Vector search + rerank + synthesis

        Args:
            question: Natural language user query.
            state: AppState initialized after successful ingestion.

        Returns:
            Markdown response suitable for display.
        """
        logger.info(f"Processing question: {question[:80]}...")

        if not state or not state.is_ready():
            logger.warning("State not ready - PDFs not ingested")
            return "Please ingest PDFs first."

        # Check cache first
        if self._cache and self.use_caching:
            with log_step(logger, "Check cache"):
                cached = self._cache.get_answer(question)
                if cached:
                    logger.info("Cache hit")
                    return cached.answer

        graph = state.get_graph()

        # Try template routing first (handles 70-80% of queries)
        with log_step(logger, "Template routing"):
            results, intent = self._template_router.route_query(question, graph)

            if intent != QueryIntent.GENERAL and results is not None:
                # Format template results (no LLM needed)
                answer = TemplateResultFormatter.format(results, intent)

                # Cache the answer
                if self._cache and self.use_caching:
                    self._cache.set_answer(
                        query=question,
                        answer=answer,
                        documents=[],
                        cypher_result=str(results[:3]) if results else "",
                    )

                logger.info(f"Template answer (intent: {intent.value})")
                return answer

            logger.info(f"Intent: {intent.value} - using RAG fallback")

        # GraphRAG fallback for general queries
        answer = self._graphrag_answer(question, state)
        logger.info("RAG answer generated")
        return answer
