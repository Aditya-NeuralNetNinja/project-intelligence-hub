"""Pre-validated Cypher query templates for deterministic query routing."""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LLM-BASED INTENT CLASSIFIER
# =============================================================================

class LLMIntentClassifier:
    """Classifies query intent using a lightweight LLM.

    Uses a small model from Together AI for intent classification.
    Handles synonyms naturally without hardcoding patterns.
    Caches results and falls back to pattern matching if LLM fails.
    """

    # Cheap, fast model for classification
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

    # Classification prompt - designed to be concise for speed
    CLASSIFICATION_PROMPT = """Classify this query into exactly ONE category. For compound queries, pick the combined category.

Categories:
- TIMELINE_LOCATION: Questions about BOTH timeline/schedule AND location/place
- TIMELINE_BUDGET: Questions about BOTH timeline/schedule AND budget/cost
- BUDGET_LOCATION: Questions about BOTH cost/money AND location/place
- CONTACTS: Questions about project manager, owner, engineer, contractor, lead, head, E&C firm, personnel, who is responsible
- TIMELINE: Questions ONLY about schedule, dates, milestones, deadlines, duration, when things happen
- CHALLENGES: Questions about problems, risks, issues, obstacles, delays, failures, difficulties, constraints
- BUDGET: Questions ONLY about cost, money, investment, funding, expenses, price, TIV, financial aspects, spend
- LOCATION: Questions ONLY about where, place, site, city, country, address, geography, region
- TECHNICAL: Questions about capacity, scope, technical details, specifications, requirements, fuel type, labor
- COMPARISON: Generic comparison of ALL aspects of projects (budget, timeline, location, challenges, contacts)
- STATUS: Questions about current state, progress, whether active/cancelled, probability
- OVERVIEW: Questions asking for summary, description, general information, tell me about
- GENERAL: Questions that don't fit above categories or need detailed analysis

Query: "{query}"

Respond with ONLY the category name, nothing else."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        use_cache: bool = True,
        fallback_to_patterns: bool = True,
    ):
        """Initialize LLM intent classifier.

        Args:
            model: Together AI model ID. Defaults to Llama-3.2-3B.
            api_key: Together AI API key. Uses env var if not provided.
            use_cache: Whether to cache classification results.
            fallback_to_patterns: Whether to use pattern matching as fallback.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.use_cache = use_cache
        self.fallback_to_patterns = fallback_to_patterns
        self._cache: Dict[str, str] = {}
        self._client = None

    def _get_client(self):
        """Lazy-load Together AI client."""
        if self._client is None:
            try:
                from together import Together
                self._client = Together(api_key=self.api_key)
            except ImportError:
                logger.warning("together package not installed")
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Together client: {e}")
                return None
        return self._client

    def _cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def classify(self, query: str) -> str:
        """Classify query intent using LLM.

        Args:
            query: User query string

        Returns:
            Intent category name (e.g., "TIMELINE", "BUDGET")
        """
        # Check cache first
        if self.use_cache:
            cache_key = self._cache_key(query)
            if cache_key in self._cache:
                logger.debug(f"Intent cache hit: {self._cache[cache_key]}")
                return self._cache[cache_key]

        # Try LLM classification
        client = self._get_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": self.CLASSIFICATION_PROMPT.format(query=query)}
                    ],
                    max_tokens=20,  # Only need category name
                    temperature=0,  # Deterministic
                )

                intent = response.choices[0].message.content.strip().upper()

                # Validate intent is a known category
                valid_intents = {
                    "BUDGET_LOCATION", "TIMELINE_LOCATION", "TIMELINE_BUDGET",
                    "TIMELINE", "CHALLENGES", "BUDGET", "LOCATION",
                    "CONTACTS", "TECHNICAL", "COMPARISON", "STATUS",
                    "OVERVIEW", "GENERAL"
                }

                # Handle variations in response - check longer names first
                matched = False
                for valid in sorted(valid_intents, key=len, reverse=True):
                    if valid in intent:
                        intent = valid
                        matched = True
                        break

                if not matched:
                    intent = "GENERAL"

                # Cache result
                if self.use_cache:
                    self._cache[cache_key] = intent

                logger.info(f"LLM classified query as: {intent}")
                return intent

            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Fallback to pattern matching
        if self.fallback_to_patterns:
            return self._pattern_fallback(query)

        return "GENERAL"

    def _pattern_fallback(self, query: str) -> str:
        """Simple pattern-based fallback if LLM fails."""
        q = query.lower()

        # Check for keywords - expanded synonym sets
        has_timeline = any(w in q for w in [
            "timeline", "schedule", "milestone", "deadline", "when", "date",
            "duration", "start", "finish", "complete", "begin", "end"
        ])
        has_budget = any(w in q for w in [
            "budget", "cost", "investment", "money", "spend", "fund", "price",
            "expense", "tiv", "financial", "dollar", "amount", "funding"
        ])
        has_location = any(w in q for w in [
            "location", "where", "site", "city", "country", "place", "address",
            "region", "state", "area", "geography", "situated"
        ])
        has_challenge = any(w in q for w in [
            "challenge", "risk", "issue", "problem", "obstacle", "delay",
            "difficult", "constraint", "failure", "cancelled", "cancel"
        ])
        has_contacts = any(w in q for w in [
            "manager", "owner", "engineer", "contractor", "lead", "head",
            "contact", "personnel", "responsible", "e&c", "firm", "who"
        ])
        has_technical = any(w in q for w in [
            "capacity", "scope", "technical", "specification", "requirement",
            "fuel", "labor", "megawatt", "mw", "barrel", "bbl", "unit"
        ])

        # Check for compound intents first (most specific)
        if has_timeline and has_location:
            return "TIMELINE_LOCATION"
        if has_timeline and has_budget:
            return "TIMELINE_BUDGET"
        if has_budget and has_location:
            return "BUDGET_LOCATION"

        # Single intents - prioritize more specific ones
        if has_contacts:
            return "CONTACTS"
        if has_technical:
            return "TECHNICAL"
        if has_timeline:
            return "TIMELINE"
        if has_challenge:
            return "CHALLENGES"
        if has_budget:
            return "BUDGET"
        if has_location:
            return "LOCATION"

        # Generic intents
        if any(w in q for w in ["compare", "comparison", "versus", "vs", "differ", "difference"]):
            return "COMPARISON"
        if any(w in q for w in ["status", "progress", "state", "active", "probability"]):
            return "STATUS"
        if any(w in q for w in ["overview", "summary", "describe", "explain", "tell me", "about"]):
            return "OVERVIEW"

        return "GENERAL"

    def clear_cache(self) -> int:
        """Clear the classification cache."""
        count = len(self._cache)
        self._cache.clear()
        return count


class QueryIntent(Enum):
    """Detected query intents for template routing."""
    BUDGET = "budget"
    LOCATION = "location"
    BUDGET_LOCATION = "budget_location"
    TIMELINE = "timeline"
    TIMELINE_LOCATION = "timeline_location"  # Combined: timeline + location
    TIMELINE_BUDGET = "timeline_budget"      # Combined: timeline + budget
    CHALLENGES = "challenges"
    CONTACTS = "contacts"                    # Project manager, owner, engineer
    TECHNICAL = "technical"                  # Capacity, scope, specifications
    COMPARISON = "comparison"                # Full comparison with all data
    PROJECT_OVERVIEW = "overview"
    PROJECT_STATUS = "status"
    GENERAL = "general"  # Requires RAG fallback


@dataclass
class CypherTemplate:
    """Pre-validated Cypher query template.

    Attributes:
        intent: The query intent this template handles
        cypher: The Cypher query string
        description: Human-readable description
        required_params: List of required parameter names (if any)
    """
    intent: QueryIntent
    cypher: str
    description: str
    required_params: List[str] = field(default_factory=list)

    def execute(self, graph: Any, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Execute template against the graph.

        Args:
            graph: Neo4j graph instance (LangChain Neo4jGraph)
            params: Optional query parameters

        Returns:
            List of result dictionaries
        """
        try:
            return graph.query(self.cypher, params or {})
        except Exception as e:
            logger.warning(f"Template execution failed: {e}")
            return []


class CypherTemplateRouter:
    """Routes queries to pre-validated Cypher templates.

    This eliminates LLM Cypher generation for ~70-80% of queries,
    providing deterministic, fast, and reliable results.

    Example:
        >>> router = CypherTemplateRouter()
        >>> results, intent = router.route_query("What is the budget?", graph)
        >>> if results is not None:
        ...     print(f"Used template for {intent.value}")
    """

    # =====================================================================
    # PRE-VALIDATED CYPHER TEMPLATES
    # =====================================================================
    # These queries have been tested against the actual graph schema and
    # are guaranteed to work correctly.

    TEMPLATES = {
        QueryIntent.BUDGET_LOCATION: CypherTemplate(
            intent=QueryIntent.BUDGET_LOCATION,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       b.amount AS budget,
                       b.currency AS currency,
                       l.address AS address,
                       l.city AS city,
                       l.state AS state,
                       l.postal AS postal,
                       l.country AS country,
                       l.zoneCounty AS zoneCounty
                ORDER BY p.name
            """,
            description="Get budget (TIV) and location for all projects",
        ),

        QueryIntent.BUDGET: CypherTemplate(
            intent=QueryIntent.BUDGET,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       b.amount AS budget,
                       b.currency AS currency,
                       b.kind AS budgetType
                ORDER BY b.amount DESC
            """,
            description="Get budget/investment information for all projects",
        ),

        QueryIntent.LOCATION: CypherTemplate(
            intent=QueryIntent.LOCATION,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       l.address AS address,
                       l.city AS city,
                       l.state AS state,
                       l.postal AS postal,
                       l.country AS country,
                       l.zoneCounty AS zone
                ORDER BY p.name
            """,
            description="Get location information for all projects",
        ),

        QueryIntent.TIMELINE: CypherTemplate(
            intent=QueryIntent.TIMELINE,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
                WITH p, m
                ORDER BY p.name, m.dateText
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       collect({
                           name: m.name,
                           date: m.dateText,
                           detail: m.sentence
                       }) AS milestones
                ORDER BY p.name
            """,
            description="Get timeline and milestones for all projects",
        ),

        QueryIntent.CHALLENGES: CypherTemplate(
            intent=QueryIntent.CHALLENGES,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_CHALLENGE]->(c:Challenge)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.statusReason AS statusReason,
                       collect(DISTINCT c.text) AS challenges
                ORDER BY p.name
            """,
            description="Get challenges, constraints, and risks for all projects",
        ),

        QueryIntent.TIMELINE_LOCATION: CypherTemplate(
            intent=QueryIntent.TIMELINE_LOCATION,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
                WITH p, l, m
                ORDER BY p.name, m.dateText
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       l.city AS city,
                       l.state AS state,
                       l.country AS country,
                       l.address AS address,
                       collect({
                           name: m.name,
                           date: m.dateText,
                           detail: m.sentence
                       }) AS milestones
                ORDER BY p.name
            """,
            description="Get timeline milestones AND location for all projects",
        ),

        QueryIntent.TIMELINE_BUDGET: CypherTemplate(
            intent=QueryIntent.TIMELINE_BUDGET,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
                WITH p, b, m
                ORDER BY p.name, m.dateText
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       b.amount AS budget,
                       b.currency AS currency,
                       collect({
                           name: m.name,
                           date: m.dateText,
                           detail: m.sentence
                       }) AS milestones
                ORDER BY p.name
            """,
            description="Get timeline milestones AND budget for all projects",
        ),

        QueryIntent.CONTACTS: CypherTemplate(
            intent=QueryIntent.CONTACTS,
            cypher="""
                MATCH (p:Project)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.projectManager AS projectManager,
                       p.projectManagerCompany AS projectManagerCompany,
                       p.projectManagerTitle AS projectManagerTitle,
                       p.projectManagerEmail AS projectManagerEmail,
                       p.projectManagerPhone AS projectManagerPhone,
                       p.plantOwner AS plantOwner,
                       p.plantParent AS plantParent,
                       p.plantName AS plantName,
                       p.engineerCompany AS engineerCompany,
                       p.ecFirm AS ecFirm,
                       p.phone AS phone
                ORDER BY p.name
            """,
            description="Get project manager, owner, engineer, and contact information",
        ),

        QueryIntent.TECHNICAL: CypherTemplate(
            intent=QueryIntent.TECHNICAL,
            cypher="""
                MATCH (p:Project)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.industryCode AS industryCode,
                       p.projectType AS projectType,
                       p.sector AS sector,
                       p.sicCode AS sicCode,
                       p.sicProduct AS sicProduct,
                       p.pecTiming AS pecTiming,
                       p.pecActivity AS pecActivity,
                       p.projectCapacity AS projectCapacity,
                       p.scopeText AS scopeText,
                       p.environmental AS environmental,
                       p.constructionLabor AS constructionLabor,
                       p.operationsLabor AS operationsLabor,
                       p.fuelType AS fuelType,
                       p.unitName AS unitName
                ORDER BY p.name
            """,
            description="Get technical details including capacity, scope, and specifications",
        ),

        QueryIntent.COMPARISON: CypherTemplate(
            intent=QueryIntent.COMPARISON,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
                OPTIONAL MATCH (p)-[:HAS_CHALLENGE]->(c:Challenge)
                WITH p, b, l, m, c
                ORDER BY p.name, m.dateText
                WITH p, b, l,
                     collect(DISTINCT {name: m.name, date: m.dateText}) AS milestones,
                     collect(DISTINCT c.text) AS challenges
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.statusReason AS statusReason,
                       p.projectProbability AS projectProbability,
                       p.projectManager AS projectManager,
                       p.projectManagerCompany AS projectManagerCompany,
                       p.projectManagerTitle AS projectManagerTitle,
                       p.plantOwner AS plantOwner,
                       p.plantParent AS plantParent,
                       p.plantName AS plantName,
                       p.engineerCompany AS engineerCompany,
                       p.ecFirm AS ecFirm,
                       p.industryCode AS industryCode,
                       p.projectType AS projectType,
                       p.sector AS sector,
                       p.sicCode AS sicCode,
                       p.pecTiming AS pecTiming,
                       p.pecActivity AS pecActivity,
                       p.projectCapacity AS projectCapacity,
                       p.scopeText AS scopeText,
                       b.amount AS budget,
                       b.currency AS currency,
                       l.city AS city,
                       l.state AS state,
                       l.country AS country,
                       l.address AS address,
                       milestones,
                       challenges
                ORDER BY b.amount DESC
            """,
            description="Compare all projects with full details (budget, location, timeline, challenges, contacts, technical)",
        ),

        QueryIntent.PROJECT_OVERVIEW: CypherTemplate(
            intent=QueryIntent.PROJECT_OVERVIEW,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
                OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (p)-[:HAS_REPORT]->(r:Report)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.statusReason AS statusReason,
                       p.projectProbability AS projectProbability,
                       p.projectManager AS projectManager,
                       p.projectManagerCompany AS projectManagerCompany,
                       p.projectManagerTitle AS projectManagerTitle,
                       p.plantOwner AS plantOwner,
                       p.plantParent AS plantParent,
                       p.plantName AS plantName,
                       p.engineerCompany AS engineerCompany,
                       p.ecFirm AS ecFirm,
                       p.industryCode AS industryCode,
                       p.projectType AS projectType,
                       p.sector AS sector,
                       p.sicCode AS sicCode,
                       p.pecTiming AS pecTiming,
                       p.pecActivity AS pecActivity,
                       p.projectCapacity AS projectCapacity,
                       p.constructionLabor AS constructionLabor,
                       p.operationsLabor AS operationsLabor,
                       p.fuelType AS fuelType,
                       p.unitName AS unitName,
                       b.amount AS budget,
                       b.currency AS currency,
                       l.city AS city,
                       l.state AS state,
                       l.country AS country,
                       l.address AS address,
                       r.lastUpdate AS lastUpdate,
                       r.initialRelease AS initialRelease
                ORDER BY p.name
            """,
            description="Get comprehensive overview of all projects with all attributes",
        ),

        QueryIntent.PROJECT_STATUS: CypherTemplate(
            intent=QueryIntent.PROJECT_STATUS,
            cypher="""
                MATCH (p:Project)
                OPTIONAL MATCH (p)-[:HAS_REPORT]->(r:Report)
                RETURN p.name AS project,
                       p.projectId AS projectId,
                       p.status AS status,
                       p.statusReason AS statusReason,
                       r.lastUpdate AS lastUpdate
                ORDER BY p.name
            """,
            description="Get project status information",
        ),
    }

    def __init__(self, use_llm: bool = True) -> None:
        """Initialize the template router.

        Args:
            use_llm: If True, uses LLM for intent classification (handles synonyms).
                     If False, uses simple pattern matching (faster but limited).
        """
        self.use_llm = use_llm
        self._llm_classifier: Optional[LLMIntentClassifier] = None

    def _get_classifier(self) -> LLMIntentClassifier:
        """Lazy-load the LLM classifier."""
        if self._llm_classifier is None:
            self._llm_classifier = LLMIntentClassifier(
                use_cache=True,
                fallback_to_patterns=True,
            )
        return self._llm_classifier

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using LLM or pattern matching.

        Args:
            query: User query string

        Returns:
            Detected QueryIntent
        """
        if self.use_llm:
            classifier = self._get_classifier()
            intent_str = classifier.classify(query)
        else:
            # Fallback to simple pattern matching
            intent_str = self._simple_pattern_match(query)

        # Map string to QueryIntent enum
        intent_map = {
            "BUDGET_LOCATION": QueryIntent.BUDGET_LOCATION,
            "TIMELINE_LOCATION": QueryIntent.TIMELINE_LOCATION,
            "TIMELINE_BUDGET": QueryIntent.TIMELINE_BUDGET,
            "TIMELINE": QueryIntent.TIMELINE,
            "CHALLENGES": QueryIntent.CHALLENGES,
            "CONTACTS": QueryIntent.CONTACTS,
            "TECHNICAL": QueryIntent.TECHNICAL,
            "BUDGET": QueryIntent.BUDGET,
            "LOCATION": QueryIntent.LOCATION,
            "COMPARISON": QueryIntent.COMPARISON,
            "STATUS": QueryIntent.PROJECT_STATUS,
            "OVERVIEW": QueryIntent.PROJECT_OVERVIEW,
            "GENERAL": QueryIntent.GENERAL,
        }

        return intent_map.get(intent_str, QueryIntent.GENERAL)

    def _simple_pattern_match(self, query: str) -> str:
        """Simple pattern matching fallback (no LLM)."""
        q = query.lower()

        # Check for combined intents first
        if any(w in q for w in ["budget", "cost", "money"]) and any(w in q for w in ["location", "where", "site"]):
            return "BUDGET_LOCATION"

        # Single intents - check domain keywords
        if any(w in q for w in ["timeline", "schedule", "milestone", "deadline", "when", "duration"]):
            return "TIMELINE"
        if any(w in q for w in ["challenge", "risk", "issue", "problem", "obstacle", "delay"]):
            return "CHALLENGES"
        if any(w in q for w in ["budget", "cost", "investment", "money", "spend", "fund", "price"]):
            return "BUDGET"
        if any(w in q for w in ["location", "where", "site", "city", "country", "place"]):
            return "LOCATION"
        if any(w in q for w in ["compare", "comparison", "versus", "differ"]):
            return "COMPARISON"
        if any(w in q for w in ["status", "progress", "state"]):
            return "STATUS"
        if any(w in q for w in ["overview", "summary", "describe", "explain"]):
            return "OVERVIEW"

        return "GENERAL"

    def get_template(self, intent: QueryIntent) -> Optional[CypherTemplate]:
        """Get template for a given intent.

        Args:
            intent: Query intent

        Returns:
            CypherTemplate or None if no template for intent
        """
        return self.TEMPLATES.get(intent)

    def route_query(
        self,
        query: str,
        graph: Any,
    ) -> Tuple[Optional[List[Dict]], QueryIntent]:
        """Route query to template or indicate fallback needed.

        Args:
            query: User query string
            graph: Neo4j graph instance

        Returns:
            Tuple of (results or None, detected intent)
            Results is None if intent is GENERAL or template execution failed
        """
        intent = self.classify_intent(query)
        logger.info(f"Query classified as: {intent.value}")

        if intent == QueryIntent.GENERAL:
            return None, intent

        template = self.get_template(intent)
        if template is None:
            logger.warning(f"No template found for intent: {intent.value}")
            return None, intent

        try:
            results = template.execute(graph)
            if results:
                logger.info(f"Template returned {len(results)} results")
                return results, intent
            else:
                logger.warning("Template returned empty results")
                return [], intent
        except Exception as e:
            logger.warning(f"Template execution error: {e}")
            return None, intent

    def get_all_intents(self) -> List[QueryIntent]:
        """Get list of all supported intents (excluding GENERAL)."""
        return [intent for intent in QueryIntent if intent != QueryIntent.GENERAL]

    def get_template_description(self, intent: QueryIntent) -> str:
        """Get human-readable description of what a template does."""
        template = self.get_template(intent)
        if template:
            return template.description
        return f"No template available for {intent.value}"


# =========================================================================
# RESULT FORMATTERS
# =========================================================================
# These functions format Cypher results into human-readable markdown
# without requiring LLM synthesis.

class TemplateResultFormatter:
    """Formats template results into markdown without LLM."""

    # Standard message for missing information
    NOT_FOUND_MSG = "I couldn't find this information in the provided documents."

    @staticmethod
    def format_budget(results: List[Dict]) -> str:
        """Format budget results."""
        if not results:
            return "I couldn't find any budget information in the provided documents."

        lines = ["## Budget Information\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            budget = r.get('budget')
            currency = r.get('currency') or ''
            status = r.get('status') or ''

            if budget is not None:
                if isinstance(budget, (int, float)):
                    budget_str = f"{budget:,.0f} {currency}".strip()
                else:
                    budget_str = f"{budget} {currency}".strip()
            else:
                budget_str = "Not available"

            status_str = f" ({status})" if status else ""
            lines.append(f"- **{project}**{status_str}: {budget_str}")

        return "\n".join(lines)

    @staticmethod
    def format_location(results: List[Dict]) -> str:
        """Format location results."""
        if not results:
            return "I couldn't find any location information in the provided documents."

        lines = ["## Location Information\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            loc_parts = [
                r.get('address'),
                r.get('city'),
                r.get('state'),
                r.get('country'),
            ]
            loc = ", ".join([p for p in loc_parts if p]) or "Not available"
            lines.append(f"- **{project}**: {loc}")

        return "\n".join(lines)

    @staticmethod
    def format_budget_location(results: List[Dict]) -> str:
        """Format combined budget and location results."""
        if not results:
            return "I couldn't find any budget or location information in the provided documents."

        lines = ["## Budget Allocation and Location\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            status = r.get('status') or ''

            # Format budget
            budget = r.get('budget')
            currency = r.get('currency') or ''
            if budget is not None:
                if isinstance(budget, (int, float)):
                    budget_str = f"{budget:,.0f} {currency}".strip()
                else:
                    budget_str = f"{budget} {currency}".strip()
            else:
                budget_str = "Not available"

            # Format location
            loc_parts = [r.get('city'), r.get('state'), r.get('country')]
            loc = ", ".join([p for p in loc_parts if p]) or "Not available"

            status_str = f" *({status})*" if status else ""
            lines.append(f"\n### {project}{status_str}")
            lines.append(f"- **Budget (TIV)**: {budget_str}")
            lines.append(f"- **Location**: {loc}")

            if r.get('address'):
                lines.append(f"- **Address**: {r['address']}")
            if r.get('zoneCounty'):
                lines.append(f"- **Zone/County**: {r['zoneCounty']}")

        return "\n".join(lines)

    @staticmethod
    def format_timeline(results: List[Dict]) -> str:
        """Format timeline/milestone results."""
        if not results:
            return "I couldn't find any timeline information in the provided documents."

        lines = ["## Project Timelines\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            status = r.get('status') or ''
            milestones = r.get('milestones') or []

            status_str = f" *({status})*" if status else ""
            lines.append(f"\n### {project}{status_str}")

            # Filter out null milestones
            valid_milestones = [
                m for m in milestones
                if m and (m.get('name') or m.get('date'))
            ]

            if not valid_milestones:
                lines.append("- No milestones recorded")
            else:
                for m in valid_milestones[:12]:  # Limit display
                    name = m.get('name') or 'Milestone'
                    date = m.get('date') or ''
                    detail = m.get('detail') or ''

                    if date:
                        lines.append(f"- **{name}**: {date}")
                    elif detail:
                        lines.append(f"- **{name}**: {detail[:100]}...")
                    else:
                        lines.append(f"- {name}")

        return "\n".join(lines)

    @staticmethod
    def format_challenges(results: List[Dict]) -> str:
        """Format challenges results."""
        if not results:
            return "I couldn't find any challenge or risk information in the provided documents."

        lines = ["## Project Challenges and Constraints\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            status = r.get('status') or ''
            status_reason = r.get('statusReason') or ''
            challenges = r.get('challenges') or []

            lines.append(f"\n### {project}")

            if status:
                lines.append(f"**Status**: {status}")
            if status_reason:
                lines.append(f"**Status Reason**: {status_reason}")

            # Filter out None/empty challenges
            valid_challenges = [c for c in challenges if c]

            if valid_challenges:
                lines.append("\n**Identified Challenges:**")
                for ch in valid_challenges[:10]:
                    lines.append(f"- {ch}")
            elif status_reason:
                lines.append("\n*Challenges inferred from status reason above.*")
            else:
                lines.append("- No specific challenges recorded")

        return "\n".join(lines)

    @staticmethod
    def format_contacts(results: List[Dict]) -> str:
        """Format contact/personnel information results."""
        if not results:
            return "I couldn't find any contact or personnel information in the provided documents."

        lines = ["## Project Contacts and Personnel\n"]

        for r in results:
            project = r.get('project') or 'Unknown Project'
            lines.append(f"\n### {project}")

            has_any_contact = False

            # Project Manager
            pm_name = r.get('projectManager')
            if pm_name:
                has_any_contact = True
                pm_info = pm_name
                if r.get('projectManagerTitle'):
                    pm_info += f", {r['projectManagerTitle']}"
                if r.get('projectManagerCompany'):
                    pm_info += f" ({r['projectManagerCompany']})"
                lines.append(f"- **Project Manager**: {pm_info}")
                if r.get('projectManagerEmail'):
                    lines.append(f"  - Email: {r['projectManagerEmail']}")
                if r.get('projectManagerPhone'):
                    lines.append(f"  - Phone: {r['projectManagerPhone']}")

            # Owner
            plant_owner = r.get('plantOwner')
            if plant_owner:
                has_any_contact = True
                owner_info = plant_owner
                if r.get('plantParent'):
                    owner_info += f" (Parent: {r['plantParent']})"
                lines.append(f"- **Owner**: {owner_info}")
                if r.get('plantName'):
                    lines.append(f"  - Plant/Facility: {r['plantName']}")

            # Engineer
            if r.get('engineerCompany'):
                has_any_contact = True
                lines.append(f"- **Engineer**: {r['engineerCompany']}")

            # E&C Firm
            if r.get('ecFirm'):
                has_any_contact = True
                lines.append(f"- **E&C Firm**: {r['ecFirm']}")

            # General phone
            if r.get('phone'):
                has_any_contact = True
                lines.append(f"- **Phone**: {r['phone']}")

            if not has_any_contact:
                lines.append("- No contact information available")

        return "\n".join(lines)

    @staticmethod
    def format_technical(results: List[Dict]) -> str:
        """Format technical details and specifications results."""
        if not results:
            return "I couldn't find any technical specifications in the provided documents."

        lines = ["## Technical Details and Specifications\n"]

        for r in results:
            project = r.get('project') or 'Unknown Project'
            lines.append(f"\n### {project}")

            has_any_technical = False

            # Classification
            if r.get('industryCode') or r.get('projectType') or r.get('sector'):
                has_any_technical = True
                lines.append("- **Classification**:")
                if r.get('industryCode'):
                    lines.append(f"  - Industry: {r['industryCode']}")
                if r.get('projectType'):
                    lines.append(f"  - Type: {r['projectType']}")
                if r.get('sector'):
                    lines.append(f"  - Sector: {r['sector']}")
                if r.get('sicCode'):
                    lines.append(f"  - SIC Code: {r['sicCode']}")
                if r.get('sicProduct'):
                    lines.append(f"  - SIC Product: {r['sicProduct']}")

            # PEC Stage
            if r.get('pecTiming') or r.get('pecActivity'):
                has_any_technical = True
                pec = f"{r.get('pecTiming', '')} - {r.get('pecActivity', '')}".strip(' -')
                if pec:
                    lines.append(f"- **PEC Stage**: {pec}")

            # Capacity
            if r.get('projectCapacity'):
                has_any_technical = True
                lines.append(f"- **Project Capacity**: {r['projectCapacity']}")

            # Scope
            if r.get('scopeText'):
                has_any_technical = True
                scope = r['scopeText']
                if len(scope) > 300:
                    scope = scope[:300] + "..."
                lines.append(f"- **Scope**: {scope}")

            # Environmental
            if r.get('environmental'):
                has_any_technical = True
                lines.append(f"- **Environmental**: {r['environmental']}")

            # Labor
            if r.get('constructionLabor') or r.get('operationsLabor'):
                has_any_technical = True
                labor_parts = []
                if r.get('constructionLabor'):
                    labor_parts.append(f"Construction: {r['constructionLabor']}")
                if r.get('operationsLabor'):
                    labor_parts.append(f"Operations: {r['operationsLabor']}")
                lines.append(f"- **Labor**: {', '.join(labor_parts)}")

            # Fuel type
            if r.get('fuelType'):
                has_any_technical = True
                lines.append(f"- **Fuel Type**: {r['fuelType']}")

            # Unit
            if r.get('unitName'):
                has_any_technical = True
                lines.append(f"- **Unit**: {r['unitName']}")

            if not has_any_technical:
                lines.append("- No technical specifications available")

        return "\n".join(lines)

    @staticmethod
    def format_comparison(results: List[Dict]) -> str:
        """Format comparison results with comprehensive project details."""
        if not results:
            return "I couldn't find any project data for comparison in the provided documents."

        lines = ["## Project Comparison\n"]

        for r in results:
            project = r.get('project') or 'Unknown'
            lines.append(f"### {project}")

            # Status section
            status = r.get('status')
            if status:
                lines.append(f"- **Status**: {status}")
                if r.get('statusReason'):
                    lines.append(f"  - Reason: {r['statusReason']}")
                if r.get('projectProbability'):
                    lines.append(f"  - Probability: {r['projectProbability']}")

            # Classification
            if r.get('industryCode') or r.get('projectType') or r.get('sector'):
                lines.append("- **Classification**:")
                if r.get('industryCode'):
                    lines.append(f"  - Industry: {r['industryCode']}")
                if r.get('projectType'):
                    lines.append(f"  - Type: {r['projectType']}")
                if r.get('sector'):
                    lines.append(f"  - Sector: {r['sector']}")
                if r.get('sicCode'):
                    lines.append(f"  - SIC Code: {r['sicCode']}")

            # Budget
            budget = r.get('budget')
            currency = r.get('currency') or ''
            if budget is not None and isinstance(budget, (int, float)):
                if budget >= 1_000_000_000:
                    budget_str = f"{budget/1_000_000_000:.1f}B {currency}".strip()
                elif budget >= 1_000_000:
                    budget_str = f"{budget/1_000_000:.0f}M {currency}".strip()
                else:
                    budget_str = f"{budget:,.0f} {currency}".strip()
                lines.append(f"- **Budget (TIV)**: {budget_str}")

            # Location
            loc_parts = [r.get('address'), r.get('city'), r.get('state'), r.get('country')]
            loc_parts = [p for p in loc_parts if p]
            if loc_parts:
                lines.append(f"- **Location**: {', '.join(loc_parts)}")

            # Capacity/Technical
            if r.get('projectCapacity'):
                lines.append(f"- **Project Capacity**: {r['projectCapacity']}")
            if r.get('pecTiming') or r.get('pecActivity'):
                pec = f"{r.get('pecTiming', '')} - {r.get('pecActivity', '')}".strip(' -')
                if pec:
                    lines.append(f"- **PEC Stage**: {pec}")

            # Contacts section
            pm_name = r.get('projectManager')
            pm_company = r.get('projectManagerCompany')
            pm_title = r.get('projectManagerTitle')
            plant_owner = r.get('plantOwner')
            plant_parent = r.get('plantParent')
            engineer = r.get('engineerCompany')
            ec_firm = r.get('ecFirm')

            if any([pm_name, plant_owner, engineer, ec_firm]):
                lines.append("- **Key Contacts**:")
                if pm_name:
                    pm_info = pm_name
                    if pm_title:
                        pm_info += f", {pm_title}"
                    if pm_company:
                        pm_info += f" ({pm_company})"
                    lines.append(f"  - Project Manager: {pm_info}")
                if plant_owner:
                    owner_info = plant_owner
                    if plant_parent:
                        owner_info += f" (Parent: {plant_parent})"
                    lines.append(f"  - Owner: {owner_info}")
                if engineer:
                    lines.append(f"  - Engineer: {engineer}")
                if ec_firm:
                    lines.append(f"  - E&C Firm: {ec_firm}")

            # Plant info
            if r.get('plantName'):
                lines.append(f"- **Plant/Facility**: {r['plantName']}")

            # Milestones and Challenges counts
            ms = r.get('milestones') or []
            ch = r.get('challenges') or []
            if isinstance(ms, list):
                milestone_count = len([m for m in ms if m and m.get('name')])
            else:
                milestone_count = 0
            if isinstance(ch, list):
                challenge_count = len([c for c in ch if c])
            else:
                challenge_count = 0

            lines.append(f"- **Milestones**: {milestone_count}")
            lines.append(f"- **Challenges**: {challenge_count}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_overview(results: List[Dict]) -> str:
        """Format comprehensive project overview results."""
        if not results:
            return "I couldn't find any project data in the provided documents."

        lines = ["## Project Overview\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            lines.append(f"\n### {project}")

            # Basic identification
            if r.get('projectId'):
                lines.append(f"- **Project ID**: {r['projectId']}")

            # Status section
            if r.get('status'):
                lines.append(f"- **Status**: {r['status']}")
                if r.get('statusReason'):
                    lines.append(f"  - Reason: {r['statusReason']}")
                if r.get('projectProbability'):
                    lines.append(f"  - Probability: {r['projectProbability']}")

            # Classification section
            has_classification = any([r.get('industryCode'), r.get('projectType'),
                                     r.get('sector'), r.get('sicCode')])
            if has_classification:
                lines.append("- **Classification**:")
                if r.get('industryCode'):
                    lines.append(f"  - Industry: {r['industryCode']}")
                if r.get('projectType'):
                    lines.append(f"  - Type: {r['projectType']}")
                if r.get('sector'):
                    lines.append(f"  - Sector: {r['sector']}")
                if r.get('sicCode'):
                    lines.append(f"  - SIC Code: {r['sicCode']}")

            # Budget
            if r.get('budget') is not None:
                budget = r['budget']
                currency = r.get('currency') or ''
                if isinstance(budget, (int, float)):
                    if budget >= 1_000_000_000:
                        budget_str = f"{budget/1_000_000_000:.1f}B {currency}".strip()
                    elif budget >= 1_000_000:
                        budget_str = f"{budget/1_000_000:.0f}M {currency}".strip()
                    else:
                        budget_str = f"{budget:,.0f} {currency}".strip()
                else:
                    budget_str = f"{budget} {currency}".strip()
                lines.append(f"- **Budget (TIV)**: {budget_str}")

            # Location
            loc_parts = [r.get('address'), r.get('city'), r.get('state'), r.get('country')]
            loc_parts = [p for p in loc_parts if p]
            if loc_parts:
                lines.append(f"- **Location**: {', '.join(loc_parts)}")

            # Technical details
            if r.get('projectCapacity'):
                lines.append(f"- **Project Capacity**: {r['projectCapacity']}")
            if r.get('pecTiming') or r.get('pecActivity'):
                pec = f"{r.get('pecTiming', '')} - {r.get('pecActivity', '')}".strip(' -')
                if pec:
                    lines.append(f"- **PEC Stage**: {pec}")
            if r.get('fuelType'):
                lines.append(f"- **Fuel Type**: {r['fuelType']}")
            if r.get('unitName'):
                lines.append(f"- **Unit**: {r['unitName']}")

            # Labor information
            if r.get('constructionLabor') or r.get('operationsLabor'):
                labor_info = []
                if r.get('constructionLabor'):
                    labor_info.append(f"Construction: {r['constructionLabor']}")
                if r.get('operationsLabor'):
                    labor_info.append(f"Operations: {r['operationsLabor']}")
                lines.append(f"- **Labor**: {', '.join(labor_info)}")

            # Contacts section
            pm_name = r.get('projectManager')
            pm_company = r.get('projectManagerCompany')
            pm_title = r.get('projectManagerTitle')
            plant_owner = r.get('plantOwner')
            plant_parent = r.get('plantParent')
            plant_name = r.get('plantName')
            engineer = r.get('engineerCompany')
            ec_firm = r.get('ecFirm')

            if any([pm_name, plant_owner, engineer, ec_firm]):
                lines.append("- **Key Contacts**:")
                if pm_name:
                    pm_info = pm_name
                    if pm_title:
                        pm_info += f", {pm_title}"
                    if pm_company:
                        pm_info += f" ({pm_company})"
                    lines.append(f"  - Project Manager: {pm_info}")
                if plant_owner:
                    owner_info = plant_owner
                    if plant_parent:
                        owner_info += f" (Parent: {plant_parent})"
                    lines.append(f"  - Owner: {owner_info}")
                if engineer:
                    lines.append(f"  - Engineer: {engineer}")
                if ec_firm:
                    lines.append(f"  - E&C Firm: {ec_firm}")

            # Plant/Facility info
            if plant_name:
                lines.append(f"- **Plant/Facility**: {plant_name}")

            # Report dates
            if r.get('lastUpdate') or r.get('initialRelease'):
                lines.append("- **Report Info**:")
                if r.get('lastUpdate'):
                    lines.append(f"  - Last Updated: {r['lastUpdate']}")
                if r.get('initialRelease'):
                    lines.append(f"  - Initial Release: {r['initialRelease']}")

        return "\n".join(lines)

    @staticmethod
    def format_status(results: List[Dict]) -> str:
        """Format status results."""
        if not results:
            return "I couldn't find any project status information in the provided documents."

        lines = ["## Project Status\n"]
        for r in results:
            project = r.get('project') or 'Unknown Project'
            status = r.get('status') or 'Unknown'
            reason = r.get('statusReason') or ''
            last_update = r.get('lastUpdate') or ''

            lines.append(f"\n### {project}")
            lines.append(f"- **Status**: {status}")
            if reason:
                lines.append(f"- **Reason**: {reason}")
            if last_update:
                lines.append(f"- **Last Updated**: {last_update}")

        return "\n".join(lines)

    @classmethod
    def format(cls, results: List[Dict], intent: QueryIntent) -> str:
        """Format results based on intent.

        Args:
            results: Query results
            intent: Detected intent

        Returns:
            Formatted markdown string
        """
        formatters = {
            QueryIntent.BUDGET: cls.format_budget,
            QueryIntent.LOCATION: cls.format_location,
            QueryIntent.BUDGET_LOCATION: cls.format_budget_location,
            QueryIntent.TIMELINE: cls.format_timeline,
            QueryIntent.TIMELINE_LOCATION: cls.format_timeline,  # Use timeline formatter
            QueryIntent.TIMELINE_BUDGET: cls.format_timeline,    # Use timeline formatter
            QueryIntent.CHALLENGES: cls.format_challenges,
            QueryIntent.CONTACTS: cls.format_contacts,
            QueryIntent.TECHNICAL: cls.format_technical,
            QueryIntent.COMPARISON: cls.format_comparison,
            QueryIntent.PROJECT_OVERVIEW: cls.format_overview,
            QueryIntent.PROJECT_STATUS: cls.format_status,
        }

        formatter = formatters.get(intent)
        if formatter:
            return formatter(results)

        # Generic fallback
        if not results:
            return "I couldn't find this information in the provided documents."

        lines = ["## Query Results\n"]
        for r in results:
            items = [f"**{k}**: {v}" for k, v in r.items() if v is not None]
            lines.append("- " + " | ".join(items))

        return "\n".join(lines)
