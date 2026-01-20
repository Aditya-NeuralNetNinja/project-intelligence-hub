"""Application state container for runtime handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.services.neo4j_service import Neo4jService


@dataclass
class AppState:
    """Runtime handles required for query-time execution after ingestion."""

    neo4j: Optional[Any] = None
    vector: Optional[Any] = None
    qa_chain: Optional[Any] = None
    llm: Optional[Any] = None

    def is_ready(self) -> bool:
        return all([self.neo4j, self.vector, self.qa_chain, self.llm])

    def get_graph(self) -> Optional[Any]:
        return self.neo4j.graph if self.neo4j else None

    def close(self) -> None:
        if self.neo4j:
            try:
                self.neo4j.close()
            except Exception:
                pass
