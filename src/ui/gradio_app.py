"""Gradio web interface for Project Intelligence Hub."""

from __future__ import annotations

from typing import Any, List

import gradio as gr

from src.config.settings import Settings, Neo4jConfig, TogetherAIConfig
from src.models.state import AppState
from src.services.builder import GraphRAGBuilder
from src.services.answerer import QueryAnswerer
from src.services.neo4j_service import Neo4jService, Neo4jConnectionError


class GradioApp:
    """Gradio controller for ingestion and query-time interactions."""

    TITLE = "Project Intelligence Hub"
    DESCRIPTION = """
# Project Intelligence Hub

Transform unstructured PDF reports into a queryable knowledge graph.

1. **Ingest** — Upload documents to extract entities and relationships
2. **Index** — Build vector embeddings and graph structure
3. **Query** — Retrieve answers via hybrid graph + semantic search
"""

    GRAPH_EXPLORER_QUERIES = {
        "node_labels": """
            CALL db.labels() YIELD label
            CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt }
            RETURN label, cnt ORDER BY cnt DESC
        """,
        "relationship_types": """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL { WITH relationshipType MATCH ()-[r]->() WHERE type(r) = relationshipType RETURN count(r) AS cnt }
            RETURN relationshipType, cnt ORDER BY cnt DESC
        """,
        "sample_projects": """
            MATCH (p:Project)
            OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
            RETURN p.name AS project, b.amount AS budget, b.currency AS currency,
                   l.city AS city, l.country AS country
            LIMIT 10
        """,
    }

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.answerer = QueryAnswerer()
        self._validate_settings()

    def _validate_settings(self) -> None:
        issues = []
        if not self.settings.together_ai.api_key:
            issues.append("TOGETHER_API_KEY not set in .env")
        if not self.settings.neo4j.uri:
            issues.append("NEO4J_URI not set in .env")
        if not self.settings.neo4j.password:
            issues.append("NEO4J_PASSWORD not set in .env")

        if issues:
            print("Configuration warnings:")
            for issue in issues:
                print(f"  - {issue}")

    def _ingest_action(self, pdf_files: List[Any], clear_db: str):
        clear_db_bool = clear_db == "Yes"

        if not pdf_files:
            yield "No documents provided. Upload at least one PDF.", gr.update(value=0, visible=True), None
            return

        if not self.settings.together_ai.api_key:
            yield "Missing API credentials: TOGETHER_API_KEY", gr.update(value=0, visible=True), None
            return

        if not self.settings.neo4j.uri or not self.settings.neo4j.password:
            yield "Missing database credentials: NEO4J_URI or NEO4J_PASSWORD", gr.update(value=0, visible=True), None
            return

        together_config = TogetherAIConfig(
            api_key=self.settings.together_ai.api_key,
            chat_model=self.settings.together_ai.chat_model,
            embedding_model=self.settings.together_ai.embedding_model,
        )

        neo4j_config = Neo4jConfig(
            uri=self.settings.neo4j.uri,
            username=self.settings.neo4j.username,
            password=self.settings.neo4j.password,
            database=self.settings.neo4j.database,
        )

        try:
            builder = GraphRAGBuilder(together_config=together_config)

            final_state = None
            for status, progress, state in builder.ingest_with_progress(
                pdf_files=pdf_files,
                neo4j_config=neo4j_config,
                clear_db=clear_db_bool,
                skip_llm_extraction=True,
            ):
                yield status, gr.update(value=progress, visible=True), state
                if state is not None:
                    final_state = state

            if final_state:
                yield "Pipeline complete. Ready for queries.", gr.update(value=1.0, visible=False), final_state

        except ValueError as e:
            yield f"Configuration error: {e}", gr.update(value=0, visible=True), None
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"Pipeline failed: {e}", gr.update(value=0, visible=True), None

    def _clear_action(self) -> str:
        if not self.settings.neo4j.uri or not self.settings.neo4j.password:
            return "Database credentials not configured."

        try:
            with Neo4jService(
                uri=self.settings.neo4j.uri,
                user=self.settings.neo4j.username,
                password=self.settings.neo4j.password,
                database=self.settings.neo4j.database,
            ) as neo4j:
                neo4j.clear()
                return "Graph database cleared. All nodes and relationships removed."
        except Neo4jConnectionError as e:
            return f"Connection error: {e}"
        except Exception as e:
            return f"Operation failed: {e}"

    def _ask_action(self, question: str, state: AppState) -> str:
        return self.answerer.answer(question, state)

    def _explore_graph_action(self) -> str:
        if not self.settings.neo4j.uri or not self.settings.neo4j.password:
            return "Database credentials not configured."

        try:
            with Neo4jService(
                uri=self.settings.neo4j.uri,
                user=self.settings.neo4j.username,
                password=self.settings.neo4j.password,
                database=self.settings.neo4j.database,
            ) as neo4j:
                output = []

                # Node counts by label
                output.append("### Node Distribution\n")
                output.append("| Label | Count |")
                output.append("|-------|-------|")
                try:
                    results = neo4j.query(self.GRAPH_EXPLORER_QUERIES["node_labels"])
                    for row in results:
                        output.append(f"| {row['label']} | {row['cnt']:,} |")
                except Exception:
                    output.append("| (unable to fetch) | - |")

                # Relationship counts
                output.append("\n### Relationship Distribution\n")
                output.append("| Type | Count |")
                output.append("|------|-------|")
                try:
                    results = neo4j.query(self.GRAPH_EXPLORER_QUERIES["relationship_types"])
                    for row in results:
                        output.append(f"| {row['relationshipType']} | {row['cnt']:,} |")
                except Exception:
                    output.append("| (unable to fetch) | - |")

                # Sample projects
                output.append("\n### Sample Projects\n")
                output.append("| Project | Budget | Location |")
                output.append("|---------|--------|----------|")
                try:
                    results = neo4j.query(self.GRAPH_EXPLORER_QUERIES["sample_projects"])
                    if not results:
                        output.append("| (no projects found) | - | - |")
                    for row in results:
                        name = row.get('project') or '-'
                        budget = f"{row.get('budget') or '-'} {row.get('currency') or ''}".strip()
                        location = f"{row.get('city') or ''}, {row.get('country') or ''}".strip(", ")
                        output.append(f"| {name} | {budget} | {location or '-'} |")
                except Exception:
                    output.append("| (unable to fetch) | - | - |")

                return "\n".join(output)

        except Neo4jConnectionError as e:
            return f"Connection error: {e}"
        except Exception as e:
            return f"Failed to fetch graph data: {e}"

    def build(self) -> gr.Blocks:
        with gr.Blocks(title=self.TITLE) as demo:
            gr.Markdown(self.DESCRIPTION)

            state = gr.State(value=None)

            with gr.Group():
                pdfs = gr.File(
                    label="Document Source",
                    file_types=[".pdf"],
                    file_count="multiple",
                )

                with gr.Row():
                    clear_toggle = gr.Radio(
                        label="Reset graph before ingestion",
                        choices=["Yes", "No"],
                        value="Yes",
                        scale=1,
                    )

                with gr.Row():
                    ingest_btn = gr.Button("Run Ingestion Pipeline", variant="primary", scale=2)
                    clear_btn = gr.Button("Reset Graph", variant="secondary", scale=1)

                progress_bar = gr.Slider(
                    label="Progress",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                    visible=False,
                )

                ingest_status = gr.Markdown()

            gr.Markdown("---")

            with gr.Group():
                gr.Markdown("### Query Interface")
                question = gr.Textbox(
                    label="Natural Language Query",
                    placeholder="e.g., Compare budget allocations and milestone timelines across projects",
                    lines=2,
                )
                ask_btn = gr.Button("Execute Query", variant="primary")
                answer = gr.Markdown(label="Response")

            with gr.Accordion("Graph Explorer", open=False):
                gr.Markdown("View database contents without direct access to credentials.")
                explore_btn = gr.Button("Load Graph Statistics", variant="secondary")
                graph_stats = gr.Markdown()

            with gr.Accordion("System Configuration", open=False):
                gr.Markdown(self._get_config_status())

            ingest_btn.click(
                fn=self._ingest_action,
                inputs=[pdfs, clear_toggle],
                outputs=[ingest_status, progress_bar, state],
            )

            clear_btn.click(
                fn=self._clear_action,
                inputs=[],
                outputs=[ingest_status],
            )

            ask_btn.click(
                fn=self._ask_action,
                inputs=[question, state],
                outputs=[answer],
            )

            explore_btn.click(
                fn=self._explore_graph_action,
                inputs=[],
                outputs=[graph_stats],
            )

        return demo

    def _get_config_status(self) -> str:
        def status(value: str) -> str:
            return "Connected" if value else "Not configured"

        return f"""
| Component | Status |
|-----------|--------|
| LLM Provider (Together AI) | {status(self.settings.together_ai.api_key)} |
| Graph Database (Neo4j) | {status(self.settings.neo4j.uri)} |
"""

    def launch(self, **kwargs) -> None:
        demo = self.build()
        demo.launch(
            server_name=kwargs.get("server_name", self.settings.app.host),
            server_port=kwargs.get("server_port", self.settings.app.port),
            theme=gr.themes.Soft(),
            **{k: v for k, v in kwargs.items() if k not in ("server_name", "server_port")},
        )
