"""GraphRAG builder for PDF ingestion."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from src.config import get_logger, trace_flow, log_step

# LangChain imports with compatibility handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Neo4jVector
except ImportError:
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores import Neo4jVector

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_together import ChatTogether, TogetherEmbeddings

from src.config.schema import SchemaPolicy
from src.config.settings import Neo4jConfig, TogetherAIConfig
from src.models.state import AppState
from src.parsers.project_parser import ProjectReportParser
from src.parsers.smart_chunker import SemanticChunker
from src.services.neo4j_service import Neo4jService, Neo4jConnectionError

# Module logger
logger = get_logger(__name__)


class GraphRAGBuilder:
    """Builds and populates Neo4j-backed GraphRAG resources from uploaded PDFs.

    Responsibilities:
        - Configure Together AI chat + embeddings models.
        - Parse PDFs into pages and chunks with provenance metadata.
        - Upsert deterministic structured graph nodes for stable Q/A.
        - Run LLMGraphTransformer for broader entity/relationship extraction.
        - Create/refresh Neo4jVector hybrid indexes.
        - Create GraphCypherQAChain for graph-native Q/A.

    This class is intentionally stateless across runs; it returns AppState
    for query-time usage.

    Attributes:
        llm: Chat model instance.
        embeddings: Embeddings model instance.

    Example:
        >>> builder = GraphRAGBuilder(
        ...     together_config=TogetherAIConfig(api_key="key")
        ... )
        >>> message, state = builder.ingest(pdf_files, neo4j_config)
    """

    # Chunk configuration
    DEFAULT_CHUNK_SIZE = 900
    DEFAULT_CHUNK_OVERLAP = 150

    # Parallel extraction configuration (optimized for speed)
    EXTRACTION_BATCH_SIZE = 8  # Increased from 5
    MAX_EXTRACTION_WORKERS = 5  # Increased from 3

    # Vector index configuration
    INDEX_NAME = "project_chunks_vector"
    KEYWORD_INDEX_NAME = "project_chunks_keyword"
    NODE_LABEL = "Chunk"

    # Enhanced Cypher QA prompt with examples
    CYPHER_PROMPT_TEMPLATE = """You are a Neo4j Cypher expert. Generate a Cypher query to answer the question.

## Schema
{schema}

## Key Patterns

1. **Project with Budget and Location:**
```cypher
MATCH (p:Project)
OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
RETURN p.name, b.amount, b.currency, l.city, l.country
```

2. **Project Milestones/Timeline:**
```cypher
MATCH (p:Project)-[:HAS_MILESTONE]->(m:Milestone)
RETURN p.name, m.name AS milestone, m.dateText
ORDER BY p.name, m.dateText
```

3. **Challenges and Risks:**
```cypher
MATCH (p:Project)-[:HAS_CHALLENGE]->(c:Challenge)
RETURN p.name, collect(c.text) AS challenges
```

4. **Cross-Project Comparison:**
```cypher
MATCH (p:Project)
OPTIONAL MATCH (p)-[:HAS_BUDGET]->(b:Budget)
OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
WITH p, b, collect(m) AS milestones
RETURN p.name, b.amount, size(milestones) AS milestone_count
ORDER BY b.amount DESC
```

5. **Entity Relationships:**
```cypher
MATCH (p:Project)-[r]->(related)
WHERE NOT related:Chunk
RETURN p.name, type(r) AS relationship, labels(related)[0] AS entity_type,
       coalesce(related.name, related.text, related.amount) AS value
LIMIT 50
```

## Rules
- Use OPTIONAL MATCH when relationships may not exist
- Always include ORDER BY for consistent results
- Use collect() to aggregate multiple related nodes
- Limit results if the query could return many rows
- Return human-readable names, not IDs
- For comparisons across projects, ensure all projects are included

## Question
{question}

Return ONLY the Cypher query, no explanation.""".strip()

    def __init__(
        self,
        together_config: Optional[TogetherAIConfig] = None,
        together_api_key: Optional[str] = None,
        chat_model: str = "deepseek-ai/DeepSeek-V3",
        embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval",
    ) -> None:
        """Initialize GraphRAG builder.

        Args:
            together_config: Together AI configuration object.
            together_api_key: API key (alternative to config object).
            chat_model: Chat model identifier.
            embedding_model: Embedding model identifier.

        Raises:
            ValueError: If no API key is provided.
        """
        # Handle configuration
        if together_config:
            api_key = together_config.api_key
            chat_model = together_config.chat_model or chat_model
            embedding_model = together_config.embedding_model or embedding_model
        else:
            api_key = together_api_key

        if not api_key:
            raise ValueError("Together API key is required.")

        # Set environment variable for SDK
        os.environ["TOGETHER_API_KEY"] = api_key

        # Initialize models
        self.llm = ChatTogether(model=chat_model, temperature=0)
        self.embeddings = TogetherEmbeddings(model=embedding_model)

        # Initialize parsers and chunkers
        self._parser = ProjectReportParser()
        self._chunker = SemanticChunker(
            max_chunk_size=self.DEFAULT_CHUNK_SIZE + 300,  # Slightly larger for semantic chunks
            min_chunk_size=200,
            overlap_sentences=2,
        )

    def _load_pdf_pages(
        self,
        pdf_files: List[Any]
    ) -> Tuple[List[Document], List[Tuple[str, str]]]:
        """Load PDF files and extract pages with metadata.

        Args:
            pdf_files: List of gradio-uploaded file handles.

        Returns:
            Tuple of (all pages as Documents, list of (source_name, full_text)).
        """
        all_pages: List[Document] = []
        raw_texts: List[Tuple[str, str]] = []

        with log_step(logger, "Load PDF files", f"{len(pdf_files)} file(s)"):
            for f in pdf_files:
                src_name = (
                    getattr(f, "name", None) or
                    getattr(f, "orig_name", None) or
                    "uploaded.pdf"
                )
                logger.substep(f"Loading: {os.path.basename(src_name)}")
                loader = PyPDFLoader(f.name)
                pages = loader.load()
                all_pages.extend(pages)
                logger.substep(f"Extracted {len(pages)} pages")

                joined = "\n".join([p.page_content for p in pages])
                raw_texts.append((os.path.basename(src_name), joined))

        logger.info(f"Total pages loaded: {len(all_pages)}")
        return all_pages, raw_texts

    def _create_chunks(
        self,
        pages: List[Document],
        use_semantic_chunking: bool = True,
    ) -> List[Document]:
        """Split pages into chunks with normalized metadata.

        Args:
            pages: List of page Documents.
            use_semantic_chunking: If True, uses section-aware chunking.

        Returns:
            List of chunk Documents with metadata.
        """
        chunking_type = "semantic" if use_semantic_chunking else "character-based"
        with log_step(logger, "Create document chunks", chunking_type):
            if use_semantic_chunking:
                # Use semantic chunker that respects document structure
                logger.substep("Using section-aware semantic chunking")
                chunks = self._chunker.chunk_pages(pages, adaptive=True)
            else:
                # Fallback to simple character-based splitting
                logger.substep("Using RecursiveCharacterTextSplitter")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=self.DEFAULT_CHUNK_OVERLAP,
                )
                chunks = splitter.split_documents(pages)

            logger.substep(f"Raw chunks created: {len(chunks)}")

            processed_chunks: List[Document] = []
            for chunk in chunks:
                meta = dict(chunk.metadata or {})
                meta["source"] = os.path.basename(meta.get("source", "")) or "uploaded.pdf"

                # Normalize page numbers (PyPDFLoader uses 0-index)
                if "page" in meta and isinstance(meta["page"], int):
                    if meta["page"] == 0 or (not use_semantic_chunking):
                        meta["page"] = int(meta["page"]) + 1

                processed_chunks.append(Document(
                    page_content=chunk.page_content.replace("\n", " "),
                    metadata=meta,
                ))

        logger.info(f"Final chunks: {len(processed_chunks)}")
        return processed_chunks

    def _extract_structured_data(
        self,
        neo4j: Neo4jService,
        raw_texts: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        """Extract and upsert structured project data.

        Args:
            neo4j: Neo4j service instance.
            raw_texts: List of (source_name, full_text) tuples.

        Returns:
            List of project dictionaries with results/warnings.
        """
        projects_created: List[Dict[str, Any]] = []

        with log_step(logger, "Extract structured data", f"{len(raw_texts)} document(s)"):
            for source, full_text in raw_texts:
                logger.substep(f"Parsing: {source}")
                record = self._parser.parse(full_text, source)
                try:
                    proj = neo4j.upsert_structured_project(record)
                    projects_created.append(proj)
                    logger.substep(f"Created project: {proj.get('name', source)}")
                except Exception as e:
                    logger.warning(f"Failed to create project {source}: {e}")
                    projects_created.append({
                        "projectId": record.project_id or source,
                        "name": record.project_name or source,
                        "warning": str(e),
                    })

        logger.info(f"Structured extraction complete: {len(projects_created)} project(s)")
        return projects_created

    def _extract_llm_graph(
        self,
        neo4j: Neo4jService,
        chunks: List[Document],
        parallel: bool = True,
    ) -> None:
        """Extract entities/relationships using LLM and add to graph.

        Args:
            neo4j: Neo4j service instance.
            chunks: Document chunks for extraction.
            parallel: If True, uses parallel batch processing.
        """
        mode = "parallel" if parallel else "sequential"
        with log_step(logger, "LLM graph extraction", f"{len(chunks)} chunks, {mode}"):
            logger.substep("Initializing LLMGraphTransformer")
            transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=SchemaPolicy.ALLOWED_NODES,
                allowed_relationships=SchemaPolicy.ALLOWED_RELATIONSHIPS,
                node_properties=True,  # Enable property extraction for richer graph
            )

            if not parallel or len(chunks) <= self.EXTRACTION_BATCH_SIZE:
                # Sequential extraction for small chunk sets
                logger.substep("Using sequential extraction (small chunk set)")
                graph_documents = transformer.convert_to_graph_documents(chunks)
                neo4j.graph.add_graph_documents(graph_documents, include_source=True)
                logger.info(f"Added {len(graph_documents)} graph documents")
                return

            # Parallel extraction for larger chunk sets
            def process_batch(batch: List[Document]) -> List:
                """Process a batch of chunks."""
                try:
                    return transformer.convert_to_graph_documents(batch)
                except Exception:
                    return []

            # Split into batches
            batches = [
                chunks[i:i + self.EXTRACTION_BATCH_SIZE]
                for i in range(0, len(chunks), self.EXTRACTION_BATCH_SIZE)
            ]
            logger.substep(f"Split into {len(batches)} batches ({self.EXTRACTION_BATCH_SIZE} chunks each)")

            all_graph_docs = []
            failed_batches = 0

            # Process batches with thread pool for IO-bound LLM calls
            logger.substep(f"Starting parallel extraction with {self.MAX_EXTRACTION_WORKERS} workers")
            with ThreadPoolExecutor(max_workers=self.MAX_EXTRACTION_WORKERS) as executor:
                futures = {
                    executor.submit(process_batch, batch): i
                    for i, batch in enumerate(batches)
                }

                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        result = future.result(timeout=120)
                        all_graph_docs.extend(result)
                        logger.substep(f"Batch {batch_idx + 1}/{len(batches)} complete")
                    except Exception as e:
                        failed_batches += 1
                        logger.warning(f"Batch {batch_idx + 1} failed: {e}")

            # Bulk add to graph
            if all_graph_docs:
                logger.substep(f"Adding {len(all_graph_docs)} graph documents to Neo4j")
                neo4j.graph.add_graph_documents(all_graph_docs, include_source=True)

            if failed_batches > 0:
                logger.warning(f"{failed_batches} batch(es) failed during extraction")

        logger.info(f"LLM extraction complete: {len(all_graph_docs)} graph documents")

    def _create_vector_index(
        self,
        chunks: List[Document],
        neo4j_config: Neo4jConfig,
    ) -> Neo4jVector:
        """Create or refresh vector index for chunks.

        Args:
            chunks: Document chunks to index.
            neo4j_config: Neo4j connection configuration.

        Returns:
            Neo4jVector index instance.
        """
        with log_step(logger, "Create vector index", f"{len(chunks)} chunks"):
            logger.substep(f"Index name: {self.INDEX_NAME}")
            logger.substep(f"Keyword index: {self.KEYWORD_INDEX_NAME}")
            logger.substep("Creating hybrid search index (dense + BM25)")

            vector = Neo4jVector.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                url=neo4j_config.uri,
                username=neo4j_config.username,
                password=neo4j_config.password,
                database=neo4j_config.database or "neo4j",
                index_name=self.INDEX_NAME,
                keyword_index_name=self.KEYWORD_INDEX_NAME,
                node_label=self.NODE_LABEL,
                embedding_node_property="embedding",
                search_type="hybrid",
            )

        logger.info("Vector index created successfully")
        return vector

    def _create_qa_chain(self, neo4j: Neo4jService) -> GraphCypherQAChain:
        """Create Cypher QA chain for graph querying.

        Args:
            neo4j: Neo4j service instance.

        Returns:
            GraphCypherQAChain instance.
        """
        with log_step(logger, "Create Cypher QA chain"):
            logger.substep("Configuring enhanced Cypher prompt template")
            cypher_prompt = PromptTemplate(
                template=self.CYPHER_PROMPT_TEMPLATE,
                input_variables=["schema", "question"],
            )

            logger.substep("Initializing GraphCypherQAChain")
            chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=neo4j.graph,
                cypher_prompt=cypher_prompt,
                verbose=False,
                allow_dangerous_requests=True,
            )

        logger.info("Cypher QA chain ready")
        return chain

    @trace_flow("PDF Ingestion Pipeline")
    def ingest(
        self,
        pdf_files: List[Any],
        neo4j_config: Optional[Neo4jConfig] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
        clear_db: bool = True,
    ) -> Tuple[str, AppState]:
        """Ingest one or more PDF reports into Neo4j and build GraphRAG indices.

        Args:
            pdf_files: List of gradio-uploaded file handles.
            neo4j_config: Neo4j configuration object (preferred).
            neo4j_uri: Neo4j connection URI (alternative).
            neo4j_user: Username (alternative).
            neo4j_password: Password (alternative).
            neo4j_database: Database name.
            clear_db: If True, deletes all existing nodes prior to ingestion.

        Returns:
            Tuple of (human-readable status message, AppState).

        Notes:
            - The ingestion process can be compute-heavy due to LLM graph extraction.
            - Even if the deterministic parser yields partial results, chunk retrieval
              still works.
        """
        # Validate inputs
        if not pdf_files:
            logger.warning("No PDF files provided")
            return "Please upload at least one PDF.", AppState()

        logger.info(f"Starting ingestion of {len(pdf_files)} PDF file(s)")

        # Build config from parameters if not provided
        if neo4j_config is None:
            neo4j_config = Neo4jConfig(
                uri=neo4j_uri or "",
                username=neo4j_user or "neo4j",
                password=neo4j_password or "",
                database=neo4j_database,
            )

        if not neo4j_config.is_valid():
            logger.error("Invalid Neo4j configuration")
            return "Please provide Neo4j connection details.", AppState()

        # Connect to Neo4j
        with log_step(logger, "Connect to Neo4j"):
            try:
                neo4j = Neo4jService(
                    uri=neo4j_config.uri,
                    user=neo4j_config.username,
                    password=neo4j_config.password,
                    database=neo4j_config.database,
                )
                logger.substep(f"Connected to {neo4j_config.uri}")
            except Neo4jConnectionError as e:
                logger.error(f"Neo4j connection failed: {e}")
                return (
                    f"Neo4j connection failed. For Aura, use the exact URI shown in the "
                    f"console (typically starts with neo4j+s://...). Error: {e}",
                    AppState(),
                )

        # Ensure constraints
        with log_step(logger, "Ensure database constraints"):
            neo4j.ensure_constraints()

        # Clear database if requested
        if clear_db:
            with log_step(logger, "Clear existing data"):
                neo4j.clear()

        # 1) Load PDF pages
        all_pages, raw_texts = self._load_pdf_pages(pdf_files)

        # 2) Structured extraction (high precision)
        projects_created = self._extract_structured_data(neo4j, raw_texts)

        # 3) Create chunks
        chunks = self._create_chunks(all_pages)

        # 4) LLM-based KG extraction (high recall)
        self._extract_llm_graph(neo4j, chunks)

        # 5) Vector index
        vector = self._create_vector_index(chunks, neo4j_config)

        # 6) Cypher QA chain
        qa_chain = self._create_qa_chain(neo4j)

        # Build status message
        proj_lines = []
        for p in projects_created:
            warn = f" (warning: {p.get('warning')})" if "warning" in p else ""
            proj_lines.append(f"- {p.get('name')} [{p.get('projectId')}]{warn}")

        msg = (
            "Ingestion complete.\n\n"
            f"Neo4j database: `{neo4j_config.database}`\n\n"
            "Projects found:\n" + "\n".join(proj_lines)
        )

        logger.info(f"Ingestion complete: {len(projects_created)} project(s), {len(chunks)} chunks")

        return msg, AppState(
            neo4j=neo4j,
            vector=vector,
            qa_chain=qa_chain,
            llm=self.llm,
        )

    def ingest_with_progress(
        self,
        pdf_files: List[Any],
        neo4j_config: Optional[Neo4jConfig] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
        clear_db: bool = True,
        skip_llm_extraction: bool = True,  # Skip LLM extraction for faster ingestion
    ) -> Generator[Tuple[str, float, Optional[AppState]], None, None]:
        """Ingest PDFs with progress updates for UI.

        This generator yields progress updates during ingestion, allowing
        the UI to display a progress bar with status messages.

        Args:
            pdf_files: List of gradio-uploaded file handles.
            neo4j_config: Neo4j configuration object (preferred).
            neo4j_uri: Neo4j connection URI (alternative).
            neo4j_user: Username (alternative).
            neo4j_password: Password (alternative).
            neo4j_database: Database name.
            clear_db: If True, deletes all existing nodes prior to ingestion.
            skip_llm_extraction: If True, skips LLM graph extraction for faster ingestion.

        Yields:
            Tuple of (status_message, progress_fraction, optional_state)
            - progress_fraction is 0.0 to 1.0
            - optional_state is None until final yield, then contains AppState

        Example:
            >>> for status, progress, state in builder.ingest_with_progress(files, config):
            ...     print(f"{progress*100:.0f}%: {status}")
            ...     if state:
            ...         print("Done!")
        """
        start_time = time.time()

        # Validate inputs
        if not pdf_files:
            yield "❌ Please upload at least one PDF file.", 0.0, None
            return

        # Build config from parameters if not provided
        if neo4j_config is None:
            neo4j_config = Neo4jConfig(
                uri=neo4j_uri or "",
                username=neo4j_user or "neo4j",
                password=neo4j_password or "",
                database=neo4j_database,
            )

        if not neo4j_config.is_valid():
            yield "❌ Please provide Neo4j connection details.", 0.0, None
            return

        # Step 1: Connect to Neo4j (5%)
        yield "🔌 Connecting to Neo4j...", 0.05, None
        try:
            neo4j = Neo4jService(
                uri=neo4j_config.uri,
                user=neo4j_config.username,
                password=neo4j_config.password,
                database=neo4j_config.database,
            )
        except Neo4jConnectionError as e:
            yield f"❌ Neo4j connection failed: {e}", 0.05, None
            return

        # Step 2: Ensure constraints (10%)
        yield "📋 Setting up database constraints...", 0.10, None
        neo4j.ensure_constraints()

        # Step 3: Clear database if requested (15%)
        if clear_db:
            yield "🗑️ Clearing existing data...", 0.15, None
            neo4j.clear()

        # Step 4: Load PDF pages (25%)
        yield f"📄 Loading {len(pdf_files)} PDF file(s)...", 0.20, None
        all_pages, raw_texts = self._load_pdf_pages(pdf_files)
        yield f"📄 Loaded {len(all_pages)} pages from PDFs", 0.25, None

        # Step 5: Structured extraction (35%)
        yield "🔍 Extracting structured project data...", 0.30, None
        projects_created = self._extract_structured_data(neo4j, raw_texts)
        project_names = [p.get('name', 'Unknown') for p in projects_created]
        yield f"✅ Found {len(projects_created)} project(s): {', '.join(project_names)}", 0.35, None

        # Step 6: Create chunks (45%)
        yield "✂️ Creating document chunks...", 0.40, None
        chunks = self._create_chunks(all_pages)
        yield f"✅ Created {len(chunks)} chunks", 0.45, None

        # Step 7: LLM Graph Extraction (optional) (45-70%)
        if not skip_llm_extraction:
            yield f"🧠 Extracting entities with LLM ({len(chunks)} chunks)...", 0.50, None
            # This is the slowest step - show batch progress
            total_batches = (len(chunks) + self.EXTRACTION_BATCH_SIZE - 1) // self.EXTRACTION_BATCH_SIZE
            for batch_num in range(total_batches):
                progress = 0.50 + (0.20 * (batch_num + 1) / total_batches)
                yield f"🧠 LLM extraction: batch {batch_num + 1}/{total_batches}...", progress, None
            self._extract_llm_graph(neo4j, chunks)
            yield "✅ LLM graph extraction complete", 0.70, None
        else:
            yield "⏩ Skipping LLM extraction (using fast mode)", 0.70, None

        # Step 8: Create vector index (90%)
        yield f"📊 Creating vector index ({len(chunks)} chunks)...", 0.75, None
        vector = self._create_vector_index(chunks, neo4j_config)
        yield "✅ Vector index created", 0.90, None

        # Step 9: Create QA chain (95%)
        yield "⚙️ Initializing QA chain...", 0.95, None
        qa_chain = self._create_qa_chain(neo4j)

        # Final step: Complete (100%)
        elapsed = time.time() - start_time
        proj_lines = []
        for p in projects_created:
            warn = f" ⚠️ {p.get('warning')}" if "warning" in p else ""
            proj_lines.append(f"- **{p.get('name')}** [{p.get('projectId')}]{warn}")

        final_msg = (
            f"## ✅ Ingestion Complete ({elapsed:.1f}s)\n\n"
            f"**Database:** `{neo4j_config.database}`\n\n"
            f"**Projects found:**\n" + "\n".join(proj_lines) + "\n\n"
            f"**Stats:** {len(chunks)} chunks indexed"
        )

        yield final_msg, 1.0, AppState(
            neo4j=neo4j,
            vector=vector,
            qa_chain=qa_chain,
            llm=self.llm,
        )
