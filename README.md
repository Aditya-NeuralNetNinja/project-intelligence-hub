# Project Intelligence Hub

Ingest PDF project reports → Extract structured entities → Build a Neo4j knowledge graph → Query with natural language.

GraphRAG pipeline combining knowledge graphs with LLM-powered retrieval for project analytics.

## Live Demo
Webapp hosted on Hugging Face Spaces: [*Live Demo*](https://huggingface.co/spaces/adi-123/Project-Report-Analyzer)

## Architecture Diagram
![Architecture](https://github.com/Aditya-NeuralNetNinja/project-intelligence-hub/blob/main/architecture-diagram.svg)

## Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM Provider** | Together AI (DeepSeek-V3 chat, Llama-3.2-3B classifier) |
| **Embeddings** | Together AI (m2-bert-80M-8k-retrieval) |
| **Graph Database** | Neo4j Aura |
| **Vector Store** | Neo4jVector (hybrid: dense + BM25) |
| **Reranker** | sentence-transformers (ms-marco-MiniLM-L-6-v2) |
| **Orchestration** | LangChain |
| **UI** | Gradio |

## Graph Schema

```
(:Project)─[:HAS_BUDGET]─▶(:Budget)
    │
    ├──[:LOCATED_IN]─▶(:Location)
    │
    ├──[:HAS_MILESTONE]─▶(:Milestone)
    │
    ├──[:HAS_CHALLENGE]─▶(:Challenge)
    │
    └──[:HAS_REPORT]─▶(:Report)
```

**Node Properties:**
- `Project`: name, status, statusReason, projectManager, plantOwner, capacity, scope
- `Budget`: amount, currency, kind (TIV)
- `Location`: address, city, state, country, postal
- `Milestone`: name, dateText, sentence
- `Challenge`: text (derived from status reasons, delays, constraints)

## Query Routing

| Intent | Method | LLM Calls |
|--------|--------|-----------|
| Budget, Location, Timeline, Challenges | Cypher Template | 0 |
| Contacts, Technical, Comparison | Cypher Template | 0 |
| General / Complex | RAG Pipeline | 1 (synthesis) |

**Intent Classification:** LLM-based (Llama-3.2-3B) with pattern-matching fallback.

## Project Structure

```
├── main.py                 # Entry point
├── src/
│   ├── config/
│   │   ├── settings.py     # Environment configuration
│   │   ├── schema.py       # Graph schema constraints
│   │   └── logging_config.py
│   ├── models/
│   │   ├── project.py      # ProjectRecord dataclass
│   │   └── state.py        # AppState (runtime context)
│   ├── parsers/
│   │   ├── project_parser.py   # Regex-based field extraction
│   │   └── smart_chunker.py    # Section-aware chunking
│   ├── services/
│   │   ├── builder.py      # Ingestion pipeline orchestrator
│   │   ├── answerer.py     # Query processing orchestrator
│   │   ├── retriever.py    # OptimizedRetriever (expansion + rerank)
│   │   ├── reranker.py     # Cross-encoder reranking
│   │   ├── neo4j_service.py    # Graph operations
│   │   ├── cypher_templates.py # Pre-validated queries
│   │   └── cache.py        # Answer caching
│   └── ui/
│       └── gradio_app.py   # Web interface
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
TOGETHER_API_KEY=your_api_key
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Run:
```bash
python main.py
```

Open `http://localhost:7860`

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM orchestration, document loaders |
| `langchain-together` | Together AI integration |
| `langchain-community` | Neo4jVector, Neo4jGraph |
| `neo4j` | Graph database driver |
| `sentence-transformers` | Cross-encoder reranking |
| `pypdf` | PDF parsing |
| `gradio` | Web UI |
| `pydantic` | Configuration validation |

## Requirements

- Python 3.10+
- Neo4j Aura (or self-hosted Neo4j 5.x)
- Together AI API key
