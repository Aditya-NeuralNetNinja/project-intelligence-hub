"""Application settings and configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Neo4jConfig:
    """Neo4j database connection configuration."""

    uri: str = ""
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    def is_valid(self) -> bool:
        return bool(self.uri and self.username and self.password)


@dataclass
class TogetherAIConfig:
    """Together AI API configuration."""

    api_key: str = ""
    chat_model: str = "meta-llama/meta-llama-3.1-8b-instruct-turbo"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    def is_valid(self) -> bool:
        return bool(self.api_key)


@dataclass
class AppConfig:
    """Application-level configuration."""

    port: int = 7860
    host: str = "0.0.0.0"


@dataclass
class Settings:
    """Centralized application settings."""

    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    together_ai: TogetherAIConfig = field(default_factory=TogetherAIConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None) -> "Settings":
        """Load settings from environment variables."""
        load_dotenv(dotenv_path)

        neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", ""),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

        together_ai = TogetherAIConfig(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            chat_model=os.getenv("TOGETHER_CHAT_MODEL", "meta-llama/meta-llama-3.1-8b-instruct-turbo"),
            embedding_model=os.getenv("TOGETHER_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        )

        app = AppConfig(
            port=int(os.getenv("PORT", "7860")),
            host=os.getenv("HOST", "0.0.0.0"),
        )

        return cls(neo4j=neo4j, together_ai=together_ai, app=app)

    def apply_to_env(self) -> None:
        """Apply current settings to environment variables."""
        if self.together_ai.api_key:
            os.environ["TOGETHER_API_KEY"] = self.together_ai.api_key
