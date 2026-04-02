"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    host: str = Field("0.0.0.0", alias="APP_HOST")
    port: int = Field(8000, alias="APP_PORT")
    debug: bool = Field(False, alias="APP_DEBUG")
    log_level: str = Field("INFO", alias="APP_LOG_LEVEL")
    # Leave blank to disable API key authentication (useful for local dev)
    api_key: str = Field("", alias="API_KEY")

    model_config = SettingsConfigDict(populate_by_name=True)


class OpenrouterSettings(BaseSettings):
    base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    api_key: str = Field("", alias="OPENROUTER_API_KEY")
    default_model: str = Field(
        "qwen/qwen3-30b-a3b-thinking-2507", alias="OPENROUTER_DEFAULT_MODEL"
    )
    max_tokens: int = Field(8192, alias="OPENROUTER_MAX_TOKENS")
    temperature: float = Field(0.0, alias="OPENROUTER_TEMPERATURE")

    model_config = SettingsConfigDict(populate_by_name=True)


class LocalModelSettings(BaseSettings):
    base_url: str = Field("http://localhost:11434/v1", alias="LOCAL_MODEL_BASE_URL")
    api_key: str = Field("ollama", alias="LOCAL_MODEL_API_KEY")
    model_name: str = Field("", alias="LOCAL_MODEL_NAME")

    model_config = SettingsConfigDict(populate_by_name=True)


class LangfuseSettings(BaseSettings):
    host: str = Field("http://localhost", alias="LANGFUSE_HOST")
    port: int = Field(3000, alias="LANGFUSE_PORT")
    public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")

    @property
    def url(self) -> str:
        return f"{self.host}:{self.port}"

    model_config = SettingsConfigDict(populate_by_name=True)


class MinIOSettings(BaseSettings):
    endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    access_key: str = Field("minioadmin", alias="MINIO_ACCESS_KEY")
    secret_key: str = Field("minioadmin", alias="MINIO_SECRET_KEY")
    bucket: str = Field("agent-files", alias="MINIO_BUCKET")
    secure: bool = Field(False, alias="MINIO_SECURE")

    model_config = SettingsConfigDict(populate_by_name=True)


class ElasticSearchSettings(BaseSettings):
    url: str = Field("http://localhost:9200", alias="ELASTICSEARCH_URL")
    index: str = Field("agent-system-logs", alias="ELASTICSEARCH_INDEX")
    username: str = Field("", alias="ELASTICSEARCH_USERNAME")
    password: str = Field("", alias="ELASTICSEARCH_PASSWORD")

    model_config = SettingsConfigDict(populate_by_name=True)


class SkillsSettings(BaseSettings):
    # "local" | "langfuse" | "hybrid"  (hybrid = langfuse preferred, local fallback)
    source: str = Field("hybrid", alias="SKILLS_SOURCE")
    local_dir: str = Field("./skills", alias="SKILLS_LOCAL_DIR")
    # How long (seconds) a Langfuse-fetched skill stays in cache before re-fetching
    langfuse_expiry_time: float = Field(100.0, alias="LANGFUSE_EXPIRY_TIME")

    model_config = SettingsConfigDict(populate_by_name=True)


class MCPServerConfig(BaseSettings):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    model_config = SettingsConfigDict(populate_by_name=True)


class Settings(BaseSettings):
    """Root settings that aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    openrouter: OpenrouterSettings = Field(default_factory=OpenrouterSettings)
    local_model: LocalModelSettings = Field(default_factory=LocalModelSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    elasticsearch: ElasticSearchSettings = Field(default_factory=ElasticSearchSettings)
    skills: SkillsSettings = Field(default_factory=SkillsSettings)

    # Direct PostgreSQL URL for agent config persistence
    # In docker-compose this is overridden to use the internal hostname.
    agent_postgres_url: str = Field(
        "postgresql://agent:agent@localhost:5433/agentdb",
        alias="AGENT_POSTGRES_URL",
    )

    # Raw JSON list of MCP server configs
    mcp_servers_raw: str = Field("[]", alias="MCP_SERVERS")

    @property
    def mcp_servers(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.mcp_servers_raw)
        except json.JSONDecodeError:
            return []

    @field_validator("mcp_servers_raw", mode="before")
    @classmethod
    def validate_mcp_servers_json(cls, v: Any) -> str:
        if isinstance(v, list):
            return json.dumps(v)
        return str(v)

    @model_validator(mode="before")
    @classmethod
    def _build_nested(cls, values: Any) -> Any:
        """Allow flat env vars to propagate into nested models."""
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
