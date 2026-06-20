import logging
import os
from pathlib import Path
from typing import ClassVar, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    url: str
    batch_size: int = 50


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    topic: str
    group_id: str


class EmbeddingConfig(BaseModel):
    model_name: str
    dimension: int


class LLMConfig(BaseModel):
    provider: str
    api_key: str
    model_name: str
    base_url: str


class DashboardConfig(BaseModel):
    rag_url: str
    inject_url: str
    transactions_url: str


class MCPServerConfig(BaseModel):
    url: str


class MCPServersConfig(BaseModel):
    analysis: MCPServerConfig
    repository: MCPServerConfig


class CorrelationAnalysisConfig(BaseModel):
    features: dict[str, float]
    thresholds: dict[str, float]


class ApplicationConfig(BaseModel):
    database: DatabaseConfig
    kafka: KafkaConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    dashboard: DashboardConfig
    correlation_analysis: CorrelationAnalysisConfig
    mcp_servers: MCPServersConfig


class ConfigLoader:
    _instance: ClassVar[Optional["ConfigLoader"]] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> ApplicationConfig:
        if self._config is None:
            project_root = Path(__file__).resolve().parents[1]
            env = os.getenv("APP_ENV", "").lower()

            if config_path is None:
                file_name = f"application-{env}.yaml" if env else "application.yaml"
                config_path = f"config/{file_name}"
                logger.info("Loading config: %s", file_name)

            full_path = project_root / config_path

            if not full_path.exists():
                logger.warning(
                    "%s not found, falling back to default application.yaml", full_path
                )
                full_path = project_root / "config/application.yaml"

            if not full_path.exists():
                raise FileNotFoundError(f"Config file not found at {full_path}")

            logger.info("Loading config from: %s", full_path)
            with open(full_path, "r") as f:
                raw_text = os.path.expandvars(f.read())
                raw_config = yaml.safe_load(raw_text)

            self._config = ApplicationConfig(**raw_config)

        return self._config


config_loader = ConfigLoader()
