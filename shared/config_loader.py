import os

import yaml
from pathlib import Path
from pydantic import BaseModel


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

class MCPServerConfig(BaseModel):
    url: str

class MCPServersConfig(BaseModel):
    analysis: MCPServerConfig
    repository: MCPServerConfig

class CorrelationAnalysisConfig(BaseModel):
    features: dict
    thresholds: dict

class ApplicationConfig(BaseModel):
    database: DatabaseConfig
    kafka: KafkaConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    dashboard: DashboardConfig
    correlation_analysis: CorrelationAnalysisConfig
    mcp_servers: MCPServersConfig

class ConfigLoader:
    _instance = None
    _config: ApplicationConfig = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def load(self, config_path: str = None) -> ApplicationConfig:
        if self._config is None:
            project_root = Path(__file__).resolve().parents[1]

            env = os.getenv("APP_ENV", "").lower()

            if config_path is None:
                file_name = f"application-{env}.yaml" if env else "application.yaml"
                config_path = f"config/{file_name}"

                print(f"Loading config: {file_name}")

            full_path = project_root / config_path

            if not full_path.exists():
                print(f"Warning: {full_path} not found, falling back to default application.yaml")
                full_path = project_root / "config/application.yaml"

            if not full_path.exists():
                raise FileNotFoundError(f"Config file not found at {full_path}")

            print(f"Successfully loaded config from: {full_path}")
            with open(full_path, "r") as f:
                raw_config = yaml.safe_load(f)

            self._config = ApplicationConfig(**raw_config)

        return self._config

config_loader = ConfigLoader()
