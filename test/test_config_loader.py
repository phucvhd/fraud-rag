import pytest
from unittest.mock import patch, mock_open
from shared.config_loader import ConfigLoader, ApplicationConfig

@pytest.fixture(autouse=True)
def reset_singleton():
    ConfigLoader._instance = None
    ConfigLoader._config = None
    yield

def test_config_loader_singleton():
    loader1 = ConfigLoader()
    loader2 = ConfigLoader()
    assert loader1 is loader2

@patch("shared.config_loader.Path.exists")
@patch("builtins.open", new_callable=mock_open, read_data="""
database:
  url: postgresql://user:pass@localhost:5432/db
  batch_size: 100
kafka:
  bootstrap_servers: localhost:9092
  topic: test_topic
  group_id: test_group
embedding:
  model_name: test_model
  dimension: 384
llm:
  provider: openai
  api_key: test_key
  model_name: gpt-3.5-turbo
  base_url: https://api.openai.com/v1
dashboard:
  rag_url: http://localhost:8000
  inject_url: http://localhost:8001
""")
def test_config_loader_load_success(mock_file, mock_exists):
    mock_exists.return_value = True
    loader = ConfigLoader()
    config = loader.load("dummy/path.yaml")
    assert isinstance(config, ApplicationConfig)
    assert config.database.url == "postgresql://user:pass@localhost:5432/db"
    assert config.database.batch_size == 100
    assert config.kafka.topic == "test_topic"
    assert config.embedding.dimension == 384

@patch("shared.config_loader.Path.exists")
def test_config_loader_file_not_found(mock_exists):
    mock_exists.return_value = False
    loader = ConfigLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("wrong/path.yaml")
