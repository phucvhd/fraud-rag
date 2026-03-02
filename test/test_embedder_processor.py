import pytest
from unittest.mock import patch, MagicMock
from services.embedder.processor import EmbeddingProcessor

@patch("services.embedder.processor.SentenceTransformer")
@patch("services.embedder.processor.config_loader")
def test_embedding_processor(mock_config_loader, mock_sentence_transformer):
    mock_config = MagicMock()
    mock_config.embedding.model_name = "test_model"
    mock_config_loader.load.return_value = mock_config

    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_sentence_transformer.return_value = mock_model

    processor = EmbeddingProcessor()
    vector, text = processor.create_embedding(100.5, {"f1": 0.5, "f2": 1.2})

    assert vector == [0.1, 0.2, 0.3]
    assert "100.5 EUR" in text
    assert "f1: 0.5" in text
    assert "f2: 1.2" in text
    mock_model.encode.assert_called_once_with(text)
