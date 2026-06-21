from unittest.mock import patch, MagicMock
from services.embedder.processor import EmbeddingProcessor


@patch("services.embedder.processor.config_loader")
def test_embedding_processor_normal(mock_config_loader):
    mock_config_loader.load.return_value = MagicMock()

    mock_sentence_transformer_model = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_sentence_transformer_model.get_model.return_value = mock_model

    processor = EmbeddingProcessor(mock_sentence_transformer_model)
    vector, text = processor.create_embedding(100.5, {"f1": 0.5, "f2": 1.2}, False)

    assert vector == [0.1, 0.2, 0.3]
    assert "100.5 EUR" in text
    assert "fraud status: normal" in text
    assert "f1: 0.5000" in text
    assert "f2: 1.2000" in text
    mock_model.encode.assert_called_once_with(text)


@patch("services.embedder.processor.config_loader")
def test_embedding_processor_fraud(mock_config_loader):
    mock_config_loader.load.return_value = MagicMock()

    mock_sentence_transformer_model = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.4, 0.5]
    mock_sentence_transformer_model.get_model.return_value = mock_model

    processor = EmbeddingProcessor(mock_sentence_transformer_model)
    _, text = processor.create_embedding(218.09, {"V1": 4.4045}, True)

    assert "fraud status: CONFIRMED FRAUD" in text
