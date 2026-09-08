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


class _FakeVector(list):
    """Stands in for a numpy row from model.encode(list_of_texts) — real vectors expose .tolist()."""

    def tolist(self):
        return list(self)


@patch("services.embedder.processor.config_loader")
def test_embedding_processor_batch_encodes_all_texts_in_one_call(mock_config_loader):
    mock_config_loader.load.return_value = MagicMock()

    mock_sentence_transformer_model = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = [_FakeVector([0.1, 0.2]), _FakeVector([0.3, 0.4])]
    mock_sentence_transformer_model.get_model.return_value = mock_model

    processor = EmbeddingProcessor(mock_sentence_transformer_model)
    jobs = [
        {"amount": 100.5, "features": {"f1": 0.5}, "is_fraud": False},
        {"amount": 50.0, "features": {"f1": 1.0}, "is_fraud": True},
    ]
    results = processor.create_embeddings(jobs)

    assert [vector for vector, _ in results] == [[0.1, 0.2], [0.3, 0.4]]
    assert "fraud status: normal" in results[0][1]
    assert "fraud status: CONFIRMED FRAUD" in results[1][1]
    mock_model.encode.assert_called_once_with([results[0][1], results[1][1]])


@patch("services.embedder.processor.config_loader")
def test_embedding_processor_batch_empty(mock_config_loader):
    mock_config_loader.load.return_value = MagicMock()
    mock_sentence_transformer_model = MagicMock()
    processor = EmbeddingProcessor(mock_sentence_transformer_model)

    assert processor.create_embeddings([]) == []
    mock_sentence_transformer_model.get_model.return_value.encode.assert_not_called()
