from unittest.mock import MagicMock

from services.embedder.processor import EmbeddingProcessor


def _vectorizer(transform_return=None, transform_many_return=None, version="v1"):
    vec = MagicMock()
    vec.transform.return_value = transform_return if transform_return is not None else [0.1, 0.2, 0.3]
    vec.transform_many.return_value = transform_many_return if transform_many_return is not None else []
    vec.model_descriptor = f"standardscaler-{version}"
    return vec


def test_embedding_processor_transforms_features_and_builds_provenance():
    processor = EmbeddingProcessor(_vectorizer(transform_return=[0.1, 0.2, 0.3]))

    vector, text = processor.create_embedding(100.5, {"V1": 0.5, "V2": 1.2})

    assert vector == [0.1, 0.2, 0.3]
    # Provenance text records what was vectorized: the amount and the features...
    assert "100.5 EUR" in text
    assert "V1: 0.5000" in text
    assert "V2: 1.2000" in text
    # ...but never the fraud label — the label must not enter the representation.
    assert "fraud" not in text.lower()


def test_embedding_processor_model_descriptor_comes_from_vectorizer():
    processor = EmbeddingProcessor(_vectorizer(version="20260914T000000Z"))
    assert processor.model_descriptor == "standardscaler-20260914T000000Z"


def test_embedding_processor_batch_transforms_all_in_one_call():
    vec = _vectorizer(transform_many_return=[[0.1, 0.2], [0.3, 0.4]])
    processor = EmbeddingProcessor(vec)
    jobs = [
        {"amount": 100.5, "features": {"V1": 0.5}},
        {"amount": 50.0, "features": {"V1": 1.0}},
    ]

    results = processor.create_embeddings(jobs)

    assert [vector for vector, _ in results] == [[0.1, 0.2], [0.3, 0.4]]
    assert "100.5 EUR" in results[0][1]
    assert "50.0 EUR" in results[1][1]
    vec.transform_many.assert_called_once_with(jobs)


def test_embedding_processor_batch_empty():
    vec = _vectorizer()
    processor = EmbeddingProcessor(vec)

    assert processor.create_embeddings([]) == []
    vec.transform_many.assert_not_called()
