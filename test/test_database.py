import uuid
import datetime
from database.model import TransactionModel, EmbeddingModel

def test_transaction_model_creation():
    tx = TransactionModel(transaction_id=uuid.uuid4(), event_time_seconds=123, event_timestamp=datetime.datetime.now(), amount=10.5, features={"a": 1}, data_source="src")
    assert isinstance(tx.transaction_id, uuid.UUID)
    assert tx.amount == 10.5
    assert tx.is_fraud is None
    assert tx.features == {"a": 1}
    assert tx.data_source == "src"

def test_embedding_model_creation():
    emb = EmbeddingModel(transaction_id=uuid.uuid4(), embedding=[0.1, 0.2, 0.3], embedding_text="text", embedding_model="model")
    assert isinstance(emb.transaction_id, uuid.UUID)
    assert emb.embedding == [0.1, 0.2, 0.3]
    assert emb.embedding_text == "text"
    assert emb.embedding_model == "model"
