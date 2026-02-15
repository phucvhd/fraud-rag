import pytest
from uuid import uuid4
from datetime import datetime
from sqlalchemy import create_engine, delete
from sqlalchemy.dialects.postgresql import insert
from services.api.rag_service import RAGQueryEngine
from database.model import TransactionModel, EmbeddingModel
from services.embedder.processor import EmbeddingProcessor
from shared.config_loader import config_loader


class TestFraudRAG:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = config_loader.load()
        self.engine = create_engine(self.cfg.database.url)
        self.rag_engine = RAGQueryEngine()
        self.processor = EmbeddingProcessor()
        self.test_id = str(uuid4())

    def _insert_real_vector(self):
        amount = 5000.0
        features = {"V1": -2.5, "V2": 1.8}

        vector, text_content = self.processor.create_embedding(amount, features)

        with self.engine.begin() as conn:
            conn.execute(delete(EmbeddingModel).where(EmbeddingModel.transaction_id == self.test_id))
            conn.execute(delete(TransactionModel).where(TransactionModel.transaction_id == self.test_id))

            conn.execute(
                insert(TransactionModel).values(
                    transaction_id=self.test_id,
                    event_time_seconds=int(datetime.now().timestamp()),
                    event_timestamp=datetime.now(),
                    amount=amount,
                    is_fraud=True,
                    features=features,
                    data_source="integration_test"
                )
            )

            conn.execute(
                insert(EmbeddingModel).values(
                    transaction_id=self.test_id,
                    embedding=vector,
                    embedding_text=text_content,
                    embedding_model=self.cfg.embedding.model_name
                )
            )

    def test_rag_ask_flow(self):
        self._insert_real_vector()

        query = "Find me suspicious transactions with amount over 4000 EUR"
        response = self.rag_engine.ask(query, top_k=1)

        assert response is not None
        assert "5000" in response or "suspicious" in response.lower()
        print(f"LLM Response: {response}")

    def test_semantic_retrieval_only(self):
        self._insert_real_vector()

        query_text = "Transaction of 5000 EUR"
        context = self.rag_engine._retrieve_context(query_text, top_k=5)

        assert len(context) > 0
        found = any(float(item["amount"]) == 5000.0 for item in context)
        assert found, f"Could not find transaction 5000.0. Top result: {context[0]['amount'] if context else 'None'}"