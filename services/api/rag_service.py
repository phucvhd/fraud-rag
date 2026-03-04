from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine, select
from sentence_transformers import SentenceTransformer

from services.agent.agent import LLMAgent
from shared.config_loader import config_loader
from database.model import TransactionModel, EmbeddingModel


class RAGQueryEngine:
    def __init__(self, agent):
        self.cfg = config_loader.load()
        self.engine = create_engine(self.cfg.database.url)
        self.embedder = SentenceTransformer(self.cfg.embedding.model_name)
        self.agent = agent

    def _retrieve_context(self, query: str, top_k: int):
        query_vector = self.embedder.encode(query).tolist()

        stmt = (
            select(
                TransactionModel.amount,
                TransactionModel.event_timestamp,
                EmbeddingModel.embedding_text
            )
            .join(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
            .order_by(EmbeddingModel.embedding.l2_distance(query_vector))
            .limit(top_k)
        )

        with self.engine.connect() as conn:
            return conn.execute(stmt).mappings().all()

    def _generate_response(self, query: str, context: list):
        context_text = "\n".join([
            f"Time: {r['event_timestamp']} | Amount: {r['amount']} | Details: {r['embedding_text']}"
            for r in context
        ])

        prompt = f"""
        You are a Fraud Detection Expert. Analyze the following transactions retrieved from our database:

        {context_text}

        Question: {query}

        Instructions:
        1. Identify any transaction with amount > 1000 EUR.
        2. If found, list them and explain why they might be suspicious.
        3. Answer in English.
        """

        llm = self.agent.get_client()

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content

    def historical_db_lookup(self, query: str, top_k: int = 5):
        print("Start retrieving context")
        context = self._retrieve_context(query, top_k)
        if not context:
            return "No data found."
        return self._generate_response(query, context)