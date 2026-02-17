from openai import OpenAI
from sqlalchemy import create_engine, select
from sentence_transformers import SentenceTransformer
from shared.config_loader import config_loader
from database.model import TransactionModel, EmbeddingModel


class RAGQueryEngine:
    def __init__(self):
        self.cfg = config_loader.load()
        self.engine = create_engine(self.cfg.database.url)
        self.embedder = SentenceTransformer(self.cfg.embedding.model_name)

        self.client = OpenAI(
            base_url=self.cfg.llm.base_url,
            api_key=self.cfg.llm.api_key
        )
        self._check_llm_connection()

    def _check_llm_connection(self):
        try:
            self.client.models.list()
            print(f"LLM Connection successful: {self.cfg.llm.base_url}")
        except Exception as e:
            print(f"Failed to connect to LLM at {self.cfg.llm.base_url}")
            print(f"Error detail: {e}")

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

        response = self.client.chat.completions.create(
            model=self.cfg.llm.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def ask(self, query: str, top_k: int = 5):
        print("Start retrieving context")
        context = self._retrieve_context(query, top_k)
        if not context:
            return "No data found."
        return self._generate_response(query, context)