import openai
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from shared.config_loader import config_loader


class RAGQueryEngine:
    def __init__(self):
        self.cfg = config_loader.load("config/application.yaml")
        self.engine = create_engine(self.cfg.database.url)
        self.embedder = SentenceTransformer(self.cfg.embedding.model_name)

        openai.api_key = self.cfg.llm.api_key

    def _get_semantic_context(self, query: str, top_k: int):
        query_vector = self.embedder.encode(query).tolist()

        sql = text("""
            SELECT t.amount, t.event_timestamp, e.embedding_text
            FROM transactions t
            JOIN transaction_embeddings e ON t.transaction_id = e.transaction_id
            ORDER BY e.embedding <-> :vector
            LIMIT :limit
        """)

        with self.engine.connect() as conn:
            results = conn.execute(sql, {
                "vector": str(query_vector),
                "limit": top_k
            }).mappings().all()
        return results

    def _build_prompt(self, query: str, context_results: list) -> str:
        context_block = "\n".join([
            f"- [{r.event_timestamp}] Amount: {r.amount} | Details: {r.embedding_text}"
            for r in context_results
        ])

        return f"""
        You are an expert Fraud Investigator. 
        Analyze the following real-time transactions retrieved from our Kafka stream.

        RELEVANT TRANSACTIONS:
        {context_block}

        INVESTIGATION QUERY: {query}

        Provide a concise risk assessment based ONLY on the provided context.
        """

    def generate_answer(self, user_query: str, top_k: int = 5):
        context = self._get_semantic_context(user_query, top_k)

        if not context:
            return "No relevant streaming data found for this query."

        prompt = self._build_prompt(user_query, context)

        response = openai.ChatCompletion.create(
            model=self.cfg.llm.model_name,
            messages=[{"role": "system", "content": "You are a helpful fraud detection assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content