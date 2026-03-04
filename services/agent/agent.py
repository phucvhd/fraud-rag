from langchain_openai import ChatOpenAI
from shared.config_loader import config_loader
from pydantic import SecretStr

class LLMAgent:
    def __init__(self):
        self.cfg = config_loader.load()
        self.client = ChatOpenAI(
            base_url=self.cfg.llm.base_url,
            api_key=SecretStr("lm-studio"),
            model="deepseek-r1-distill-qwen-7b"
        )
        self._check_llm_connection()

    def _check_llm_connection(self):
        try:
            model_name = self.client.model_name
            print(f"LLM Connection successful: {model_name} - {self.cfg.llm.base_url}")
        except Exception as e:
            print(f"Failed to connect to LLM at {self.cfg.llm.base_url}")
            print(f"Error detail: {e}")

    def get_client(self):
        return self.client