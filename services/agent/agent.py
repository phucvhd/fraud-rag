import logging

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from shared.config_loader import config_loader

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self):
        cfg = config_loader.load()
        self.client = ChatOpenAI(
            base_url=cfg.llm.base_url,
            api_key=SecretStr(cfg.llm.api_key),
            model=cfg.llm.model_name,
        )
        logger.info("LLM initialised: %s - %s", cfg.llm.model_name, cfg.llm.base_url)

    def get_client(self):
        return self.client
