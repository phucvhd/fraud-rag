import os

from sentence_transformers import SentenceTransformer

from shared.config_loader import config_loader


class SentenceTransformerModel:
    def __init__(self):
        self.cfg = config_loader.load()
        self.model = SentenceTransformer(
            self.cfg.embedding.model_name,
            cache_folder=os.getenv("HF_HOME"),
        )

    def get_model(self):
        return self.model