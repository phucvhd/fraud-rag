from sqlalchemy import create_engine

from shared.config_loader import config_loader


class BaseRepository:
    def __init__(self):
        cfg = config_loader.load()
        self.engine = create_engine(cfg.database.url)
