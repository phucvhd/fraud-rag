from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from shared.config_loader import config_loader


@lru_cache(maxsize=None)
def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


class BaseRepository:
    def __init__(self) -> None:
        cfg = config_loader.load()
        self.engine: Engine = get_engine(cfg.database.url)
