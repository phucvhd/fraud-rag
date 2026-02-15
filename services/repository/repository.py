from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from shared.config_loader import config_loader
from schemas.transaction import TransactionCanonical
from database.model import TransactionModel


class TransactionRepository:
    def __init__(self):
        self.cfg = config_loader.load("config/application.yaml")
        self.engine = create_engine(self.cfg.database.url)

    def insert_if_not_exists(self, transaction: TransactionCanonical):
        data = transaction.model_dump()

        stmt = insert(TransactionModel).values(
            transaction_id=data['transaction_id'],
            event_time_seconds=data['event_time_seconds'],
            event_timestamp=data['event_timestamp'],
            amount=data['amount'],
            is_fraud=data['is_fraud'],
            features=data['features'],
            data_source=data['data_source']
        )

        stmt = stmt.on_conflict_do_nothing(
            index_elements=['transaction_id']
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)