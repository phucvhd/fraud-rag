from datetime import datetime

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from database.model import TransactionModel
from schemas.transaction import TransactionCanonical
from services.repository.base import BaseRepository


class TransactionCanonicalRepository(BaseRepository):
    def insert_if_not_exists(self, transaction: TransactionCanonical) -> None:
        data = transaction.model_dump()
        stmt = insert(TransactionModel).values(
            transaction_id=data["transaction_id"],
            event_time_seconds=data["event_time_seconds"],
            event_timestamp=data["event_timestamp"],
            amount=data["amount"],
            is_fraud=data["is_fraud"],
            features=data["features"],
            data_source=data["data_source"],
        ).on_conflict_do_nothing(index_elements=["transaction_id"])
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_timeseries(self, start_dt: datetime, end_dt: datetime) -> list[dict]:
        query = text("""
            SELECT
                to_char(date_trunc('minute', event_timestamp), 'HH24:MI') AS bucket,
                COUNT(*)::int                                               AS transactions,
                SUM(CASE WHEN is_fraud     THEN 1 ELSE 0 END)::int         AS fraud,
                SUM(CASE WHEN NOT is_fraud THEN 1 ELSE 0 END)::int         AS normal
            FROM transactions
            WHERE event_timestamp >= :start
              AND event_timestamp <  :end
            GROUP BY date_trunc('minute', event_timestamp)
            ORDER BY date_trunc('minute', event_timestamp)
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"start": start_dt, "end": end_dt}).mappings().all()
        return [dict(r) for r in rows]


TransactionRepository = TransactionCanonicalRepository
