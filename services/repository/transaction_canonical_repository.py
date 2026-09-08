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

    # Whitelisted so sort_by can be safely interpolated into the ORDER BY clause
    # (bind params can't parameterize identifiers).
    _SORT_COLUMNS = {"time": "event_timestamp", "amount": "amount", "status": "is_fraud"}

    def get_transactions(
        self,
        start_dt: datetime,
        end_dt: datetime,
        limit: int = 50,
        offset: int = 0,
        is_fraud: bool | None = None,
        search: str | None = None,
        sort_by: str = "time",
        sort_dir: str = "desc",
    ) -> tuple[list[dict], int]:
        column = self._SORT_COLUMNS.get(sort_by, "event_timestamp")
        direction = "ASC" if sort_dir == "asc" else "DESC"

        where = ["event_timestamp >= :start", "event_timestamp < :end"]
        params: dict = {"start": start_dt, "end": end_dt, "limit": limit, "offset": offset}
        if is_fraud is not None:
            where.append("is_fraud = :is_fraud")
            params["is_fraud"] = is_fraud
        if search:
            where.append("(transaction_id::text ILIKE :search OR data_source ILIKE :search)")
            params["search"] = f"%{search}%"
        where_clause = " AND ".join(where)

        rows_query = text(f"""
            SELECT
                transaction_id,
                event_timestamp,
                amount::float AS amount,
                is_fraud,
                data_source
            FROM transactions
            WHERE {where_clause}
            ORDER BY {column} {direction}
            LIMIT :limit OFFSET :offset
        """)
        count_query = text(f"""
            SELECT COUNT(*)::int AS total
            FROM transactions
            WHERE {where_clause}
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(rows_query, params).mappings().all()
            total = conn.execute(count_query, params).scalar_one()
        return [dict(r) for r in rows], total

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


# Backwards-compatible name kept for existing call sites/tests.
TransactionRepository = TransactionCanonicalRepository
