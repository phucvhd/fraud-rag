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
            fraud_probability=data["fraud_probability"],
            top_shap_features=data["top_shap_features"],
            features=data["features"],
            data_source=data["data_source"],
        ).on_conflict_do_nothing(index_elements=["transaction_id"])
        with self.engine.begin() as conn:
            conn.execute(stmt)

    # Whitelisted so sort_by can be safely interpolated into the ORDER BY clause
    # (bind params can't parameterize identifiers). Prefixed with t. since the
    # query below joins against transaction_status, which also has an id column.
    _SORT_COLUMNS = {
        "time": "t.event_timestamp",
        "amount": "t.amount",
        "status": "t.is_fraud",
        "risk": "t.fraud_probability",
    }

    def get_transactions(
        self,
        start_dt: datetime,
        end_dt: datetime,
        limit: int = 50,
        offset: int = 0,
        is_fraud: bool | None = None,
        pipeline_status: str | None = None,
        search: str | None = None,
        sort_by: str = "time",
        sort_dir: str = "desc",
    ) -> tuple[list[dict], int]:
        column = self._SORT_COLUMNS.get(sort_by, "t.event_timestamp")
        direction = "ASC" if sort_dir == "asc" else "DESC"

        where = ["t.event_timestamp >= :start", "t.event_timestamp < :end"]
        params: dict = {"start": start_dt, "end": end_dt, "limit": limit, "offset": offset}
        if is_fraud is not None:
            where.append("t.is_fraud = :is_fraud")
            params["is_fraud"] = is_fraud
        if pipeline_status is not None:
            where.append("ts.status = :pipeline_status")
            params["pipeline_status"] = pipeline_status
        if search:
            where.append("(t.transaction_id::text ILIKE :search OR t.data_source ILIKE :search)")
            params["search"] = f"%{search}%"
        where_clause = " AND ".join(where)

        rows_query = text(f"""
            SELECT
                t.transaction_id,
                t.event_timestamp,
                t.amount::float AS amount,
                t.is_fraud,
                t.fraud_probability::float AS fraud_probability,
                t.data_source,
                ts.status
            FROM transactions t
            LEFT JOIN transaction_status ts ON ts.transaction_id = t.transaction_id
            WHERE {where_clause}
            ORDER BY {column} {direction}
            LIMIT :limit OFFSET :offset
        """)
        count_query = text(f"""
            SELECT COUNT(*)::int AS total
            FROM transactions t
            LEFT JOIN transaction_status ts ON ts.transaction_id = t.transaction_id
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
