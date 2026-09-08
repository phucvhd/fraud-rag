import logging
from datetime import datetime, timezone

from sqlalchemy import text

from services.repository.base import BaseRepository

logger = logging.getLogger(__name__)

# Whitelisted so the timestamp column can be safely interpolated into the SQL
# (bind params can't parameterize identifiers). Only reached via the four
# mark_*() methods below, never with caller-supplied input.
_STATUS_COLUMNS = {
    "received": "received_at",
    "flagged": "flagged_at",
    "embedding": "embedding_started_at",
    "embedded": "embedded_at",
}


class TransactionStatusRepository(BaseRepository):
    def _mark_many(self, transaction_ids: list[str], status: str) -> None:
        if not transaction_ids:
            return
        timestamp_column = _STATUS_COLUMNS[status]
        now = datetime.now(timezone.utc)

        stmt = text(f"""
            INSERT INTO transaction_status (transaction_id, status, {timestamp_column}, updated_at)
            VALUES (:transaction_id, :status, :ts, :ts)
            ON CONFLICT (transaction_id) DO UPDATE SET
                status = EXCLUDED.status,
                {timestamp_column} = EXCLUDED.{timestamp_column},
                updated_at = EXCLUDED.updated_at
        """)
        params = [{"transaction_id": tx_id, "status": status, "ts": now} for tx_id in transaction_ids]
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt, params)
        except Exception:
            # Status tracking is observability, not the pipeline itself — a
            # Postgres hiccup here must not re-trigger reprocessing of work
            # (e.g. re-embedding) that already completed successfully.
            logger.error("Failed to mark %d transaction(s) as %s", len(transaction_ids), status, exc_info=True)

    def mark_received(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "received")

    def mark_flagged(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "flagged")

    def mark_embedding(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "embedding")

    def mark_embedded(self, transaction_ids: list[str]) -> None:
        self._mark_many(transaction_ids, "embedded")

    def get_status_counts(self, start_dt: datetime, end_dt: datetime) -> dict[str, int]:
        """How many transactions that entered the pipeline in this window are
        currently sitting at each stage — a nonzero count at 'flagged' or
        'embedding' (for a window that's otherwise fully in the past) means
        something is stuck there.
        """
        stmt = text("""
            SELECT status, COUNT(*)::int AS count
            FROM transaction_status
            WHERE received_at >= :start AND received_at < :end
            GROUP BY status
        """)
        counts = dict.fromkeys(_STATUS_COLUMNS, 0)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt, {"start": start_dt, "end": end_dt}).mappings().all()
        for row in rows:
            if row["status"] in counts:
                counts[row["status"]] = row["count"]
        return counts
