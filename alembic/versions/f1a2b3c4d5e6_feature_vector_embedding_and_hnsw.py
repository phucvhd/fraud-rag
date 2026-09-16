"""switch embedding to 29-dim feature vector + hnsw l2 index

Transactions are now embedded as the standardized [V1..V28, Amount] feature
vector (29 dims) instead of a 384-dim text embedding. Existing 384-dim vectors
are meaningless in the new space, so they are cleared and repopulated by the
embedding worker / scripts.backfill_embeddings (after scripts.fit_feature_scaler
has produced the scaler artifact). An HNSW index with vector_l2_ops backs the
query-by-example nearest-neighbour search.

Revision ID: f1a2b3c4d5e6
Revises: e4a6c1f8d902
Create Date: 2026-09-14 00:00:00.000000

"""
from typing import Sequence, Union

from pgvector.sqlalchemy import Vector

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e4a6c1f8d902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "ix_transaction_embeddings_embedding_hnsw"


def upgrade() -> None:
    # Truncate first so the NOT NULL column can be dropped and re-added on an
    # empty table (a dimension change is not an implicit cast in pgvector).
    op.execute("TRUNCATE TABLE transaction_embeddings")
    op.drop_column("transaction_embeddings", "embedding")
    op.add_column("transaction_embeddings", sa.Column("embedding", Vector(29), nullable=False))
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {_INDEX} "
        "ON transaction_embeddings USING hnsw (embedding vector_l2_ops)"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
    op.execute("TRUNCATE TABLE transaction_embeddings")
    op.drop_column("transaction_embeddings", "embedding")
    op.add_column("transaction_embeddings", sa.Column("embedding", Vector(384), nullable=False))
