"""partial btree index on transactions.amount for high-value filters

The structured amount-filter paths (find_known_fraud / find_suspected_fraud /
context_lookup listing) filter transactions.amount, which had no index at all.
A partial index over the high-value band directly targets the common
"over 1000 EUR" query (the original filter bug) without indexing the whole
low-value long tail. The planner uses it when it can prove amount_min >= 1000
(custom/literal plans); lower or open-ended filters fall back to a seq scan as
before.

Revision ID: a7c3e1f9b2d4
Revises: f1a2b3c4d5e6
Create Date: 2026-09-14 00:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a7c3e1f9b2d4'
down_revision: Union[str, Sequence[str], None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "ix_transactions_amount_high"


def upgrade() -> None:
    op.create_index(
        _INDEX,
        "transactions",
        ["amount"],
        postgresql_where=sa.text("amount >= 1000"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="transactions")
