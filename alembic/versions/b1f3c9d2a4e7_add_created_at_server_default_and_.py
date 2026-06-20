"""add_created_at_server_default_and_event_timestamp_index

Revision ID: b1f3c9d2a4e7
Revises: a991223705af
Create Date: 2026-06-20 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b1f3c9d2a4e7'
down_revision: Union[str, Sequence[str], None] = 'a991223705af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('transactions', 'created_at', server_default=sa.text('now()'))
    op.alter_column('transaction_embeddings', 'created_at', server_default=sa.text('now()'))
    op.create_index(op.f('ix_transactions_event_timestamp'), 'transactions', ['event_timestamp'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_transactions_event_timestamp'), table_name='transactions')
    op.alter_column('transaction_embeddings', 'created_at', server_default=None)
    op.alter_column('transactions', 'created_at', server_default=None)
