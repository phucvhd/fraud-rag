"""add_fraud_probability_to_transactions

Revision ID: c274e8a1b6d3
Revises: b1f3c9d2a4e7
Create Date: 2026-09-08 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c274e8a1b6d3'
down_revision: Union[str, Sequence[str], None] = 'b1f3c9d2a4e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('transactions', sa.Column('fraud_probability', sa.DECIMAL(6, 5), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('transactions', 'fraud_probability')
