"""add_top_shap_features_to_transactions

Revision ID: d385f9b2c7e1
Revises: c274e8a1b6d3
Create Date: 2026-09-08 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd385f9b2c7e1'
down_revision: Union[str, Sequence[str], None] = 'c274e8a1b6d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('transactions', sa.Column('top_shap_features', postgresql.JSONB, nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('transactions', 'top_shap_features')
