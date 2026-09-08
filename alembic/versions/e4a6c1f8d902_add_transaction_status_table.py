"""add_transaction_status_table

Revision ID: e4a6c1f8d902
Revises: d385f9b2c7e1
Create Date: 2026-09-08 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e4a6c1f8d902'
down_revision: Union[str, Sequence[str], None] = 'd385f9b2c7e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'transaction_status',
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('status', sa.VARCHAR(20), nullable=False),
        sa.Column('received_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('flagged_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('embedding_started_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('embedded_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('transaction_status')
