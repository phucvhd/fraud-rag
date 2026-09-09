"""Push the fixed regression question set to Langfuse as a Dataset.

Run once (and again whenever the fixture changes). Items use stable ids, so
re-running updates in place rather than duplicating. The Dataset is the fixed
input side of regression: `run_regression_experiment.py` runs the agent over it
and scores each answer, so two runs (e.g. before/after a prompt change) are
comparable in the Langfuse UI.

    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... PYTHONPATH=. \
        python scripts/upload_regression_dataset.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load LANGFUSE_* from .env before get_client() reads the environment. This
# script does not import config_loader (which loads .env as a side effect), so
# it must do it itself, or auth silently fails with the keys "unset".
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langfuse import get_client  # noqa: E402

from shared.logging_config import configure_logging  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "test/fixtures/regression_dataset.json"


def main() -> int:
    configure_logging()
    client = get_client()
    if not client.auth_check():
        print("Langfuse auth failed — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.", file=sys.stderr)
        return 1

    spec = json.loads(FIXTURE.read_text())
    client.create_dataset(name=spec["name"], description=spec.get("description"))

    for item in spec["items"]:
        client.create_dataset_item(
            dataset_name=spec["name"],
            id=item["id"],  # stable id -> idempotent upsert
            input=item["input"],
            expected_output=item.get("expected_output"),
            metadata=item.get("metadata"),
        )

    client.flush()
    print(f"Uploaded {len(spec['items'])} items to dataset '{spec['name']}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
