from unittest.mock import patch

from services.monitoring.tracing import resolve_environment


def test_environment_falls_back_to_local_when_app_env_is_unset():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_environment() == "local"


def test_environment_follows_app_env():
    with patch.dict("os.environ", {"APP_ENV": "PROD"}, clear=True):
        # Langfuse only accepts lowercase environment names.
        assert resolve_environment() == "prod"


def test_blank_app_env_is_treated_as_unset():
    with patch.dict("os.environ", {"APP_ENV": "  "}, clear=True):
        assert resolve_environment() == "local"
