import os
import tempfile
from pathlib import Path

# Settings are read at import time, so configure the environment first.
_tmp = tempfile.mkdtemp()
os.environ["SPAM_API_KEYS"] = "test-key,other-key"
os.environ["SPAM_FEEDBACK_PATH"] = str(Path(_tmp) / "feedback.jsonl")
os.environ["SPAM_RATE_LIMIT_DEMO"] = "5/minute"
os.environ["SPAM_RATE_LIMIT_API"] = "1000/minute"
os.environ["SPAM_MAX_TEXT_CHARS"] = "20000"

import pytest
from fastapi.testclient import TestClient

from app.main import app, limiter


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}
