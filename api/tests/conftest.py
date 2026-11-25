from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

DEMO_DOCS = Path(__file__).resolve().parents[2] / "demo" / "docs"

# Testcontainers' clean-up container (Ryuk) can't start under Docker Desktop on macOS: it mounts
# the Docker socket, which lives where the VM can't reach. The fixture below stops its own
# container, so Ryuk is only a safety net; Linux and CI keep it.
if sys.platform == "darwin":
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


@pytest.fixture
def demo_docs() -> Path:
    return DEMO_DOCS


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A throwaway PostgreSQL with pgvector, started once for the whole run."""

    def unavailable(reason: str) -> None:
        # Locally a missing Docker skips these tests; in CI it must fail, not hide them.
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        unavailable("testcontainers isn't installed")
    container = PostgresContainer("pgvector/pgvector:pg17", driver=None)
    try:
        container.start()
    except Exception as error:  # Docker not running, image not pullable
        unavailable(f"PostgreSQL container unavailable: {error}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()
