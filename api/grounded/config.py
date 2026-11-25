"""Settings, read from the environment so the same image runs anywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass

from grounded.embed import DEFAULT_MODEL
from grounded.ingest import DEFAULT_CHUNK_SIZE
from grounded.retrieve import DEFAULT_THRESHOLD, DEFAULT_TOP_K
from grounded.verify import DEFAULT_ENTAILMENT_THRESHOLD, DEFAULT_NLI_MODEL

DEFAULT_DATABASE_URL = "postgresql://grounded:grounded@127.0.0.1:5433/grounded"


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    embedding_model: str = DEFAULT_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    top_k: int = DEFAULT_TOP_K
    refusal_threshold: float = DEFAULT_THRESHOLD
    # "auto": Claude when ANTHROPIC_API_KEY is set, otherwise the extractive answer.
    answer_model: str = "auto"
    nli_model: str = DEFAULT_NLI_MODEL
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            database_url=env.get("GROUNDED_DATABASE_URL", DEFAULT_DATABASE_URL),
            embedding_model=env.get("GROUNDED_EMBEDDING_MODEL", DEFAULT_MODEL),
            chunk_size=int(env.get("GROUNDED_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            top_k=int(env.get("GROUNDED_TOP_K", DEFAULT_TOP_K)),
            refusal_threshold=float(env.get("GROUNDED_REFUSAL_THRESHOLD", DEFAULT_THRESHOLD)),
            answer_model=env.get("GROUNDED_ANSWER_MODEL", "auto"),
            nli_model=env.get("GROUNDED_NLI_MODEL", DEFAULT_NLI_MODEL),
            entailment_threshold=float(
                env.get("GROUNDED_ENTAILMENT_THRESHOLD", DEFAULT_ENTAILMENT_THRESHOLD)
            ),
        )
