import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .episodic_memory import Episode


# ==============================================================================
# 1. Vector DB Manager Class
# ==============================================================================
class VectorMemory:
    """
    Vector memory extractor for semantic search of past experiences.
    Uses ChromaDB for storage and OpenAI Embeddings for text-to-vector conversion.
    """

    def __init__(
        self,
        collection_name: str = "agent_episodes",
        persist_directory: str = "data/memory/vector_db",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Episodic memory for self-improving LLM agent"},
        )

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(
            f"[VectorMemory] Initialized with collection '{collection_name}' ({self.collection.count()} episodes)"
        )
