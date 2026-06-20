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

    def _get_embedding(self, text: str) -> List[float]:
        """
        Convert TEXT to vector use OpenAI Embeddings
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[VectorMemory] Error getting embedding: {e}")
            raise

    def _prepare_episode_text(self, episode: Episode) -> str:
        """
        Prepare Episodic Text for Embeddings
        """
        text = f"""
            Puzzle: {episode.puzzle_text}

            Reflection: {episode.reflection}

            Outcome: {episode.outcome}
            """
        return text.strip()

    def _generate_document_id(self, episode: Episode) -> str:
        """
        Create ID for every document for ChromaDB
        """
        return f"episode_{episode.episode_id}"

    def add_episode(self, episode: Episode) -> None:
        """
        Add Episode in vectorDB
        """
        document_id = self._generate_document_id(episode)
        text = self._prepare_episode_text(episode)
        embedding = self._get_embedding(text)

        # Create metadata
        metadata = {
            "puzzle_id": episode.puzzle_id,
            "outcome": episode.outcome,
            "timestamp": episode.timestamp,
            "final_answer": episode.final_answer[:100],
        }

        # Add in ChromaDB
        self.collection.add(
            ids=[document_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

        print(
            f"[VectorMemory] Added episode {episode.episode_id[:8]}... (outcome: {episode.outcome})"
        )
