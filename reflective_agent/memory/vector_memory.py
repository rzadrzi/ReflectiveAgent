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

    def search_similar(
        self, query_text: str, n_results: int = 5, filter_outcome: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search similarity based on query text

        Args:
            query_text: puzzle text or new question
            n_results: number of result
            filter_outcome: ('success' or 'failure')

        Returns:
            Dict of {episode_id, metadata, similarity_score}
        """
        # query to embedding
        query_embedding = self._get_embedding(query_text)

        where_filter = None
        if filter_outcome:
            where_filter = {"outcome": filter_outcome}

        # Search in vectorDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        similar_episodes = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity_score = 1 - (distance / 2)  # Convert to 0, 1

                similar_episodes.append(
                    {
                        "document_id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity_score,
                        "distance": distance,
                    }
                )

        return similar_episodes

    def search_similar_failures(
        self, query_text: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for failure
        """
        return self.search_similar(
            query_text=query_text, n_results=n_results, filter_outcome="failure"
        )

    def get_episode_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        get an episode by ID
        """
        document_id = f"episode_{episode_id}"
        try:
            result = self.collection.get(
                ids=[document_id], include=["documents", "metadatas"]
            )

            if result["ids"]:
                return {
                    "document_id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            print(f"[VectorMemory] Error retrieving episode: {e}")

        return None

    def delete_episode(self, episode_id: str) -> None:
        """
        Delete an epsode by ID
        """
        document_id = f"episode_{episode_id}"
        try:
            self.collection.delete(ids=[document_id])
            print(f"[VectorMemory] Deleted episode {episode_id[:8]}...")
        except Exception as e:
            print(f"[VectorMemory] Error deleting episode: {e}")

    def clear(self) -> None:
        """
        Clear memory
        """
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Episodic memory for self-improving LLM agent"},
        )
        print("[VectorMemory] Vector memory cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get status of vectorDB
        """
        return {
            "total_episodes": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }
