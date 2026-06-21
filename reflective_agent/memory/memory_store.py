from pathlib import Path
from typing import Any, Dict, List, Optional

from .episodic_memory import Episode, EpisodicMemory
from .vector_memory import VectorMemory


# ==============================================================================
# Unified Memory Store (Facade Pattern)
# ==============================================================================
class MemoryStore:
    """
    Unified abstraction layer for memory management.

    This class coordinates between two memory layers:
    - EpisodicMemory: Structured storage with JSON
    - VectorMemory: Semantic search with ChromaDB

    When an episode is added, it's stored in both layers.
    When searching, it uses Vector for semantic similarity but retrieves
    full details from Episodic.
    """

    def __init__(
        self,
        episodic_storage_path: str = "data/memory/episodes.json",
        vector_db_path: str = "data/memory/vector_db",
        collection_name: str = "agent_episodes",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize both memory layers.

        Args:
            episodic_storage_path: Path to JSON file for episodic memory
            vector_db_path: Path to directory for ChromaDB
            collection_name: Collection name in ChromaDB
            embedding_model: Embedding model for vector conversion
        """
        print("[MemoryStore] Initializing memory layers...")

        # Initialize episodic memory (JSON-based)
        self.episodic_memory = EpisodicMemory(storage_path=episodic_storage_path)

        # Initialize vector memory (ChromaDB-based)
        self.vector_memory = VectorMemory(
            collection_name=collection_name,
            persist_directory=vector_db_path,
            embedding_model=embedding_model,
        )

        print("[MemoryStore] Memory layers initialized successfully.")

    def add_episode(
        self,
        puzzle_id: str,
        puzzle_text: str,
        reasoning_path: str,
        final_answer: str,
        outcome: str,
        reflection: str,
    ) -> Episode:
        """
        Add a new episode to both memory layers.

        This method:
        1. Creates an Episode object
        2. Stores it in episodic memory (JSON) for structured retrieval
        3. Stores it in vector memory (ChromaDB) for semantic search

        Args:
            puzzle_id: Unique identifier for the puzzle
            puzzle_text: Complete puzzle text
            reasoning_path: Agent's reasoning steps
            final_answer: Final answer provided by agent
            outcome: Result ('success' or 'failure')
            reflection: Agent's self-reflection on why it succeeded/failed

        Returns:
            The created Episode object
        """
        # Create Episode object
        episode = Episode(
            puzzle_id=puzzle_id,
            puzzle_text=puzzle_text,
            reasoning_path=reasoning_path,
            final_answer=final_answer,
            outcome=outcome,
            reflection=reflection,
        )

        # Store in episodic memory (JSON)
        self.episodic_memory.add_episode(
            puzzle_id=puzzle_id,
            puzzle_text=puzzle_text,
            reasoning_path=reasoning_path,
            final_answer=final_answer,
            outcome=outcome,
            reflection=reflection,
        )

        # Store in vector memory (ChromaDB)
        try:
            self.vector_memory.add_episode(episode)
        except Exception as e:
            print(f"[MemoryStore] Warning: Failed to add to vector memory: {e}")
            print("[MemoryStore] Episode saved in episodic memory only.")

        return episode

    def search_similar_episodes(
        self, query_text: str, n_results: int = 5, filter_outcome: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar episodes using semantic search.

        This method:
        1. Uses VectorMemory to find semantically similar episodes
        2. Retrieves full details from EpisodicMemory
        3. Combines the results

        Args:
            query_text: The new puzzle text
            n_results: Number of similar episodes to retrieve
            filter_outcome: Filter by outcome ('success' or 'failure')

        Returns:
            List of dictionaries containing full episode + similarity_score
        """
        # Search in vector memory
        vector_results = self.vector_memory.search_similar(
            query_text=query_text, n_results=n_results, filter_outcome=filter_outcome
        )

        if not vector_results:
            return []

        # Retrieve full details from episodic memory
        enriched_results = []
        all_episodes = self.episodic_memory.get_all()

        # Build dictionary for fast lookup by episode_id
        episode_dict = {ep.episode_id: ep for ep in all_episodes}

        for result in vector_results:
            # Extract episode_id from document_id
            doc_id = result["document_id"]
            episode_id = doc_id.replace("episode_", "")

            # Retrieve full details
            if episode_id in episode_dict:
                full_episode = episode_dict[episode_id]
                enriched_results.append(
                    {
                        "episode": full_episode,
                        "similarity_score": result["similarity_score"],
                        "distance": result["distance"],
                    }
                )

        return enriched_results

    def search_similar_failures(
        self, query_text: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past failures using semantic search.

        This method is used to help the agent avoid repeating past mistakes.

        Args:
            query_text: The new puzzle text
            n_results: Number of similar failures to retrieve

        Returns:
            List of similar failed episodes with full details
        """
        return self.search_similar_episodes(
            query_text=query_text, n_results=n_results, filter_outcome="failure"
        )

    def get_recent_failures(self, limit: int = 5) -> List[Episode]:
        """
        Retrieve the most recent failures (without semantic search).

        This method is useful for identifying recurring error patterns.

        Args:
            limit: Number of recent failures to retrieve

        Returns:
            List of recent failed episodes
        """
        return self.episodic_memory.get_failures(limit=limit)

    def get_all_episodes(self, limit: int = 100) -> List[Episode]:
        """
        Retrieve all episodes with a limit.

        Args:
            limit: Maximum number of episodes to retrieve

        Returns:
            List of episodes
        """
        all_episodes = self.episodic_memory.get_all()
        return all_episodes[:limit]

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """
        Retrieve a specific episode by its ID.

        Args:
            episode_id: Unique identifier of the episode

        Returns:
            Episode object if found, None otherwise
        """
        all_episodes = self.episodic_memory.get_all()
        for ep in all_episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def clear(self) -> None:
        """
        Clear all memory layers.

        Warning: This operation is irreversible!
        """
        print("[MemoryStore] Clearing all memory layers...")
        self.episodic_memory.clear()
        self.vector_memory.clear()
        print("[MemoryStore] All memory cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about both memory layers.

        Returns:
            Dictionary containing memory statistics
        """
        all_episodes = self.episodic_memory.get_all()
        total = len(all_episodes)
        successes = sum(1 for ep in all_episodes if ep.outcome.lower() == "success")
        failures = total - successes

        episodic_stats = {
            "total_episodes": total,
            "successful_episodes": successes,
            "failed_episodes": failures,
            "success_rate": (successes / total * 100) if total > 0 else 0.0,
            "storage_type": "JSON (Episodic)",
            "storage_path": str(self.episodic_memory.storage_path),
        }

        vector_stats = self.vector_memory.get_stats()

        return {
            "episodic_memory": episodic_stats,
            "vector_memory": vector_stats,
            "total_storage_locations": 2,
        }

    def sync_memories(self) -> Dict[str, int]:
        """
        Synchronize vector memory with episodic memory.

        This method is useful when vector memory is empty or corrupted
        but episodic memory still has data.

        Returns:
            Dictionary with number of episodes added and errors
        """
        print("[MemoryStore] Syncing vector memory with episodic memory...")

        all_episodes = self.episodic_memory.get_all()
        added = 0
        errors = 0

        for episode in all_episodes:
            # Check if this episode exists in vector memory
            doc_id = f"episode_{episode.episode_id}"
            try:
                existing = self.vector_memory.collection.get(ids=[doc_id])
                if not existing["ids"]:
                    # Episode not in vector memory, add it
                    self.vector_memory.add_episode(episode)
                    added += 1
            except Exception as e:
                errors += 1
                print(
                    f"[MemoryStore] Error syncing episode {episode.episode_id[:8]}: {e}"
                )

        print(f"[MemoryStore] Sync complete: {added} added, {errors} errors")
        return {"added": added, "errors": errors}
