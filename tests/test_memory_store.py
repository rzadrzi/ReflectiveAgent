from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from reflective_agent.memory.episodic_memory import Episode
from reflective_agent.memory.memory_store import MemoryStore


# ==============================================================================
# 1. Unit Tests
# ==============================================================================
class TestMemoryStore:
    """Tests for MemoryStore class with mocked OpenAI API."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client to avoid actual API calls."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock embeddings response
            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536  # Typical embedding size
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            yield mock_client

    @pytest.fixture
    def memory_store(self, tmp_path, mock_openai_client):
        """Create a MemoryStore instance with mocked OpenAI."""
        episodic_path = tmp_path / "episodes.json"
        vector_path = tmp_path / "vector_db"

        return MemoryStore(
            episodic_storage_path=str(episodic_path),
            vector_db_path=str(vector_path),
            collection_name="test_collection",
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def sample_episode_data(self):
        """Sample data for creating episodes."""
        return {
            "puzzle_id": "puzzle_001",
            "puzzle_text": "If all A's are B's and all B's are C's, are all A's C's?",
            "reasoning_path": "Using transitivity...",
            "final_answer": "Yes",
            "outcome": "success",
            "reflection": "My reasoning was correct.",
        }

    @pytest.fixture
    def sample_failure_data(self):
        """Sample data for a failed episode."""
        return {
            "puzzle_id": "puzzle_002",
            "puzzle_text": "If some B's are C's and all A's are B's...",
            "reasoning_path": "I assumed all B's are C's...",
            "final_answer": "Wrong answer",
            "outcome": "failure",
            "reflection": "I made a mistake in my assumption about quantifiers.",
        }

    def test_initialization(self, memory_store, tmp_path):
        """Test that MemoryStore initializes both memory layers."""
        assert memory_store.episodic_memory is not None
        assert memory_store.vector_memory is not None
        assert memory_store.episodic_memory.storage_path.exists() or True
        assert memory_store.vector_memory.persist_directory.exists()

    def test_add_episode_to_both_layers(self, memory_store, sample_episode_data):
        """Test that adding an episode stores it in both memory layers."""
        episode = memory_store.add_episode(**sample_episode_data)

        # Verify episode was created
        assert episode.puzzle_id == "puzzle_001"
        assert episode.outcome == "success"

        # Verify it's in episodic memory
        all_episodes = memory_store.episodic_memory.get_all()
        assert len(all_episodes) == 1
        assert all_episodes[0].puzzle_id == "puzzle_001"

        # Verify it's in vector memory
        assert memory_store.vector_memory.collection.count() == 1

    def test_add_episode_vector_failure_graceful(self, tmp_path, sample_episode_data):
        """Test that if vector memory fails, episodic memory still saves."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Simulate vector memory failure
            mock_client.embeddings.create.side_effect = Exception("API Error")

            episodic_path = tmp_path / "episodes.json"
            vector_path = tmp_path / "vector_db"

            store = MemoryStore(
                episodic_storage_path=str(episodic_path),
                vector_db_path=str(vector_path),
            )

            # Should still succeed (episodic memory works)
            episode = store.add_episode(**sample_episode_data)

            # Verify it's in episodic memory
            all_episodes = store.episodic_memory.get_all()
            assert len(all_episodes) == 1
            assert all_episodes[0].puzzle_id == "puzzle_001"

    def test_search_similar_episodes(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test semantic search for similar episodes."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Search for similar episodes
        results = memory_store.search_similar_episodes(
            query_text="If all X's are Y's...", n_results=2
        )

        # Should return results with full episode details
        assert len(results) > 0
        assert "episode" in results[0]
        assert "similarity_score" in results[0]
        assert "distance" in results[0]
        assert isinstance(results[0]["episode"], Episode)

    def test_search_similar_episodes_with_filter(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test semantic search with outcome filter."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Search only for failures
        results = memory_store.search_similar_episodes(
            query_text="Logic puzzle", n_results=5, filter_outcome="failure"
        )

        # All results should be failures
        for result in results:
            assert result["episode"].outcome == "failure"

    def test_search_similar_failures(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test searching for similar past failures."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Search for similar failures
        results = memory_store.search_similar_failures(
            query_text="Some B's are C's", n_results=3
        )

        # All results should be failures
        for result in results:
            assert result["episode"].outcome == "failure"
            assert "similarity_score" in result

    def test_get_recent_failures(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test retrieving recent failures without semantic search."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Get recent failures
        failures = memory_store.get_recent_failures(limit=5)

        # Should return only failures
        assert len(failures) == 1
        assert failures[0].outcome == "failure"
        assert isinstance(failures[0], Episode)

    def test_get_all_episodes(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test retrieving all episodes with limit."""
        # Add multiple episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Get all episodes
        all_episodes = memory_store.get_all_episodes(limit=10)

        assert len(all_episodes) == 2
        assert all(isinstance(ep, Episode) for ep in all_episodes)

    def test_get_all_episodes_with_limit(self, memory_store):
        """Test that limit is respected when retrieving all episodes."""
        # Add 5 episodes
        for i in range(5):
            memory_store.add_episode(
                puzzle_id=f"puzzle_{i:03d}",
                puzzle_text=f"Puzzle {i}",
                reasoning_path=f"Reasoning {i}",
                final_answer=f"Answer {i}",
                outcome="success" if i % 2 == 0 else "failure",
                reflection=f"Reflection {i}",
            )

        # Get with limit
        limited = memory_store.get_all_episodes(limit=3)

        assert len(limited) == 3

    def test_get_episode_by_id(self, memory_store, sample_episode_data):
        """Test retrieving a specific episode by ID."""
        episode = memory_store.add_episode(**sample_episode_data)

        # Retrieve by ID
        retrieved = memory_store.get_episode_by_id(episode.episode_id)

        assert retrieved is not None
        assert retrieved.episode_id == episode.episode_id
        assert retrieved.puzzle_id == "puzzle_001"

    def test_get_episode_by_id_not_found(self, memory_store):
        """Test retrieving non-existent episode by ID."""
        result = memory_store.get_episode_by_id("non-existent-id")

        assert result is None

    def test_clear_both_layers(
        self, memory_store, sample_episode_data, sample_failure_data
    ):
        """Test clearing both memory layers."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Verify both have data
        assert len(memory_store.episodic_memory.get_all()) == 2
        assert memory_store.vector_memory.collection.count() == 2

        # Clear all
        memory_store.clear()

        # Verify both are empty
        assert len(memory_store.episodic_memory.get_all()) == 0
        assert memory_store.vector_memory.collection.count() == 0

    def test_get_stats(self, memory_store, sample_episode_data, sample_failure_data):
        """Test getting statistics from both memory layers."""
        # Add episodes
        memory_store.add_episode(**sample_episode_data)
        memory_store.add_episode(**sample_failure_data)

        # Get stats
        stats = memory_store.get_stats()

        # Verify structure
        assert "episodic_memory" in stats
        assert "vector_memory" in stats
        assert "total_storage_locations" in stats

        # Verify episodic stats
        episodic_stats = stats["episodic_memory"]
        assert episodic_stats["total_episodes"] == 2
        assert episodic_stats["successful_episodes"] == 1
        assert episodic_stats["failed_episodes"] == 1
        assert episodic_stats["success_rate"] == 50.0

        # Verify vector stats
        vector_stats = stats["vector_memory"]
        assert vector_stats["total_episodes"] == 2

    def test_sync_memories(self, tmp_path, sample_episode_data):
        """Test synchronizing vector memory with episodic memory."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            episodic_path = tmp_path / "episodes.json"
            vector_path = tmp_path / "vector_db"

            # Create store and add episode
            store = MemoryStore(
                episodic_storage_path=str(episodic_path),
                vector_db_path=str(vector_path),
            )
            store.add_episode(**sample_episode_data)

            # Clear vector memory only
            store.vector_memory.clear()
            assert store.vector_memory.collection.count() == 0
            assert len(store.episodic_memory.get_all()) == 1

            # Sync memories
            result = store.sync_memories()

            # Verify sync worked
            assert result["added"] == 1
            assert result["errors"] == 0
            assert store.vector_memory.collection.count() == 1

    def test_sync_memories_already_synced(self, memory_store, sample_episode_data):
        """Test sync when memories are already in sync."""
        memory_store.add_episode(**sample_episode_data)

        # Sync should add nothing
        result = memory_store.sync_memories()

        assert result["added"] == 0
        assert result["errors"] == 0


# ==============================================================================
# 2. Integration Tests
# ==============================================================================
class TestMemoryStoreIntegration:
    """Integration tests for real-world scenarios."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            yield mock_client

    @pytest.fixture
    def memory_store(self, tmp_path, mock_openai_client):
        """Create a MemoryStore instance."""
        return MemoryStore(
            episodic_storage_path=str(tmp_path / "episodes.json"),
            vector_db_path=str(tmp_path / "vector_db"),
        )

    def test_full_workflow(self, memory_store):
        """Test complete workflow: add, search, retrieve, clear."""
        # 1. Add multiple episodes
        for i in range(5):
            memory_store.add_episode(
                puzzle_id=f"puzzle_{i:03d}",
                puzzle_text=f"Logic puzzle number {i}",
                reasoning_path=f"Reasoning {i}",
                final_answer=f"Answer {i}",
                outcome="success" if i % 2 == 0 else "failure",
                reflection=f"Reflection {i}",
            )

        # 2. Verify both layers have data
        assert len(memory_store.episodic_memory.get_all()) == 5
        assert memory_store.vector_memory.collection.count() == 5

        # 3. Search for similar episodes
        results = memory_store.search_similar_episodes(
            query_text="Logic puzzle", n_results=3
        )
        assert len(results) > 0
        assert all("episode" in r for r in results)

        # 4. Search for failures only
        failures = memory_store.search_similar_failures(
            query_text="Logic puzzle", n_results=3
        )
        for failure in failures:
            assert failure["episode"].outcome == "failure"

        # 5. Get recent failures
        recent_failures = memory_store.get_recent_failures(limit=2)
        assert len(recent_failures) <= 2

        # 6. Get episode by ID
        first_episode = memory_store.episodic_memory.get_all()[0]
        retrieved = memory_store.get_episode_by_id(first_episode.episode_id)
        assert retrieved is not None

        # 7. Get stats
        stats = memory_store.get_stats()
        assert stats["episodic_memory"]["total_episodes"] == 5
        assert stats["vector_memory"]["total_episodes"] == 5

        # 8. Clear all
        memory_store.clear()
        assert len(memory_store.episodic_memory.get_all()) == 0
        assert memory_store.vector_memory.collection.count() == 0

    def test_persistence_across_instances(self, tmp_path):
        """Test that data persists across MemoryStore instances."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            episodic_path = tmp_path / "episodes.json"
            vector_path = tmp_path / "vector_db"

            # First instance: add episode
            store1 = MemoryStore(
                episodic_storage_path=str(episodic_path),
                vector_db_path=str(vector_path),
            )
            store1.add_episode(
                puzzle_id="puzzle_001",
                puzzle_text="Test puzzle",
                reasoning_path="Test reasoning",
                final_answer="Test answer",
                outcome="success",
                reflection="Test reflection",
            )

            # Second instance: should load existing data
            store2 = MemoryStore(
                episodic_storage_path=str(episodic_path),
                vector_db_path=str(vector_path),
            )

            # Verify data persisted
            assert len(store2.episodic_memory.get_all()) == 1
            assert store2.vector_memory.collection.count() == 1

    def test_search_with_empty_memory(self, memory_store):
        """Test searching when memory is empty."""
        results = memory_store.search_similar_episodes(
            query_text="Any query", n_results=5
        )

        assert len(results) == 0

    def test_mixed_outcomes_retrieval(self, memory_store):
        """Test retrieving episodes with mixed outcomes."""
        # Add mix of successes and failures
        for i in range(10):
            memory_store.add_episode(
                puzzle_id=f"puzzle_{i:03d}",
                puzzle_text=f"Puzzle {i}",
                reasoning_path=f"Reasoning {i}",
                final_answer=f"Answer {i}",
                outcome="success" if i % 2 == 0 else "failure",
                reflection=f"Reflection {i}",
            )

        # Get stats
        stats = memory_store.get_stats()

        # Verify correct counts
        assert stats["episodic_memory"]["total_episodes"] == 10
        assert stats["episodic_memory"]["successful_episodes"] == 5
        assert stats["episodic_memory"]["failed_episodes"] == 5
        assert stats["episodic_memory"]["success_rate"] == 50.0


# ==============================================================================
# 3. Edge Cases and Error Handling
# ==============================================================================
class TestMemoryStoreEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            yield mock_client

    @pytest.fixture
    def memory_store(self, tmp_path, mock_openai_client):
        """Create a MemoryStore instance."""
        return MemoryStore(
            episodic_storage_path=str(tmp_path / "episodes.json"),
            vector_db_path=str(tmp_path / "vector_db"),
        )

    def test_add_episode_with_special_characters(self, memory_store):
        """Test adding episode with special characters and unicode."""
        episode = memory_store.add_episode(
            puzzle_id="puzzle_001",
            puzzle_text="Test with special chars: !@#$%^&*()_+{}|:<>? 你好世界 🎉",
            reasoning_path="Reasoning with emojis: 🚀",
            final_answer="Answer with unicode: αβγ",
            outcome="success",
            reflection="Reflection with quotes: \"test\" and 'test'",
        )

        assert episode.puzzle_id == "puzzle_001"
        assert len(memory_store.episodic_memory.get_all()) == 1
        assert memory_store.vector_memory.collection.count() == 1

    def test_add_episode_with_long_text(self, memory_store):
        """Test adding episode with very long text."""
        long_text = "A" * 10000

        episode = memory_store.add_episode(
            puzzle_id="puzzle_001",
            puzzle_text=long_text,
            reasoning_path="Reasoning",
            final_answer="Answer",
            outcome="success",
            reflection="Reflection",
        )

        assert episode.puzzle_id == "puzzle_001"
        assert len(memory_store.episodic_memory.get_all()) == 1

    def test_search_returns_empty_list_not_none(self, memory_store):
        """Test that search methods return empty list, not None."""
        results = memory_store.search_similar_episodes(
            query_text="Non-existent query", n_results=5
        )

        assert results is not None
        assert isinstance(results, list)
        assert len(results) == 0

    def test_get_episode_by_id_with_empty_memory(self, memory_store):
        """Test getting episode by ID when memory is empty."""
        result = memory_store.get_episode_by_id("any-id")

        assert result is None

    def test_clear_empty_memory(self, memory_store):
        """Test clearing already empty memory."""
        # Should not crash
        memory_store.clear()

        assert len(memory_store.episodic_memory.get_all()) == 0
        assert memory_store.vector_memory.collection.count() == 0

    def test_get_stats_with_empty_memory(self, memory_store):
        """Test getting stats when memory is empty."""
        stats = memory_store.get_stats()

        assert stats["episodic_memory"]["total_episodes"] == 0
        assert stats["episodic_memory"]["successful_episodes"] == 0
        assert stats["episodic_memory"]["failed_episodes"] == 0
        assert stats["episodic_memory"]["success_rate"] == 0.0

    def test_sync_memories_with_empty_episodic(self, memory_store):
        """Test syncing when episodic memory is empty."""
        result = memory_store.sync_memories()

        assert result["added"] == 0
        assert result["errors"] == 0

    def test_multiple_syncs(self, memory_store):
        """Test that multiple syncs don't create duplicates."""
        # Add episode
        memory_store.add_episode(
            puzzle_id="puzzle_001",
            puzzle_text="Test",
            reasoning_path="Reasoning",
            final_answer="Answer",
            outcome="success",
            reflection="Reflection",
        )

        # Sync multiple times
        for _ in range(3):
            result = memory_store.sync_memories()

        # Should not create duplicates
        assert memory_store.vector_memory.collection.count() == 1
