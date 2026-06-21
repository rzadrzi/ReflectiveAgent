from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from reflective_agent.memory.episodic_memory import Episode
from reflective_agent.memory.vector_memory import VectorMemory


# ==============================================================================
# 1. Unit Tests with Mocked OpenAI API
# ==============================================================================
class TestVectorMemory:
    """Tests for VectorMemory class with mocked OpenAI embeddings."""

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
    def vector_memory(self, tmp_path, mock_openai_client):
        """Create a VectorMemory instance with mocked OpenAI."""
        db_path = tmp_path / "vector_db"
        return VectorMemory(
            collection_name="test_collection",
            persist_directory=str(db_path),
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            puzzle_id="puzzle_001",
            puzzle_text="If all A's are B's and all B's are C's, are all A's C's?",
            reasoning_path="Using transitivity...",
            final_answer="Yes",
            outcome="success",
            reflection="My reasoning was correct.",
        )

    @pytest.fixture
    def sample_failure_episode(self):
        """Create a sample failed episode for testing."""
        return Episode(
            puzzle_id="puzzle_002",
            puzzle_text="If some B's are C's...",
            reasoning_path="I assumed all B's are C's...",
            final_answer="Wrong answer",
            outcome="failure",
            reflection="I made a mistake in my assumption.",
        )

    def test_initialization(self, vector_memory, tmp_path):
        """Test that VectorMemory initializes correctly."""
        assert vector_memory.collection_name == "test_collection"
        assert vector_memory.persist_directory.exists()
        assert vector_memory.collection is not None
        assert vector_memory.collection.count() == 0

    def test_add_episode(self, vector_memory, sample_episode, mock_openai_client):
        """Test adding an episode to vector memory."""
        vector_memory.add_episode(sample_episode)

        # Verify episode was added
        assert vector_memory.collection.count() == 1

        # Verify embedding was called
        assert mock_openai_client.embeddings.create.called

        # Verify document was stored
        doc_id = f"episode_{sample_episode.episode_id}"
        result = vector_memory.collection.get(ids=[doc_id])
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["puzzle_id"] == "puzzle_001"
        assert result["metadatas"][0]["outcome"] == "success"

    def test_add_multiple_episodes(
        self, vector_memory, sample_episode, sample_failure_episode
    ):
        """Test adding multiple episodes."""
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)

        assert vector_memory.collection.count() == 2

    def test_search_similar(
        self, vector_memory, sample_episode, sample_failure_episode
    ):
        """Test semantic search for similar episodes."""
        # Add episodes
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)

        # Search for similar episodes
        query = "If all X's are Y's..."
        results = vector_memory.search_similar(query_text=query, n_results=2)

        # Should return results
        assert len(results) > 0
        assert "document_id" in results[0]
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "similarity_score" in results[0]
        assert "distance" in results[0]

    def test_search_similar_with_filter(
        self, vector_memory, sample_episode, sample_failure_episode
    ):
        """Test semantic search with outcome filter."""
        # Add episodes
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)

        # Search only for failures
        query = "Logic puzzle"
        results = vector_memory.search_similar(
            query_text=query, n_results=5, filter_outcome="failure"
        )

        # All results should be failures
        for result in results:
            assert result["metadata"]["outcome"] == "failure"

    def test_search_similar_failures(
        self, vector_memory, sample_episode, sample_failure_episode
    ):
        """Test searching for similar failures specifically."""
        # Add episodes
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)

        # Search for similar failures
        query = "Some B's are C's"
        results = vector_memory.search_similar_failures(query_text=query, n_results=3)

        # All results should be failures
        for result in results:
            assert result["metadata"]["outcome"] == "failure"

    def test_get_episode_by_id(self, vector_memory, sample_episode):
        """Test retrieving a specific episode by ID."""
        vector_memory.add_episode(sample_episode)

        result = vector_memory.get_episode_by_id(sample_episode.episode_id)

        assert result is not None
        assert result["document_id"] == f"episode_{sample_episode.episode_id}"
        assert result["metadata"]["puzzle_id"] == "puzzle_001"

    def test_get_episode_by_id_not_found(self, vector_memory):
        """Test retrieving non-existent episode."""
        result = vector_memory.get_episode_by_id("non-existent-id")

        assert result is None

    def test_delete_episode(self, vector_memory, sample_episode):
        """Test deleting an episode from vector memory."""
        vector_memory.add_episode(sample_episode)
        assert vector_memory.collection.count() == 1

        vector_memory.delete_episode(sample_episode.episode_id)

        assert vector_memory.collection.count() == 0

    def test_clear(self, vector_memory, sample_episode, sample_failure_episode):
        """Test clearing all vector memory."""
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)
        assert vector_memory.collection.count() == 2

        vector_memory.clear()

        assert vector_memory.collection.count() == 0

    def test_get_stats(self, vector_memory, sample_episode, sample_failure_episode):
        """Test getting memory statistics."""
        vector_memory.add_episode(sample_episode)
        vector_memory.add_episode(sample_failure_episode)

        stats = vector_memory.get_stats()

        assert stats["total_episodes"] == 2
        assert stats["collection_name"] == "test_collection"
        assert "persist_directory" in stats

    def test_persistence(self, tmp_path, sample_episode):
        """Test that data persists across instances."""
        db_path = tmp_path / "vector_db"

        # First instance: add episode
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            vm1 = VectorMemory(
                collection_name="test_collection", persist_directory=str(db_path)
            )
            vm1.add_episode(sample_episode)

        # Second instance: should load existing data
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            vm2 = VectorMemory(
                collection_name="test_collection", persist_directory=str(db_path)
            )

            # Should have the episode from first instance
            assert vm2.collection.count() == 1

    def test_embedding_error_handling(self, tmp_path, sample_episode):
        """Test handling of embedding API errors."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Simulate API error
            mock_client.embeddings.create.side_effect = Exception("API Error")

            vm = VectorMemory(
                collection_name="test_collection",
                persist_directory=str(tmp_path / "vector_db"),
            )

            # Should raise exception
            with pytest.raises(Exception):
                vm.add_episode(sample_episode)

    def test_prepare_episode_text(self, vector_memory, sample_episode):
        """Test text preparation for embedding."""
        text = vector_memory._prepare_episode_text(sample_episode)

        # Should contain puzzle text and reflection
        assert "If all A's are B's" in text
        assert "My reasoning was correct" in text
        assert "success" in text

    def test_generate_document_id(self, vector_memory, sample_episode):
        """Test document ID generation."""
        doc_id = vector_memory._generate_document_id(sample_episode)

        assert doc_id == f"episode_{sample_episode.episode_id}"
        assert doc_id.startswith("episode_")


# ==============================================================================
# 2. Integration Tests
# ==============================================================================
class TestVectorMemoryIntegration:
    """Integration tests for real-world scenarios."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch("reflective_agent.memory.vector_memory.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock embeddings response
            mock_response = Mock()
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536
            mock_response.data = [mock_embedding_data]
            mock_client.embeddings.create.return_value = mock_response

            yield mock_client

    @pytest.fixture
    def vector_memory(self, tmp_path, mock_openai_client):
        """Create a VectorMemory instance."""
        return VectorMemory(
            collection_name="test_collection",
            persist_directory=str(tmp_path / "vector_db"),
        )

    def test_full_workflow(self, vector_memory):
        """Test complete workflow: add, search, delete."""
        # 1. Add multiple episodes
        for i in range(5):
            episode = Episode(
                puzzle_id=f"puzzle_{i:03d}",
                puzzle_text=f"Logic puzzle number {i}",
                reasoning_path=f"Reasoning {i}",
                final_answer=f"Answer {i}",
                outcome="success" if i % 2 == 0 else "failure",
                reflection=f"Reflection {i}",
            )
            vector_memory.add_episode(episode)

        # 2. Verify all added
        assert vector_memory.collection.count() == 5

        # 3. Search for similar episodes
        results = vector_memory.search_similar(query_text="Logic puzzle", n_results=3)
        assert len(results) > 0

        # 4. Search for failures only
        failures = vector_memory.search_similar_failures(
            query_text="Logic puzzle", n_results=3
        )
        for failure in failures:
            assert failure["metadata"]["outcome"] == "failure"

        # 5. Get stats
        stats = vector_memory.get_stats()
        assert stats["total_episodes"] == 5

        # 6. Clear all
        vector_memory.clear()
        assert vector_memory.collection.count() == 0

    def test_search_with_empty_memory(self, vector_memory):
        """Test searching when memory is empty."""
        results = vector_memory.search_similar(query_text="Any query", n_results=5)

        assert len(results) == 0

    def test_metadata_filtering(self, vector_memory):
        """Test that metadata is correctly stored and filtered."""
        # Add episodes with different outcomes
        success_episode = Episode(
            puzzle_id="success_001",
            puzzle_text="Success puzzle",
            reasoning_path="Good reasoning",
            final_answer="Correct",
            outcome="success",
            reflection="Did well",
        )

        failure_episode = Episode(
            puzzle_id="failure_001",
            puzzle_text="Failure puzzle",
            reasoning_path="Bad reasoning",
            final_answer="Wrong",
            outcome="failure",
            reflection="Made mistake",
        )

        vector_memory.add_episode(success_episode)
        vector_memory.add_episode(failure_episode)

        # Search with filter
        failures = vector_memory.search_similar(
            query_text="puzzle", n_results=5, filter_outcome="failure"
        )

        # All results should be failures
        for result in failures:
            assert result["metadata"]["outcome"] == "failure"
            assert result["metadata"]["puzzle_id"] == "failure_001"


# ==============================================================================
# 3. Edge Cases and Error Handling
# ==============================================================================
class TestVectorMemoryEdgeCases:
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
    def vector_memory(self, tmp_path, mock_openai_client):
        """Create a VectorMemory instance."""
        return VectorMemory(
            collection_name="test_collection",
            persist_directory=str(tmp_path / "vector_db"),
        )

    def test_duplicate_episode_addition(self, vector_memory):
        """Test adding the same episode twice."""
        episode = Episode(
            puzzle_id="puzzle_001",
            puzzle_text="Test puzzle",
            reasoning_path="Test reasoning",
            final_answer="Test answer",
            outcome="success",
            reflection="Test reflection",
        )

        # Add first time
        vector_memory.add_episode(episode)
        assert vector_memory.collection.count() == 1

        # Add second time (should update or create duplicate)
        vector_memory.add_episode(episode)

        # ChromaDB will update the existing document
        assert vector_memory.collection.count() == 1

    def test_long_text_handling(self, vector_memory):
        """Test handling of very long text."""
        long_text = "A" * 10000  # Very long text

        episode = Episode(
            puzzle_id="puzzle_001",
            puzzle_text=long_text,
            reasoning_path="Reasoning",
            final_answer="Answer",
            outcome="success",
            reflection="Reflection",
        )

        # Should not crash
        vector_memory.add_episode(episode)
        assert vector_memory.collection.count() == 1

    def test_special_characters_in_text(self, vector_memory):
        """Test handling of special characters."""
        episode = Episode(
            puzzle_id="puzzle_001",
            puzzle_text="Test with special chars: !@#$%^&*()_+{}|:<>?",
            reasoning_path="Reasoning with emojis: 🎉🚀",
            final_answer="Answer",
            outcome="success",
            reflection="Reflection with unicode: 你好世界",
        )

        # Should not crash
        vector_memory.add_episode(episode)
        assert vector_memory.collection.count() == 1

    def test_delete_nonexistent_episode(self, vector_memory):
        """Test deleting an episode that doesn't exist."""
        # Should not crash
        vector_memory.delete_episode("non-existent-id")

    def test_search_with_zero_results(self, vector_memory):
        """Test searching when no results match."""
        episode = Episode(
            puzzle_id="puzzle_001",
            puzzle_text="Sudoku puzzle",
            reasoning_path="Reasoning",
            final_answer="Answer",
            outcome="success",
            reflection="Reflection",
        )

        vector_memory.add_episode(episode)

        # Search for something completely different
        results = vector_memory.search_similar(query_text="Chess game", n_results=5)

        # May still return results due to semantic similarity
        # but should not crash
        assert isinstance(results, list)
