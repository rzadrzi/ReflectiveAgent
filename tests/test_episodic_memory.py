import json
from pathlib import Path

import pytest

from reflective_agent.memory.episodic_memory import Episode, EpisodicMemory


# ==============================================================================
# 1. Data Model Tests (Episode)
# ==============================================================================
class TestEpisode:
    """Tests for the Pydantic Episode model."""

    def test_episode_creation(self):
        """Test creating an episode with all required fields."""
        episode = Episode(
            puzzle_id="puzzle_001",
            puzzle_text="If A > B and B > C...",
            reasoning_path="I assume A is the largest...",
            final_answer="A",
            outcome="success",
            reflection="My reasoning was correct.",
        )

        assert episode.puzzle_id == "puzzle_001"
        assert episode.outcome == "success"
        assert episode.episode_id  # Should be auto-generated UUID
        assert episode.timestamp  # Should be auto-generated timestamp

    def test_episode_to_dict(self):
        """Test converting episode to dictionary."""
        episode = Episode(
            puzzle_id="puzzle_002",
            puzzle_text="Test puzzle",
            reasoning_path="Test reasoning",
            final_answer="Test answer",
            outcome="failure",
            reflection="Test reflection",
        )

        data = episode.to_dict()

        assert isinstance(data, dict)
        assert data["puzzle_id"] == "puzzle_002"
        assert data["outcome"] == "failure"
        assert "episode_id" in data
        assert "timestamp" in data

    def test_episode_missing_required_field(self):
        """Test that missing required fields raise an error."""
        with pytest.raises(Exception):
            Episode(
                puzzle_id="puzzle_003",
                puzzle_text="Test puzzle",
                # reasoning_path is missing
                final_answer="Test answer",
                outcome="success",
                reflection="Test reflection",
            )


# ==============================================================================
# 2. EpisodicMemory Class Tests
# ==============================================================================
class TestEpisodicMemory:
    """Tests for the EpisodicMemory management class."""

    @pytest.fixture
    def memory(self, tmp_path):
        """
        Fixture to create a clean EpisodicMemory instance in a temporary directory.
        tmp_path is a temporary directory that pytest cleans up after tests.
        """
        db_path = tmp_path / "test_episodes.json"
        return EpisodicMemory(storage_path=str(db_path))

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

    def test_initialization_creates_directory(self, tmp_path):
        """Test that the directory is created if it doesn't exist."""
        nested_path = tmp_path / "nested" / "dir" / "episodes.json"
        memory = EpisodicMemory(storage_path=str(nested_path))

        assert nested_path.parent.exists()
        assert isinstance(memory.episodes, list)
        assert len(memory.episodes) == 0

    def test_add_episode(self, memory, sample_episode_data):
        """Test adding a single episode."""
        episode = memory.add_episode(**sample_episode_data)

        assert episode.puzzle_id == "puzzle_001"
        assert episode.outcome == "success"
        assert len(memory.episodes) == 1
        assert memory.episodes[0].puzzle_id == "puzzle_001"

    def test_add_multiple_episodes(self, memory):
        """Test adding multiple episodes."""
        for i in range(5):
            memory.add_episode(
                puzzle_id=f"puzzle_{i:03d}",
                puzzle_text=f"Puzzle number {i}",
                reasoning_path="Reasoning path",
                final_answer=f"Answer {i}",
                outcome="success" if i % 2 == 0 else "failure",
                reflection="Reflection",
            )

        assert len(memory.episodes) == 5

    def test_persistence_save_and_load(self, tmp_path, sample_episode_data):
        """Test data persistence: save and reload."""
        db_path = tmp_path / "episodes.json"

        # First time: add episode
        memory1 = EpisodicMemory(storage_path=str(db_path))
        memory1.add_episode(**sample_episode_data)

        # Second time: create new instance and verify auto-loading
        memory2 = EpisodicMemory(storage_path=str(db_path))

        assert len(memory2.episodes) == 1
        assert memory2.episodes[0].puzzle_id == "puzzle_001"
        assert memory2.episodes[0].reflection == "My reasoning was correct."

    def test_get_failures(self, memory):
        """Test retrieving only failed episodes."""
        # Add mix of successes and failures
        memory.add_episode(
            puzzle_id="p1",
            puzzle_text="Test 1",
            reasoning_path="r",
            final_answer="a",
            outcome="success",
            reflection="ref",
        )
        memory.add_episode(
            puzzle_id="p2",
            puzzle_text="Test 2",
            reasoning_path="r",
            final_answer="a",
            outcome="failure",
            reflection="ref",
        )
        memory.add_episode(
            puzzle_id="p3",
            puzzle_text="Test 3",
            reasoning_path="r",
            final_answer="a",
            outcome="failure",
            reflection="ref",
        )

        failures = memory.get_failures()

        assert len(failures) == 2
        assert all(ep.outcome == "failure" for ep in failures)

    def test_get_failures_with_limit(self, memory):
        """Test limit on number of returned failures."""
        for i in range(10):
            memory.add_episode(
                puzzle_id=f"p{i}",
                puzzle_text="Test",
                reasoning_path="r",
                final_answer="a",
                outcome="failure",
                reflection="ref",
            )

        failures = memory.get_failures(limit=3)

        assert len(failures) == 3

    def test_get_failures_ordered_by_timestamp(self, memory):
        """Test that failures are ordered by timestamp (newest first)."""
        import time

        memory.add_episode(
            puzzle_id="old",
            puzzle_text="Old puzzle",
            reasoning_path="r",
            final_answer="a",
            outcome="failure",
            reflection="ref",
        )
        time.sleep(0.01)  # Ensure timestamp difference
        memory.add_episode(
            puzzle_id="new",
            puzzle_text="New puzzle",
            reasoning_path="r",
            final_answer="a",
            outcome="failure",
            reflection="ref",
        )

        failures = memory.get_failures()

        assert failures[0].puzzle_id == "new"
        assert failures[1].puzzle_id == "old"

    def test_get_similar_episodes_by_keyword(self, memory):
        """Test searching episodes by keyword."""
        memory.add_episode(
            puzzle_id="p1",
            puzzle_text="Sudoku with numbers 1 to 9",
            reasoning_path="r",
            final_answer="a",
            outcome="success",
            reflection="Solved",
        )
        memory.add_episode(
            puzzle_id="p2",
            puzzle_text="Logic puzzle",
            reasoning_path="r",
            final_answer="a",
            outcome="failure",
            reflection="Mistake in sudoku",
        )
        memory.add_episode(
            puzzle_id="p3",
            puzzle_text="Chess game",
            reasoning_path="r",
            final_answer="a",
            outcome="success",
            reflection="Won",
        )

        results = memory.get_similar_episodes(keyword="sudoku")

        assert len(results) == 2
        assert all(
            "sudoku" in ep.puzzle_text.lower() or "sudoku" in ep.reflection.lower()
            for ep in results
        )

    def test_get_similar_episodes_case_insensitive(self, memory):
        """Test that search is case-insensitive."""
        memory.add_episode(
            puzzle_id="p1",
            puzzle_text="SUDOKU puzzle",
            reasoning_path="r",
            final_answer="a",
            outcome="success",
            reflection="ref",
        )

        results = memory.get_similar_episodes(keyword="sudoku")

        assert len(results) == 1

    def test_get_similar_episodes_with_limit(self, memory):
        """Test limit on keyword search results."""
        for i in range(5):
            memory.add_episode(
                puzzle_id=f"p{i}",
                puzzle_text="sudoku test",
                reasoning_path="r",
                final_answer="a",
                outcome="success",
                reflection="ref",
            )

        results = memory.get_similar_episodes(keyword="sudoku", limit=2)

        assert len(results) == 2

    def test_get_all(self, memory):
        """Test retrieving all episodes."""
        for i in range(3):
            memory.add_episode(
                puzzle_id=f"p{i}",
                puzzle_text="Test",
                reasoning_path="r",
                final_answer="a",
                outcome="success",
                reflection="ref",
            )

        all_episodes = memory.get_all()

        assert len(all_episodes) == 3

    def test_clear(self, memory, sample_episode_data):
        """Test clearing all memory."""
        memory.add_episode(**sample_episode_data)
        assert len(memory.episodes) == 1
        assert memory.storage_path.exists()

        memory.clear()

        assert len(memory.episodes) == 0
        assert not memory.storage_path.exists()

    def test_load_corrupted_file(self, tmp_path, capsys):
        """Test handling of corrupted JSON file."""
        db_path = tmp_path / "corrupted.json"
        # Write invalid content
        db_path.write_text("This is not valid JSON {{{", encoding="utf-8")

        memory = EpisodicMemory(storage_path=str(db_path))

        # Should print error but not crash
        captured = capsys.readouterr()
        assert "EpisodicMemory" in captured.out
        assert len(memory.episodes) == 0

    def test_json_file_format(self, memory, sample_episode_data, tmp_path):
        """Test that JSON file is saved with correct format."""
        memory.add_episode(**sample_episode_data)

        # Read file directly
        with open(memory.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["puzzle_id"] == "puzzle_001"
        assert "episode_id" in data[0]
        assert "timestamp" in data[0]


# ==============================================================================
# 3. Integration Tests
# ==============================================================================
class TestEpisodicMemoryIntegration:
    """Integration tests for real-world scenarios."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: add, retrieve, clear."""
        db_path = tmp_path / "workflow.json"
        memory = EpisodicMemory(storage_path=str(db_path))

        # 1. Add multiple episodes
        memory.add_episode(
            puzzle_id="p1",
            puzzle_text="Simple sudoku",
            reasoning_path="r1",
            final_answer="a1",
            outcome="success",
            reflection="Easy",
        )
        memory.add_episode(
            puzzle_id="p2",
            puzzle_text="Hard sudoku",
            reasoning_path="r2",
            final_answer="a2",
            outcome="failure",
            reflection="Mistake in row 3",
        )
        memory.add_episode(
            puzzle_id="p3",
            puzzle_text="Logic puzzle",
            reasoning_path="r3",
            final_answer="a3",
            outcome="failure",
            reflection="Forgot one condition",
        )

        # 2. Retrieve failures
        failures = memory.get_failures()
        assert len(failures) == 2

        # 3. Search by keyword
        sudoku_episodes = memory.get_similar_episodes("sudoku")
        assert len(sudoku_episodes) == 2

        # 4. Reload (simulate restart)
        memory_reloaded = EpisodicMemory(storage_path=str(db_path))
        assert len(memory_reloaded.episodes) == 3

        # 5. Clear
        memory_reloaded.clear()
        assert len(memory_reloaded.episodes) == 0
