import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# ==============================================================================
# 1. Data Model
# ==============================================================================
class Episode(BaseModel):

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="episode id")
    puzzle_id: str = Field(..., description="puzzle id")
    puzzle_text: str = Field(..., description="complete puzzle text")
    reasoning_path: str = Field(..., description="reasoning path")
    final_answer: str = Field(..., description="final answer")
    outcome: str = Field(..., description="outcome: success or failure")
    reflection: str = Field(..., description="why success or failure")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="time of experience")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EpisodicMemory:
    def __init__(self, storage_path: str = "data/memory/episodes.json"):
        self.storage_path = Path(storage_path)
        self.episodes: List[Episode] = []
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.episodes = [Episode(**item) for item in data]
                print(f"[EpisodicMemory] {len(self.episodes)}")
            except Exception as e:
                print(f"[EpisodicMemory] {e}")
                self.episodes = []

    def _save_to_disk(self):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:

                json_data = [ep.to_dict() for ep in self.episodes]
                json.dump(json_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[EpisodicMemory] {e}")

    def add_episode(
            self,
            puzzle_id: str,
            puzzle_text: str,
            reasoning_path: str,
            final_answer: str,
            outcome: str,
            reflection: str
    ) -> Episode:
        new_episode = Episode(
            puzzle_id=puzzle_id,
            puzzle_text=puzzle_text,
            reasoning_path=reasoning_path,
            final_answer=final_answer,
            outcome=outcome,
            reflection=reflection
        )
        self.episodes.append(new_episode)
        self._save_to_disk()  # (Persistence)
        return new_episode

    def get_failures(self, limit: int = 5) -> List[Episode]:
        failures = [ep for ep in self.episodes if ep.outcome.lower() == 'failure']

        failures.sort(key=lambda x: x.timestamp, reverse=True)
        return failures[:limit]

    def get_similar_episodes(self, keyword: str, limit: int = 3) -> List[Episode]:
        keyword = keyword.lower()
        matched = [
            ep for ep in self.episodes
            if keyword in ep.puzzle_text.lower() or keyword in ep.reflection.lower()
        ]
        return matched[:limit]

    def get_all(self) -> List[Episode]:
        return self.episodes

    def clear(self):
        self.episodes = []
        if self.storage_path.exists():
            self.storage_path.unlink()
        print("[EpisodicMemory]")
