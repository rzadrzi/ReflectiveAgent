import uuid
from datetime import datetime
from typing import Dict, Any
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
