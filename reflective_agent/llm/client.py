import os
from typing import Optional

class LLMClient:
    def __init__(
            self,
            model_name:str="gpt-4o-mini",
            temperature:float=0.2,
            api_key:Optional[str]=None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("LLM_API_KEY")

    def generate(self, prompt:str)->str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        return f"[Mock LLM response for prompt]: {prompt[:100]}"