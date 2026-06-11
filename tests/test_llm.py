# from llm.prompt_builder import PromptBuilder
from click import prompt

from reflective_agent.llm.prompt_builder import PromptBuilder
from reflective_agent.llm.client import LLMClient
from reflective_agent.utils.helpers import load_yaml

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPT_FILE = BASE_DIR / "config" / "prompts.yaml"
prompts = load_yaml(str(PROMPT_FILE))

def llm_prompt_builder():
    print(prompts)
    # builder = PromptBuilder()


if __name__ == "__main__":
    llm_prompt_builder()