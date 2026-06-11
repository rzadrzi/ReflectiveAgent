import pytest
from reflective_agent.llm.client import LLMClient

def test_mock_llm_client():
    # response = "answer"
    client = LLMClient( model_name="gpt-4o-mini",
            temperature=0.2,
            api_key=None)

    print(client.generate("I am Reza"))