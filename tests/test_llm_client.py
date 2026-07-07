"""Unit tests for LLM client module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from reflective_agent.llm.client import (
    LLMClientFactory,
    LLMConfig,
    LLMResponse,
    OllamaClient,
    OpenAIClient,
    create_llm_client,
)


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=1000,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_invalid_provider(self):
        """Test that invalid provider raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(provider="invalid_provider")

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

        # Invalid
        with pytest.raises(Exception):
            LLMConfig(temperature=-0.1)

        with pytest.raises(Exception):
            LLMConfig(temperature=2.1)


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_response_creation(self):
        """Test LLMResponse creation."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
            total_tokens=30,
        )
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.total_tokens == 30


class TestLLMClientFactory:
    """Tests for LLM client factory."""

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "llama3.1:8b",
            "OLLAMA_BASE_URL": "http://localhost:11434/v1",
            "OLLAMA_API_KEY": "ollama",
        },
    )
    def test_create_ollama_client(self):
        """Test factory creates Ollama client."""
        client = create_llm_client()
        assert isinstance(client, OllamaClient)
        assert client.config.provider == "ollama"

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-4",
        },
    )
    def test_create_openai_client(self):
        """Test factory creates OpenAI client."""
        client = create_llm_client()
        assert isinstance(client, OpenAIClient)
        assert client.config.provider == "openai"

    @patch.dict(os.environ, {"LLM_PROVIDER": "invalid"})
    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_llm_client()

    def test_explicit_provider_override(self):
        """Test explicit provider parameter overrides environment."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test"}):
            client = create_llm_client(provider="ollama")
            assert isinstance(client, OllamaClient)


class TestOllamaClient:
    """Tests for Ollama client."""

    def test_client_initialization(self):
        """Test Ollama client initialization."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        client = OllamaClient(config)
        assert client.config.model == "llama3.1:8b"
        assert client.client is not None

    @patch("reflective_agent.llm.client.OpenAI")
    def test_chat_success(self, mock_openai):
        """Test successful chat completion."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "llama3.1:8b"
        mock_response.usage = None  # Ollama may not return usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create client and test
        config = LLMConfig(provider="ollama", model="llama3.1:8b")
        client = OllamaClient(config)
        client.client = mock_client

        response = client.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Test response"
        assert response.model == "llama3.1:8b"
        assert response.finish_reason == "stop"

    @patch("reflective_agent.llm.client.OpenAI")
    def test_generate_method(self, mock_openai):
        """Test generate method constructs messages correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "llama3.1:8b"
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama3.1:8b")
        client = OllamaClient(config)
        client.client = mock_client

        response = client.generate(prompt="What is 2+2?", system_prompt="You are helpful.")

        # Verify messages structure
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"


class TestOpenAIClient:
    """Tests for OpenAI client."""

    def test_client_initialization(self):
        """Test OpenAI client initialization."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        client = OpenAIClient(config)
        assert client.config.model == "gpt-4"
        assert client.client is not None


# ============================================================================
# Integration Tests (require actual Ollama server running)
# Run with: pytest tests/test_llm_client.py -m integration
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)
class TestOllamaIntegration:
    """Integration tests requiring actual Ollama server."""

    def test_ollama_connection(self):
        """Test actual connection to Ollama."""
        client = create_llm_client(provider="ollama")
        response = client.generate(
            prompt="Say 'Hello' in exactly one word.", system_prompt="You are a helpful assistant."
        )
        assert response.content
        assert len(response.content) > 0
        print(f"\n[Ollama Response]: {response.content}")
