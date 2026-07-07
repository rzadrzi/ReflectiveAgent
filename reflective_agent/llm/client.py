"""LLM client for interacting with OpenAI and Ollama APIs."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import APIError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from reflective_agent.utils.helpers import get_env_var
from reflective_agent.utils.logging import get_logger

logger = get_logger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    provider: str = Field(default="ollama", pattern="^(openai|ollama)$")
    model: str = Field(default="llama3.1:8b")
    base_url: str = Field(default="http://localhost:11434/v1")
    api_key: str = Field(default="ollama")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0.0)


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    total_tokens: int = Field(default=0)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request."""
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion from a single prompt."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, **kwargs)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        logger.info(f"Initialized OpenAI client with model: {self.config.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request to OpenAI."""
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        params.update(kwargs)

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Sending request to OpenAI (attempt {attempt + 1})")

                response = self.client.chat.completions.create(**params)

                content = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                result = LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=response.choices[0].finish_reason,
                    total_tokens=response.usage.total_tokens,
                )

                logger.debug(f"Received response: {result.total_tokens} tokens")
                return result

            except RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)

            except APIError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(f"API error: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        logger.error(f"All {self.config.max_retries} retries failed")
        raise last_error


class OllamaClient(BaseLLMClient):
    """Client for Ollama API (using OpenAI SDK)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=self.config.api_key,  # Ollama doesn't need a real key
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        logger.info(f"Initialized Ollama client with model: {self.config.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request to Ollama."""
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False,
        }
        # Remove parameters not supported by Ollama
        params.pop("frequency_penalty", None)
        params.pop("presence_penalty", None)
        params.update(kwargs)

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Sending request to Ollama (attempt {attempt + 1})")

                response = self.client.chat.completions.create(**params)

                content = response.choices[0].message.content or ""

                # Ollama may not provide detailed usage stats
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0)
                    if response.usage
                    else 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0)
                    if response.usage
                    else 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                    if response.usage
                    else 0,
                }

                result = LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=response.choices[0].finish_reason,
                    total_tokens=usage["total_tokens"],
                )

                logger.debug(f"Received response from Ollama")
                return result

            except Exception as e:
                last_error = e
                wait_time = self.config.retry_delay * (2**attempt)
                logger.warning(f"Ollama error: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

        logger.error(f"All {self.config.max_retries} retries failed")
        raise last_error


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(provider: Optional[str] = None) -> BaseLLMClient:
        """
        Create LLM client based on provider.

        Args:
            provider: "openai" or "ollama" (if None, reads from environment)

        Returns:
            Configured LLM client
        """
        # Determine provider
        if provider is None:
            provider = get_env_var("LLM_PROVIDER", "ollama")

        # Build config based on provider
        if provider == "openai":
            config = LLMConfig(
                provider="openai",
                model=get_env_var("OPENAI_MODEL", "gpt-4-turbo-preview"),
                base_url=get_env_var("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=get_env_var("OPENAI_API_KEY"),
                temperature=float(get_env_var("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(get_env_var("OPENAI_MAX_TOKENS", "2000")),
            )
            return OpenAIClient(config)

        elif provider == "ollama":
            config = LLMConfig(
                provider="ollama",
                model=get_env_var("OLLAMA_MODEL", "llama3.1:8b"),
                base_url=get_env_var("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=get_env_var("OLLAMA_API_KEY", "ollama"),
                temperature=float(get_env_var("OLLAMA_TEMPERATURE", "0.7")),
                max_tokens=int(get_env_var("OLLAMA_MAX_TOKENS", "2000")),
            )
            return OllamaClient(config)

        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Convenience function
def create_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    Create LLM client (convenience function).

    Args:
        provider: "openai" or "ollama" (if None, reads from environment)

    Returns:
        Configured LLM client
    """
    return LLMClientFactory.create_client(provider)
