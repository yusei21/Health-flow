import logging
import re
from collections.abc import Sequence

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMResponseError, LLMUnavailableError
from app.llm.schemas import ChatMessage

logger = logging.getLogger(__name__)

# Reasoning models may prepend a thinking block even when JSON output is requested.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLMProvider:
    """Talks to Ollama through its OpenAI-compatible API.

    Transport retries (connection errors, 5xx, 429) are delegated to the OpenAI SDK;
    this class additionally retries a bounded number of times when the model returns
    output that fails schema validation.
    """

    def __init__(self, client: AsyncOpenAI, model: str, max_validation_retries: int = 1) -> None:
        self._client = client
        self._model = model
        self._max_validation_retries = max_validation_retries

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaLLMProvider":
        client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        return cls(client, settings.llm_model, settings.llm_max_retries)

    async def generate_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], output_model: type[T]
    ) -> T:
        attempts = 1 + self._max_validation_retries
        for attempt in range(1, attempts + 1):
            content = await self._complete(messages, output_model)
            try:
                return output_model.model_validate_json(_strip_thinking(content))
            except ValidationError as exc:
                logger.warning(
                    "llm_invalid_structured_output",
                    extra={"attempt": attempt, "error_count": exc.error_count()},
                )
        raise LLMResponseError(f"invalid structured output after {attempts} attempts")

    async def _complete(
        self, messages: Sequence[ChatMessage], output_model: type[BaseModel]
    ) -> str:
        payload: list[ChatCompletionMessageParam] = [
            {"role": message.role.value, "content": message.content}  # type: ignore[misc]
            for message in messages
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=0,
                reasoning_effort="none",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_model.__name__,
                        "schema": output_model.model_json_schema(),
                        "strict": True,
                    },
                },
            )
        except APIError as exc:
            raise LLMUnavailableError(type(exc).__name__) from exc
        if not response.choices or response.choices[0].message.content is None:
            raise LLMResponseError("empty completion")
        return response.choices[0].message.content


def _strip_thinking(content: str) -> str:
    return _THINK_BLOCK.sub("", content).strip()
