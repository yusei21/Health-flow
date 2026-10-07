from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from app.llm.schemas import ChatMessage


class LLMProvider(Protocol):
    """Provider-agnostic structured generation.

    Implementations must return an instance of `output_model` or raise a subclass of
    `app.core.exceptions.LLMError`; they must never return unvalidated text.
    """

    async def generate_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], output_model: type[T]
    ) -> T: ...
