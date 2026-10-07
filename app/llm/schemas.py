from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
