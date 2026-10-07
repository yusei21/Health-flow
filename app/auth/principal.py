from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
