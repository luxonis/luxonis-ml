from pydantic import BaseModel, Extra


class CustomBaseModel(BaseModel):
    class Config:
        extra = Extra.forbid  # forbid extra fields
