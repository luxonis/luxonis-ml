from pydantic import BaseModel, ConfigDict


class BaseModelExtraForbid(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="forbid")
