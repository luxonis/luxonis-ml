from pydantic import BaseModel
from ..enums import *

class Metadata(BaseModel):
    name: str
    backbone: str
    platform: Platform
    version: int