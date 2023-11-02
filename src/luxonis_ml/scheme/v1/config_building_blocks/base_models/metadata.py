from .custom_base_model import CustomBaseModel
from ..enums import *

class Metadata(BaseModel):
    name: str
    backbone: str
    platform: Platform
    version: float