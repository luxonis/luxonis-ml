from .custom_base_model import CustomBaseModel
from ..enums import *

class Metadata(CustomBaseModel):
    name: str
    backbone: str
    platform: Platform
    version: float