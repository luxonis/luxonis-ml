from pydantic import BaseModel

from ..enums import Platform


class Metadata(BaseModel):
    """Represents metadata of a model.

    @type name: str
    @ivar name: Name of the model.
    @type platform: Platform
    @ivar platform: Luxonis hardware platform for which the model was exported (e.g.
        'rvc4').
    """

    name: str
    platform: Platform
