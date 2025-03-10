from typing import Any, Dict

from pydantic import Field, field_validator

from luxonis_ml.data.utils import ImageType, MediaType
from luxonis_ml.utils import BaseModelExtraForbid


class LuxonisComponent(BaseModelExtraForbid):
    """Abstraction for a piece of media within a source. Most commonly,

    this abstracts an image sensor.
    @type name: str
    @ivar name: A recognizable name for the component.
    @type media_type: L{MediaType}
    @ivar media_type: Enum for the type of media for the component.
    @type image_type: L{ImageType}
    @ivar image_type: Enum for the image type. Only used if
        C{media_type==MediaType.IMAGE}.
    """

    name: str
    media_type: MediaType = MediaType.IMAGE
    image_type: ImageType = ImageType.COLOR


class LuxonisSource(BaseModelExtraForbid):
    """Abstracts the structure of a dataset and which components/media
    are included.

    For example, with an U{OAK-D<https://docs.luxonis.com/projects/hardware
    /en/latest/pages/BW1098OAK/>}, you can have a source with 4 image
    components: C{rgb} (color), C{left} (mono), C{right} (mono), and C{depth}.

    @type name: str
    @ivar name: A recognizable name for the source. Defaults to "default".

    @type components: Optional[List[LuxonisComponent]]
    @ivar components: If not using the default configuration, a list of
    L{LuxonisComponent} to group together in the source.

    @type main_component: Optional[str]
    @ivar main_component: The name of the component that should be
        primarily visualized.
    """

    name: str = "default"
    components: Dict[str, LuxonisComponent] = Field(
        default_factory=lambda: {"default": LuxonisComponent(name="default")}
    )
    main_component: str = "default"

    def merge_with(self, other: "LuxonisSource") -> "LuxonisSource":
        """Merge two sources together.

        @type other: LuxonisSource
        @param other: The other source to merge with.
        @rtype: LuxonisSource
        @return: A new source with the components of both sources.
        """
        components = self.components.copy()
        components.update(other.components)
        return LuxonisSource(
            name=self.name,
            components=components,
            main_component=self.main_component,
        )

    @field_validator("components", mode="before")
    @classmethod
    def validate_components(cls, components: Any) -> Any:
        if isinstance(components, list):
            migrated_components = {}
            for component in components:
                component = LuxonisComponent(**component)
                migrated_components[component.name] = component
            return migrated_components
        return components
