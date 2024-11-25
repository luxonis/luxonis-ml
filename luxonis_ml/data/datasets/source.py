from dataclasses import dataclass
from typing import List, Optional, TypedDict

from ..utils.enums import ImageType, MediaType


@dataclass
class LuxonisComponent:
    """Abstraction for a piece of media within a source. Most commonly,
    this abstracts an image sensor.

    @type name: str
    @param name: A recognizable name for the component.
    @type media_type: L{MediaType}
    @param media_type: Enum for the type of media for the component.
    @type image_type: L{ImageType}
    @param image_type: Enum for the image type. Only used if
        C{media_type==MediaType.IMAGE}.
    """

    name: str
    media_type: MediaType = MediaType.IMAGE
    image_type: ImageType = ImageType.COLOR

    class LuxonisComponentDocument(TypedDict):
        name: str
        media_type: str
        image_type: str

    def to_document(self) -> LuxonisComponentDocument:
        return {
            "name": self.name,
            "media_type": self.media_type.value,
            "image_type": self.image_type.value,
        }

    @classmethod
    def from_document(
        cls, document: LuxonisComponentDocument
    ) -> "LuxonisComponent":
        if document["image_type"] is not None:
            return cls(
                name=document["name"],
                media_type=MediaType(document["media_type"]),
                image_type=ImageType(document["image_type"]),
            )
        else:
            return cls(
                name=document["name"],
                media_type=MediaType(document["media_type"]),
            )


class LuxonisSource:
    """Abstracts the structure of a dataset and which components/media
    are included."""

    class LuxonisSourceDocument(TypedDict):
        name: str
        main_component: str
        components: List[LuxonisComponent.LuxonisComponentDocument]

    def __init__(
        self,
        name: str = "default",
        components: Optional[List[LuxonisComponent]] = None,
        main_component: Optional[str] = None,
    ) -> None:
        """Abstracts the structure of a dataset by grouping together
        components.

        For example, with an U{OAK-D<https://docs.luxonis.com/projects/hardware
        /en/latest/pages/BW1098OAK/>}, you can have a source with 4 image
        components: C{rgb} (color), C{left} (mono), C{right} (mono), and C{depth}.

        @type name: str
        @param name: A recognizable name for the source. Defaults to "default".

        @type components: Optional[List[LuxonisComponent]]
        @param components: If not using the default configuration, a list of
        L{LuxonisComponent} to group together in the source.

        @type main_component: Optional[str]
        @param main_component: The name of the component that should be
            primarily visualized.
        """

        self.name = name
        if components is None:
            components = [
                LuxonisComponent(name)
            ]  # basic source includes a single color image

        self.components = {
            component.name: component for component in components
        }
        self.main_component = main_component or next(iter(self.components))

    def to_document(self) -> LuxonisSourceDocument:
        return {
            "name": self.name,
            "main_component": self.main_component,
            "components": [
                component.to_document()
                for component in self.components.values()
            ],
        }

    @classmethod
    def from_document(cls, document: LuxonisSourceDocument) -> "LuxonisSource":
        return cls(
            name=document.get("name", "default"),
            main_component=document.get("main_component"),
            components=[
                LuxonisComponent.from_document(component_doc)
                for component_doc in document["components"]
            ]
            if "components" in document
            else None,
        )
