from typing import Any

from pydantic import Field, field_validator

from luxonis_ml.data.utils import ImageType, MediaType
from luxonis_ml.typing import BaseModelExtraForbid


class LuxonisComponent(BaseModelExtraForbid):
    """Media component within a source.

    Most commonly, this represents one image sensor.

    Attributes:
        name: Human-readable component name.
        media_type: Media kind represented by the component.
        image_type: Image kind. Only used for image media.

    """

    name: str
    media_type: MediaType = MediaType.IMAGE
    image_type: ImageType = ImageType.COLOR


class LuxonisSource(BaseModelExtraForbid):
    """Source definition for a dataset.

    A source describes which components or media streams are included. For
    example, an `OAK-D`_ source can contain ``rgb``, ``left``,
    ``right``, ``depth`` components.

    Attributes:
        name: Human-readable source name.
        components: Components grouped in the source.
        main_component: Component name used as the primary visualization
            target.

    .. _OAK-D: https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK/

    """

    name: str = "default"
    components: dict[str, LuxonisComponent] = Field(
        default_factory=lambda: {"default": LuxonisComponent(name="default")}
    )
    main_component: str = "default"

    def merge_with(self, other: "LuxonisSource") -> "LuxonisSource":
        """Merge two sources together.

        ``name`` and ``main_component`` are taken from the first source.
        ``components`` are merged together, with the second source's components
        taking precedence in case of name conflicts.

        Example:
            >>> source1 = LuxonisSource(
            ...     name="source1",
            ...     components={
            ...         "rgb": LuxonisComponent(name="rgb"),
            ...     },
            ...     main_component="rgb",
            ... )
            >>> source2 = LuxonisSource(
            ...     name="source2",
            ...     components={
            ...         "depth": LuxonisComponent(name="depth"),
            ...     },
            ...     main_component="depth",
            ... )
            >>> merged_source = source1.merge_with(source2)
            >>> merged_source.name
            'source1'
            >>> merged_source.main_component
            'rgb'
            >>> sorted(merged_source.components.keys())
            ['depth', 'rgb']

        Args:
            other: Source to merge with.

        Returns:
            New source containing components from both sources.

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
    def _validate_components(cls, components: Any) -> Any:
        if isinstance(components, list):
            migrated_components = {}
            for component in components:
                component = LuxonisComponent(**component)
                migrated_components[component.name] = component
            return migrated_components
        return components
