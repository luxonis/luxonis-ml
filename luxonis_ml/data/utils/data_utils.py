from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np
import numpy.typing as npt


def check_arrays(values: Iterable[Any]) -> None:
    """Checks whether paths to numpy arrays are valid. This checks that th file exists
    and is readable by numpy.

    @type values: List[Any]
    @param values: A list of paths to numpy arrays.
    @rtype: NoneType
    @return: None
    """

    def _check_valid_array(path: str) -> bool:
        try:
            np.load(path)
            return True
        except Exception:
            return False

    for value in values:
        if not isinstance(value, str):
            raise Exception(
                f"Array value {value} must be a path to a numpy array (.npy)"
            )
        if not _check_valid_array(value):
            raise Exception(f"Array at path {value} is not a valid numpy array (.npy)")


def rgb_to_bool_masks(
    segmentation_mask: npt.NDArray[np.uint8],
    class_colors: Dict[str, Tuple[int, int, int]],
    add_background_class: bool = False,
) -> Iterator[Tuple[str, npt.NDArray[np.bool_]]]:
    """Helper function to convert an RGB segmentation mask to boolean masks for each
    class.

    Example::
        >>> segmentation_mask = np.array([[[0, 0, 0], [255, 0, 0], [0, 255, 0]],
        ...                                [[0, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        >>> class_colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
        >>> for class_name, mask in rgb_to_bool_masks(segmentation_mask, class_colors):
        ...     print(class_name, mask)
        red [[False  True False],
             [False False False]]
        green [[False False  True],
               [False True False]]
        blue [[False False False],
              [False False  True]]

    @type segmentation_mask: npt.NDArray[np.uint8]
    @param segmentation_mask: An RGB segmentation mask where each pixel is colored according to the class it belongs to.
    @type class_colors: Dict[str, Tuple[int, int, int]]
    @param class_colors: A dictionary mapping class names to RGB colors.
    @rtype: Iterator[Tuple[str, npt.NDArray[np.bool_]]]
    @return: An iterator of tuples where the first element is the class name and
        the second element is a boolean mask for that class.
    """
    color_to_id = {tuple(color): i for i, color in enumerate(class_colors.values())}

    lookup_table = np.zeros((256, 256, 256), dtype=np.uint8)
    for color, id in color_to_id.items():
        lookup_table[color[0], color[1], color[2]] = (
            id + 1
        )  # +1 to reserve 0 for background

    segmentation_ids = lookup_table[
        segmentation_mask[:, :, 0],
        segmentation_mask[:, :, 1],
        segmentation_mask[:, :, 2],
    ]

    for class_name, color in class_colors.items():
        class_id = color_to_id[tuple(color)] + 1
        yield class_name, segmentation_ids == class_id

    if add_background_class:
        background_mask = segmentation_ids == 0
        yield "background", background_mask
