# Augmentations

## `AlbumentationsEngine`

The default engine used with `LuxonisLoader`. It is powered by the [Albumentations](https://albumentations.ai/) library and should be satisfactory for most use cases. In addition to the built-in Albumentations transformations, it also supports custom transformations registered in the `TRANSFORMATIONS` registry.

### Creating and Registering a Custom Augmentation

The process of creating custom augmentations follows the same principles as described in the [Albumentations custom transform guide](https://albumentations.ai/docs/4-advanced-guides/creating-custom-transforms/#creating-custom-albumentations-transforms). You can subclass from their base classes such as `DualTransform`, `ImageOnlyTransform`, or others depending on the target types you want to support.

The example below shows how to define, register, and use a custom transform:

```python
import numpy as np
from typing import Any, Sequence
from albumentations import DualTransform

from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.augmentations.custom import TRANSFORMATIONS

class CustomTransform(DualTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def apply(self, image: np.ndarray, **_: Any) -> np.ndarray:
        return image

    def apply_to_mask(self, mask: np.ndarray, **_: Any) -> np.ndarray:
        return mask

    def apply_to_bboxes(self, bboxes: Sequence[Any], **_: Any) -> Sequence[Any]:
        return bboxes

    def apply_to_keypoints(self, keypoints: Sequence[Any], **_: Any) -> Sequence[Any]:
        return keypoints

# Register the transform
TRANSFORMATIONS.register(module=CustomTransform)

# Use it in the config
augmentation_config = [{
    "name": "CustomTransform",
    "params": {"p": 1},
}]

loader = LuxonisLoader(
    LuxonisDataset("coco_test"),
    augmentation_config=augmentation_config,
    view="train",
    height=640,
    width=640,
)

for data in loader:
  pass
```

### Examples of Custom Augmentations

- [`letterbox_resize.py`](./custom/letterbox_resize.py)
- [`symetric_keypoints_flip.py`](./custom/symetric_keypoints_flip.py)

### Batch-Level Augmentations

We also support **batch-level transformations**, built on top of the `BatchTransform` base class. These follow the same creation and registration pattern as standard custom transforms but operate on batches of data. This allows you to construct augmentations that combine multiple images and labels.

Examples:

- [`mosaic.py`](./custom/mosaic.py)
- [`mixup.py`](./custom/mixup.py)

### Configuration Format

The configuration format for the `AlbumentationsEngine` consists of a list of records, where each record contain 2 fields; `name` and `params`:

- `name`: The name of the transformation class to be applied (e.g., `HorizontalFlip`, `RandomCrop`, etc.). The name must be either a valid name of an Albumentations transformation (accessible under the `albumentations` namespace), or a name of a custom transformation registered in the `TRANSFORMATIONS` registry.
- `params`: A dictionary of parameters to be passed to the transformation.
- `use_for_resizing`: An optional boolean flag that indicates whether the transformation should be used for resizing. If no resizing augmentation is provided, the engine will use either `A.Resize` or `LetterboxResize` depending on the `keep_aspect_ratio` parameter (provided through the `LuxonisLoader`).

**Example:**

```yaml
- name: Defocus
  params:
    p: 1
- name: Sharpen
  params:
    p: 1
- name: Affine
  params:
    p: 1
- name: RandomCrop
  params:
    p: 1
    width: 512
    height: 512
- name: Mosaic4
  params:
    p: 1.
    out_width: 256
    out_height: 256
```

### Order of Transformations

The order of transformations provided in the configuration is not
guaranteed to be preserved. The transformations are divided into
the following groups and are applied in the same order:

1. Batch transformations - Subclasses of our custom `BatchTransform`

1. Spatial transformations - Subclasses of `A.DualTransform`

1. Custom transformations - Subclasses of `A.BasicTransform`,
   but not subclasses of more specific base classes above

1. Pixel transformations: Subclasses of `A.ImageOnlyTransform`.
   These transformations act only on the image

The resize transformation is applied either before or after the pixel transformations, depending on desired output size. If the output size is smaller than the initial image size, the resize transformation is applied before the pixel transformations to save compute. In the other case it is applied last.

## Extensibility

`LuxonisLoader` can work with any subclass of `BaseEngine` that is registered in the `AUGMENTATION_ENGINES` registry.

To implement a custom augmentation engine, you need to create a new class that inherits from `BaseEngine` and implements the required methods. Any subclass of `BaseEngine` is automatically registered in the aforementioned registry.

**Required Methods:**

- `__init__`: The constructor method that initializes the engine with the provided configuration. It needs to create a new instance of the augmentation engine from the following arguments:
  - `height`: The output height of the images
  - `width`: The output width of the images
  - `n_classes`: The number of classes in the dataset
  - `config`: The configuration for the augmentation engine as an iterable of dictionaries. Interpretation of the configuration is left to the engine (it doesn't need to follow the format used in `AlbumentationsEngine`).
  - `keep_aspect_ratio`: A boolean flag that indicates whether to keep the aspect ratio of the images during resizing. The engine should respect this flag when applying the resizing transformation.
  - `is_validation_pipeline`: A boolean flag that indicates whether the engine is being used for validation. Typically, the applied transformations differ between training and validation (for example validation pipeline would only use resizing and normalization).
  - `targets`: A dictionary mapping names of individual labels (given in `apply`) to their respective label types. Possible values of the label types are `"boundingbox"`, `"segmentation"`, `"instance_segmentation"`, `"keypoints"`, `"array"`, `"classification"`, and `"metadata"`. Interpretation of the targets is left to the engine.
- `apply`: This method applies the augmentation engine to the provided batch of images and labels. It should return a tuple containing the augmented images and labels. The method should also handle the resizing of images and targets according to the specified output size and aspect ratio.
- `batch_size`: An abstract property that returns the expected batch size of the inputs. This is required for the `LuxonisLoader` to properly handle the input data for batched augmentations. For example if the pipeline contains `MixUp` augmentation (which requires 2 images) and `Mosaic4` (requiring 4 images), the batch size should be $$2 * 4 = 8$$.
