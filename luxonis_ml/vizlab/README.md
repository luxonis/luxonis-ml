# vizlab

The visualization engine of `luxonis-ml`. Composition-first, genuinely pretty
visualization of computer-vision labels and predictions — bounding boxes,
instance/semantic masks, keypoints with skeletons, classification tags, and
nested sub-labels — rendered with Skia for anti-aliasing, true alpha, rounded
corners, soft shadows, and good typography.

vizlab is **LDF-native**: it renders Luxonis Data Format objects directly. Pass a
`luxonis_ml.ldf.Detection` (or a whole `DatasetRecord`) straight to
`Image.add(...)`, or use `visualize_record(record, image, config=...)`. The
`BBox`/`Keypoints`/`Mask`/… render classes remain available as lower-level
drawing primitives (with a `from_ldf` constructor each).

Install with the `viz` extra: `pip install luxonis-ml[viz]`.

## Design at a glance

- **`Image` is the composition root.** Annotations are collected with `.add(...)`;
  nothing is drawn until `.render()` (lazy rendering), so labels are laid out with
  full knowledge of the whole scene.
- **Smart defaults.** Colors, label placement, and style are chosen for you. Box
  formats (`xyxy`/`xywh`/`cxcywh`, normalized or pixel), color spaces, and mask
  formats (binary array, polygons, label maps) are resolved automatically. Override
  anything when you want to.
- **Distinct colors.** A class gets a stable, well-spread color via golden-ratio hue
  spacing, so no two classes land on a near-identical shade.
- **Hierarchical styling.** A sub-label's style is *derived* from its parent — a
  lighter shade with a thinner, dashed outline — so nesting (a `driver` box inside a
  `car` box) reads at a glance.
- **Collision-aware labels.** Label chips are placed to avoid overlapping each other,
  which matters most when instances overlap (e.g. a mixup).
- **Composition.** Blend images (mixup), stack them side by side, or lay them out in a
  grid — all pure, returning a new `Image`.
- **Theming.** `DARK_THEME` (default) and `LIGHT_THEME`, or set your own.

## Label types

`BBox` (axis-aligned or oriented — pass `angle=`, a `cxcywha` box, or four corner
points), `Keypoints` (+ `Skeleton`, `COCO_17`), `Mask` (instance: binary array,
polygons, or COCO RLE), `SemanticMask` (dense label map), `Classification`
(image-level corner tags), plus `Caption` and `Legend` overlays. Boxes accept
`xywh` (default), `xyxy`, and `cxcywh` formats in pixel or normalized units. Any
annotation can carry a `label`, a `score`, and an arbitrary `payload` (the OCR case:
a box plus its transcribed text).

## Metadata panel

Append a "second window" of arbitrary JSON-like metadata beside an image — it never
occludes the pixels or labels:

```python
img.with_panel(
    {"source": "val2017/042.jpg", "augmentations": ["flip", "blur"],
     "tags": {"difficulty": "hard", "verified": True}},
    title="metadata",
).save("out.png")
```

## Intended usage

Render LDF objects directly — this is the first-class path:

```python
from luxonis_ml.ldf import Detection
from luxonis_ml.vizlab import Image

det = Detection(class_name="car", boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
Image(image).add(det).save("out.png")
```

Visualize a whole dataset sample via the loader → record path (what the
`luxonis_ml data inspect` CLI uses):

```python
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.data.loaders.label_converter import loader_output_to_records
from luxonis_ml.vizlab import VizConfig, Palette, visualize_record

dataset = LuxonisDataset("parking_lot")
config = VizConfig(palette=Palette(sorted(dataset.get_classes()[""])))
for sample in LuxonisLoader(dataset):
    records = loader_output_to_records(
        sample.labels, classes=dataset.get_classes(),
        image_shape=next(iter(sample.images.values())).shape[:2],
    )
    for task_name, record in records.items():
        visualize_record(record, sample.images["image"], config=config).save(f"{task_name}.png")
```

The lower-level render classes are still available (with all their features —
oriented boxes, multiple formats, scores, payloads) and compose the same way:

```python
from luxonis_ml.vizlab import Image, BBox, Classification, grid

car = BBox((40, 60, 300, 380), label="car", score=0.98)
car.add(BBox((120, 150, 260, 340), label="driver", score=0.9))   # derived styling
page = Image(scan).add(BBox((10, 10, 200, 60), label="word", payload="INVOICE"))
grid([Image(street).add(car), page], ncols=2, titles=["street", "document"]).save("out.png")
```

## Development

vizlab is part of `luxonis-ml`; use the repo-wide workflow (see the root
`CONTRIBUTING.md` / `AGENTS.md`):

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_vizlab -q
pre-commit run --all-files
```
