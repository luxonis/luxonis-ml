"""Build the LDF datasets that ``luxonis_ml test viz`` opens.

`build_feature_datasets` paints street frames with `street` and stores them
as three local datasets, so that one command shows every annotation kind that
LDF stores and vizlab draws:

========================  =====================================================
Dataset                   Contents
========================  =====================================================
``<prefix>``              Focused cases, then four sequences of moving
                          traffic. Boxes, instance masks from arrays and from
                          polygons, semantic masks, image-level classes, named
                          keypoints in two skeletons, sub-detections,
                          detection metadata of each type, sample metadata,
                          true negatives, and eleven array tasks.
``<prefix>_predictions``  Scored detections for the same images, with misses,
                          false alarms, class errors, swapped joints, and
                          scores below the threshold. One image has no
                          prediction, and one prediction has no ground truth.
``<prefix>_stereo``       Left and right images of one sequence, with a
                          disparity map that marks unmatched pixels with
                          ``0``.
========================  =====================================================

The media and the arrays go under ``LUXONISML_BASE_PATH/testing/<prefix>``,
because a local dataset keeps the paths of its files. The same folder holds
``augmentations.json`` for ``data inspect --aug-config``.

The array tasks of the main dataset:

=================  ======================  ===================================
Task               Stored as               Read as
=================  ======================  ===================================
``depth``          ``(H, W)``, NaN in sky  reserved name: scalar; full size
``normals``        ``(H, W, 3)``           reserved name: normals
``flow``           ``(2, H, W)``           reserved name: flow
``error``          ``(H, W)``, signed      reserved name: signed
``class_scores``   one ``(H, W)`` a class  reserved name: scores, named
``confidence``     the same files          reserved name: confidence
``logits``         ``(9, H, W)``           reserved name: scores, unnamed
``rgb``            ``(H, W, 3)`` bytes     reserved name: image
``anomaly``        ``(H, W)``              reserved name: scalar
``occupancy``      ``(96, 96)``            shape: scalar; not overlaid
``embedding``      ``(128,)``              never drawn
=================  ======================  ===================================

The last four are on some samples only. Except for the depth, the arrays
have a quarter of the image resolution.

Example:
    >>> from luxonis_ml.testing import FeatureDatasets
    >>> FeatureDatasets.named("demo").stereo
    'demo_stereo'

"""

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.ldf import Category
from luxonis_ml.typing import Params
from luxonis_ml.utils.environ import environ

from .street import (
    FACADE,
    FACE_EDGES,
    POSE_EDGES,
    POSE_FLIP_PAIRS,
    POSE_SIGMAS,
    SEMANTIC_CLASSES,
    VEHICLE_LENGTH,
    VEHICLE_SIZE,
    Box,
    Camera,
    Frame,
    Instance,
    LabeledPoint,
    Street,
    make_sign,
    random_person,
    random_street,
    random_vehicle,
)

_Raw: TypeAlias = dict[str, Any]
"""A record or a detection as `LuxonisDataset.add` reads it.

It is not `Params`, because a mask in it is an array.
"""

#: The image tasks of the main dataset, in the order the tiles show them.
IMAGE_TASKS = ("vehicles", "people", "signs", "semantic", "scene")

#: The array tasks of the main dataset.
ARRAY_TASKS = (
    "depth",
    "normals",
    "flow",
    "error",
    "class_scores",
    "confidence",
    "logits",
    "rgb",
    "anomaly",
    "occupancy",
    "embedding",
)

_SPARSE_ARRAYS = ("logits", "rgb", "anomaly", "occupancy")
_SHRINK = 4
_SIGN_SHAPES = {
    "stop": "octagon",
    "yield": "triangle",
    "speed_limit": "circle",
    "info": "rectangle",
}
_SWAPS = {
    "car": "truck",
    "truck": "bus",
    "bus": "truck",
    "person": "child",
    "child": "person",
    "stop": "yield",
    "yield": "stop",
    "speed_limit": "info",
    "info": "speed_limit",
}


@dataclass(frozen=True)
class FeatureDatasets:
    """The names of the three feature datasets and their media folder.

    Attributes:
        main: The dataset with every annotation kind.
        predictions: The scored detections for the images of ``main``.
        stereo: The dataset with left and right images.
        root: The folder of the media, the arrays and the augmentations.

    """

    main: str
    predictions: str
    stereo: str
    root: Path

    @classmethod
    def named(
        cls, prefix: str = "vizlab_features", root: Path | None = None
    ) -> "FeatureDatasets":
        """Name the datasets after a prefix.

        Args:
            prefix: The name of the main dataset, and the start of the
                other two names.
            root: The media folder. Defaults to
                ``LUXONISML_BASE_PATH/testing/<prefix>``.

        Returns:
            The names and the folder.

        """
        return cls(
            main=prefix,
            predictions=f"{prefix}_predictions",
            stereo=f"{prefix}_stereo",
            root=root or environ.LUXONISML_BASE_PATH / "testing" / prefix,
        )

    @property
    def augmentations(self) -> Path:
        """The augmentation config for ``data inspect --aug-config``."""
        return self.root / "augmentations.json"

    def exist(self) -> bool:
        """Report whether all three datasets and their media exist.

        Returns:
            ``True`` when nothing needs to be built.

        """
        return self.augmentations.exists() and all(
            LuxonisDataset.exists(name)
            for name in (self.main, self.predictions, self.stereo)
        )


@dataclass(frozen=True)
class _Sample:
    """One frame of the main dataset, and where it goes."""

    stem: str
    frame: Frame
    street: Street
    split: str
    metadata: Params
    arrays: tuple[str, ...]
    unlabeled: bool = False


def augmentation_config(camera: Camera) -> list[Params]:
    """Return augmentations that move every kind of label.

    The two batch transforms merge samples, which the side panel of
    ``data inspect`` lists. The flip mirrors the joints but keeps their
    names, so a flipped left wrist sits on the right of the body.
    ``HorizontalSymmetricKeypointsFlip`` would swap the names, but it
    cannot handle a skeleton with an unpaired joint or two keypoint tasks.

    Args:
        camera: The camera of the images.

    Returns:
        The config, as the records that `LuxonisLoader` reads.

    """
    return [
        {
            "name": "Mosaic4",
            "params": {
                "out_width": camera.width,
                "out_height": camera.height,
                "p": 0.3,
            },
        },
        {"name": "MixUp", "params": {"p": 0.2}},
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "Affine",
            "params": {"rotate": [-8, 8], "scale": [0.85, 1.15], "p": 0.6},
        },
        {"name": "ColorJitter", "params": {"p": 0.5}},
    ]


def build_feature_datasets(
    prefix: str = "vizlab_features",
    *,
    root: Path | None = None,
    camera: Camera | None = None,
    sequences: int = 4,
    frames: int = 12,
    seed: int = 0,
) -> FeatureDatasets:
    """Paint the frames and store the three feature datasets.

    Existing datasets with the same names, and the media folder, are
    replaced.

    Args:
        prefix: The name of the main dataset, and the start of the other
            two names.
        root: The media folder. See `FeatureDatasets.named`.
        camera: The camera of every frame. A smaller one builds faster.
        sequences: How many sequences of moving traffic follow the focused
            cases, at most four.
        frames: How many frames each sequence has. The stereo dataset has
            the same number.
        seed: The seed of everything random.

    Returns:
        The names of the datasets.

    """
    names = FeatureDatasets.named(prefix, root)
    camera = camera or Camera()
    rng = np.random.default_rng(seed)
    if names.root.exists():
        shutil.rmtree(names.root)
    for folder in ("images", "arrays", "stereo"):
        (names.root / folder).mkdir(parents=True)
    names.augmentations.write_text(
        json.dumps(augmentation_config(camera), indent=2)
    )

    samples = [
        *_case_samples(rng, camera),
        *_sequence_samples(rng, camera, sequences, frames),
    ]
    main = [_main_record(names.root, sample) for sample in samples]
    split_of = {
        str(_image_path(names.root, sample.stem)): sample.split
        for sample in samples
    }
    _store(names.main, main, _splits(main, split_of))

    predictions = list(_prediction_records(rng, names.root, samples, camera))
    _store(names.predictions, predictions, _splits(predictions, split_of))

    stereo = list(_stereo_records(rng, names.root, camera, frames))
    _store(names.stereo, stereo, {"train": 1.0})
    return names


def _store(
    name: str,
    records: list[_Raw],
    splits: dict[str, list[str]] | dict[str, float],
) -> None:
    """Write records to a fresh local dataset and split it."""
    dataset = LuxonisDataset(name, delete_local=True)
    dataset.add(iter(records))
    dataset.make_splits(splits)


def _splits(
    records: list[_Raw], split_of: dict[str, str]
) -> dict[str, list[str]]:
    """Group the images of records by split; an unknown one is train."""
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for record in records:
        media = str(record["media"])
        splits[split_of.get(media, "train")].append(media)
    return splits


def _image_path(root: Path, stem: str) -> Path:
    """Return where the image of a sample goes."""
    return root / "images" / f"{stem}.jpg"


def _write_image(path: Path, image: np.ndarray) -> str:
    """Write an RGB image and return its path."""
    cv2.imwrite(str(path), image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(path)


def _cases(
    rng: np.random.Generator,
) -> list[tuple[str, str, Street, bool]]:
    """Return the focused cases: a name, a description, a street, a flag.

    The flag marks the case that gets no annotation at all.
    """
    everything = random_street(
        rng, vehicles=0, people=0, signs=0, cone=True, far=60
    )
    everything.things += [
        replace(random_vehicle(rng, 1.75, 16, 1), label="car", color="red"),
        replace(random_vehicle(rng, -1.75, 26, 2), label="truck"),
        replace(random_vehicle(rng, 1.8, 42, 3), label="bus", caption="42"),
        replace(
            random_person(rng, 4.6, 9, 4),
            label="person",
            height=1.78,
            pose="waving",
        ),
        replace(random_person(rng, -4.4, 12, 5), pose="pointing"),
        replace(random_person(rng, 4.2, 15, 6), label="child", height=1.2),
        replace(random_person(rng, 0.6, 14, 7), pose="walking", vx=-1.2),
        make_sign("stop", -3.95, 11, 8),
        make_sign("yield", 3.95, 19, 9),
        make_sign("speed_limit", -3.95, 24, 10),
        make_sign("info", 3.95, 30, 11),
    ]
    # LDF keys a file by its content, so the two negatives need two streets.
    empty = random_street(rng, vehicles=0, people=0, signs=0)
    declared = random_street(rng, vehicles=0, people=0, signs=0)
    crowd = random_street(rng, vehicles=16, people=48, signs=6, far=140)
    far = random_street(rng, vehicles=6, people=6, signs=3, near=70, far=160)
    occlusion = random_street(rng, vehicles=0, people=0, signs=0)
    occlusion.things += [
        random_vehicle(rng, 0.4, 9, 1),
        replace(random_person(rng, 0.9, 10.5, 2), pose="walking", vx=1.2),
        make_sign("stop", 4.0, 8, 3),
        replace(random_person(rng, 4.1, 9.0, 4), pose="standing"),
    ]
    border = random_street(rng, vehicles=0, people=0, signs=0)
    border.things += [
        replace(random_vehicle(rng, -2.4, 4.2, 1), label="car"),
        replace(random_person(rng, 3.0, 3.6, 2), pose="waving"),
        make_sign("info", 3.95, 4.5, 3),
    ]
    lone = random_street(rng, vehicles=0, people=0, signs=0)
    lone.things.append(
        replace(random_vehicle(rng, 1.75, 11, 1), label="car", color="blue")
    )
    away = random_street(rng, vehicles=0, people=0, signs=0)
    away.things += [
        replace(random_person(rng, 4.4, 8, 1), facing=False, pose="walking"),
        replace(random_person(rng, -4.2, 9, 2), facing=False),
        random_person(rng, 4.8, 13, 3),
    ]
    markup = random_street(rng, vehicles=2, people=2, signs=1, far=30)
    for thing in markup.things:
        if thing.kind in ("vehicle", "person", "sign"):
            thing.note = "<b>not bold</b> [red]not red[/red] & <i>Ünïcödé</i>"
    close = random_street(rng, vehicles=0, people=0, signs=0)
    close.things.append(
        replace(random_person(rng, 0.0, 4.0, 1), pose="waving", height=1.8)
    )
    night = random_street(
        rng, vehicles=5, people=5, signs=2, light="night", weather="rain"
    )
    dusk = random_street(
        rng, vehicles=5, people=5, signs=2, light="dusk", weather="fog"
    )
    return [
        (
            "everything",
            "every annotation kind and every array",
            everything,
            False,
        ),
        ("true_negative", "no annotation at all", empty, True),
        (
            "declared_negative",
            "detection tasks declared empty",
            declared,
            False,
        ),
        ("crowd", "dense scene: the d key declutters it", crowd, False),
        ("far_away", "tiny objects near the horizon", far, False),
        ("occlusion", "cut masks and occluded joints", occlusion, False),
        ("image_border", "objects cut by the image border", border, False),
        ("lone_object", "one object: its metadata card", lone, False),
        ("facing_away", "no face sub-detections: no nesting", away, False),
        ("markup_text", "markup and Unicode in metadata", markup, False),
        ("night_rain", "night and rain", night, False),
        ("dusk_fog", "dusk and fog", dusk, False),
        ("close_pose", "one large pose for keypoint labels", close, False),
    ]


def _case_samples(
    rng: np.random.Generator, camera: Camera
) -> Iterator[_Sample]:
    """Render the focused cases into the train split."""
    for index, (name, description, street, unlabeled) in enumerate(
        _cases(rng)
    ):
        arrays = (
            ARRAY_TASKS
            if name == "everything"
            else ()
            if unlabeled
            else tuple(t for t in ARRAY_TASKS if t not in _SPARSE_ARRAYS)
        )
        metadata: Params = {
            "case": description,
            "camera": _camera_metadata(camera),
            "conditions": {"light": street.light, "weather": street.weather},
        }
        if name == "markup_text":
            metadata["notes"] = {
                "markup": "<b>bold?</b> [red]red?[/red] {braces} & ampersand",
                "unicode": "Zürich → Genève · 東京 · ✓",
                # Right to left: Hebrew reads backwards and Arabic letters do
                # not join unless the text is shaped.
                "rtl": "שלום עולם · مرحبا بالعالم · km 12",
                "long": "A long value that has to wrap in the side panel " * 4,
                "tags": ["<tag>", "[markup]", "&amp;"],
            }
        yield _Sample(
            stem=f"case_{index:02d}_{name}",
            frame=street.render(camera),
            street=street,
            split="train",
            metadata=metadata,
            arrays=arrays,
            unlabeled=unlabeled,
        )


_SEQUENCES = (
    ("day", "clear", "train"),
    ("night", "clear", "train"),
    ("dusk", "rain", "val"),
    ("day", "fog", "test"),
)


def _sequence_samples(
    rng: np.random.Generator, camera: Camera, sequences: int, frames: int
) -> Iterator[_Sample]:
    """Render sequences of moving traffic, one split for each."""
    for number, (light, weather, split) in enumerate(_SEQUENCES[:sequences]):
        street = random_street(
            rng,
            vehicles=6,
            people=7,
            signs=3,
            light=light,
            weather=weather,
            cone=number == 0,
        )
        for step in range(frames):
            moment = street.at(step)
            yield _Sample(
                stem=f"seq{number}_{step:03d}",
                frame=moment.render(camera),
                street=moment,
                split=split,
                metadata={
                    "sequence": f"seq{number}",
                    "frame": step,
                    "camera": _camera_metadata(camera),
                    "conditions": {"light": light, "weather": weather},
                    "speed_mps": round(street.speed, 2),
                    "tags": ["synthetic", light, weather],
                },
                arrays=tuple(
                    task
                    for task in ARRAY_TASKS
                    if step % 3 == 0 or task not in _SPARSE_ARRAYS
                ),
            )


def _camera_metadata(camera: Camera) -> Params:
    """Describe a camera for the sample metadata."""
    return {
        "fov_deg": camera.fov,
        "mount_m": camera.mount,
        "size": [camera.width, camera.height],
    }


def _bbox(box: Box) -> Params:
    """Turn a normalized box into its LDF form."""
    x, y, w, h = box
    return {"x": x, "y": y, "w": w, "h": h}


def _vehicle(instance: Instance) -> _Raw:
    """Describe a vehicle, with its number plate as a sub-detection."""
    thing = instance.thing
    metadata: Params = {
        "color": Category(thing.color),
        "speed_kmh": round(thing.vz * 3.6, 1),
        "lane": Category("left" if thing.x < 0 else "right"),
    }
    if thing.note:
        metadata["note"] = thing.note
    detection: _Raw = {
        "class": thing.label,
        "instance_id": thing.track,
        "boundingbox": _bbox(instance.box),
        "instance_segmentation": {"mask": instance.mask},
        "metadata": metadata,
    }
    if instance.plate_box is not None:
        detection["sub_detections"] = {
            "plate": {
                "class": "plate",
                "boundingbox": _bbox(instance.plate_box),
                "metadata": {"text": thing.text},
            }
        }
    return detection


def _person(instance: Instance) -> _Raw:
    """Describe a person, with the face as a sub-detection."""
    thing = instance.thing
    metadata: Params = {
        "action": Category(thing.pose),
        "height_m": round(thing.height, 2),
        "facing": Category("camera" if thing.facing else "away"),
    }
    if thing.note:
        metadata["note"] = thing.note
    detection: _Raw = {
        "class": thing.label,
        "instance_id": thing.track,
        "boundingbox": _bbox(instance.box),
        "instance_segmentation": {"mask": instance.mask},
        "keypoints": {
            "keypoints": instance.keypoints,
            "edges": POSE_EDGES,
            "sigmas": POSE_SIGMAS,
        },
        "metadata": metadata,
    }
    if instance.face_box is not None:
        detection["sub_detections"] = {
            "face": {
                "boundingbox": _bbox(instance.face_box),
                "keypoints": {
                    "keypoints": instance.face_keypoints,
                    "edges": FACE_EDGES,
                },
            }
        }
    return detection


def _sign(instance: Instance, camera: Camera) -> _Raw:
    """Describe a sign, with its panel as a polygon."""
    thing = instance.thing
    metadata: Params = {
        "text": thing.caption,
        "shape": Category(_SIGN_SHAPES[thing.label]),
    }
    if thing.note:
        metadata["note"] = thing.note
    return {
        "class": thing.label,
        "instance_id": thing.track,
        "boundingbox": _bbox(instance.box),
        "instance_segmentation": {
            "points": instance.outline,
            "width": camera.width,
            "height": camera.height,
        },
        "metadata": metadata,
    }


def _detections(frame: Frame) -> dict[str, list[_Raw]]:
    """Describe the visible objects of a frame, grouped by task.

    A task without objects is an empty list, which declares it negative.
    """
    return {
        "vehicles": [
            _vehicle(i) for i in frame.instances if i.thing.kind == "vehicle"
        ],
        "people": [
            _person(i) for i in frame.instances if i.thing.kind == "person"
        ],
        "signs": [
            _sign(i, frame.camera)
            for i in frame.instances
            if i.thing.kind == "sign"
        ],
    }


def _scene(street: Street, frame: Frame) -> list[_Raw]:
    """Describe the whole image: two classes and box-less metadata."""
    lux = {"day": 12000.0, "dusk": 400.0, "night": 3.0}[street.light]
    visibility = {"clear": 10000.0, "rain": 2000.0, "fog": 60.0}
    return [
        {"class": street.light},
        {"class": street.weather},
        {
            "metadata": {
                "illuminance_lux": lux,
                "visibility_m": visibility[street.weather],
                "objects": len(frame.instances),
            }
        },
    ]


def _main_record(root: Path, sample: _Sample) -> _Raw:
    """Write the media of one sample and describe its annotations."""
    frame = sample.frame
    image = _write_image(_image_path(root, sample.stem), frame.image)
    if sample.unlabeled:
        return {
            "media": image,
            "annotation": {},
            "sample_metadata": sample.metadata,
        }
    annotation: dict[str, list[_Raw]] = _detections(frame)
    semantic = [
        {"class": name, "segmentation": {"mask": frame.semantic == i + 1}}
        for i, name in enumerate(SEMANTIC_CLASSES)
        if (frame.semantic == i + 1).any()
    ]
    annotation |= {
        "semantic": semantic,
        "scene": _scene(sample.street, frame),
    }
    arrays = _arrays(frame, sample.street)
    for task in sample.arrays:
        annotation[task] = _array_detections(
            root, sample.stem, task, arrays[task]
        )
    return {
        "media": image,
        "annotation": annotation,
        "sample_metadata": sample.metadata,
    }


def _array_detections(
    root: Path, stem: str, task: str, array: np.ndarray | list[np.ndarray]
) -> list[_Raw]:
    """Save the arrays of one task and describe them.

    A list holds one array for each class of the task, and each one is
    stored as a detection of its own class.
    """
    if isinstance(array, np.ndarray):
        path = root / "arrays" / f"{stem}_{task}.npy"
        np.save(path, array)
        return [{"class": task, "array": {"data": str(path)}}]
    detections = []
    for name, channel in zip(SEMANTIC_CLASSES, array, strict=True):
        path = root / "arrays" / f"{stem}_scores_{name}.npy"
        if not path.exists():
            np.save(path, channel)
        detections.append({"class": name, "array": {"data": str(path)}})
    return detections


def _shrink(field: np.ndarray) -> np.ndarray:
    """Take every fourth pixel of a map, which keeps NaN and inf intact."""
    start = _SHRINK // 2
    return field[start::_SHRINK, start::_SHRINK]


def _arrays(frame: Frame, street: Street) -> dict[str, np.ndarray | list]:
    """Derive every array label of a frame.

    Depth keeps the size of the image; the rest has a quarter of it.
    """
    sky = ~np.isfinite(frame.depth)
    depth = np.where(sky, np.nan, frame.depth).astype(np.float16)
    rng = np.random.default_rng(street.seed)

    sky = _shrink(sky)
    near = np.minimum(np.where(sky, 0.0, _shrink(frame.depth)), 40.0)
    noise = cv2.GaussianBlur(
        rng.normal(0, 1, near.shape).astype(np.float32), (0, 0), 3
    )
    # A stereo depth error grows with distance. It is capped at 40 m, so the
    # far facades do not outweigh the scene in front of the camera.
    error = np.where(sky, np.nan, 0.05 * near * noise / noise.std())

    labels = _shrink(frame.semantic).astype(int)
    one_hot = np.eye(len(SEMANTIC_CLASSES) + 1, dtype=np.float32)[labels]
    logits = cv2.GaussianBlur(one_hot[..., 1:] * 4, (0, 0), 1.0)
    logits += rng.normal(0, 0.7, logits.shape).astype(np.float32)
    scores = np.exp(logits - logits.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)

    cone = _shrink(frame.cone).astype(np.float32)
    anomaly = cv2.GaussianBlur(cone, (0, 0), 2.0) * 3
    anomaly += np.abs(rng.normal(0, 0.03, cone.shape))
    return {
        "depth": depth,
        "normals": _shrink(frame.normals).astype(np.float16),
        "flow": np.moveaxis(_shrink(frame.flow) / _SHRINK, -1, 0).astype(
            np.float16
        ),
        "error": error.astype(np.float16),
        "class_scores": list(np.moveaxis(scores, -1, 0).astype(np.float16)),
        "confidence": list(np.moveaxis(scores, -1, 0).astype(np.float16)),
        "logits": np.moveaxis(logits, -1, 0).astype(np.float16),
        "rgb": _shrink(frame.albedo),
        "anomaly": np.clip(anomaly, 0, 1).astype(np.float16),
        "occupancy": _occupancy(street),
        "embedding": rng.normal(0, 1, 128).astype(np.float32),
    }


def _occupancy(
    street: Street, cells: int = 96, reach: float = 24.0
) -> np.ndarray:
    """Draw a bird's-eye grid of what stands on the street, far rows first."""
    grid = np.zeros((cells, cells), np.float32)
    scale = cells / (2 * reach)

    def cell(x: float, z: float) -> tuple[int, int]:
        return int((x + reach) * scale), int(cells - 1 - z * scale)

    for building in street.buildings:
        x0 = building.side * FACADE
        z0 = building.z0 - street.travel
        z1 = building.z1 - street.travel
        x1 = x0 + building.side * 8
        cv2.rectangle(grid, cell(x0, z0), cell(x1, z1), 1.0, -1)
    for thing in street.things:
        if thing.kind == "vehicle":
            half = VEHICLE_SIZE[thing.label][0] / 2
            end = thing.z + VEHICLE_LENGTH[thing.label]
            corner, far = (
                cell(thing.x - half, thing.z),
                cell(thing.x + half, end),
            )
            cv2.rectangle(grid, corner, far, 1.0, -1)
        else:
            radius = {"tree": 1.2, "person": 0.4}.get(thing.kind, 0.3)
            cv2.circle(
                grid,
                cell(thing.x, thing.z),
                max(1, int(radius * scale)),
                0.8,
                -1,
            )
    return cv2.GaussianBlur(grid, (0, 0), 0.8)


def _prediction_records(
    rng: np.random.Generator,
    root: Path,
    samples: list[_Sample],
    camera: Camera,
) -> Iterator[_Raw]:
    """Make up scored predictions for the images of the main dataset.

    The last two train images with objects get no prediction, and one extra
    image has predictions but no ground truth, so that ``data compare``
    reports both in its default view.
    """
    skipped = {
        s.stem for s in samples if s.frame.instances and s.split == "train"
    }
    skipped = set(sorted(skipped)[-2:])
    for sample in samples:
        if sample.stem in skipped:
            continue
        yield {
            "media": str(_image_path(root, sample.stem)),
            "annotation": _predict(rng, sample.frame),
        }
    extra = random_street(rng, vehicles=4, people=3, signs=1)
    frame = extra.render(camera)
    path = _write_image(
        root / "images" / "unpaired_prediction.jpg", frame.image
    )
    yield {"media": path, "annotation": _predict(rng, frame)}


def _predict(rng: np.random.Generator, frame: Frame) -> dict[str, list[_Raw]]:
    """Turn the ground truth of a frame into imperfect scored detections.

    One object in eight is missed, and each frame gets up to two false
    alarms.
    """
    tasks: dict[str, list[_Raw]] = {
        "vehicles": [],
        "people": [],
        "signs": [],
    }
    task_of = {"vehicle": "vehicles", "person": "people", "sign": "signs"}
    for instance in frame.instances:
        if rng.random() >= 0.12:
            task = task_of[instance.thing.kind]
            tasks[task].append(_predict_one(rng, instance))
    for _ in range(int(rng.integers(0, 3))):
        box = (
            float(rng.uniform(0.05, 0.85)),
            float(rng.uniform(0.4, 0.75)),
            float(rng.uniform(0.03, 0.1)),
            float(rng.uniform(0.05, 0.2)),
        )
        label = str(rng.choice(["car", "truck"]))
        score = float(rng.uniform(0.25, 0.6))
        tasks["vehicles"].append(_scored(rng, label, box, score))
    return tasks


def _predict_one(rng: np.random.Generator, instance: Instance) -> _Raw:
    """Predict one object: a class error, a low score or a loose box.

    The prediction keeps the sub-detections and the keypoints of the
    object, moved a little. One pose in seven has left and right swapped.
    """
    label, score = instance.thing.label, float(rng.uniform(0.55, 0.99))
    if rng.random() < 0.08:
        label, score = _SWAPS[label], float(rng.uniform(0.3, 0.8))
    elif rng.random() < 0.06:
        score = float(rng.uniform(0.05, 0.2))
    loose = rng.random() < 0.1
    detection = _scored(rng, label, instance.box, score, loose=loose)
    if instance.keypoints:
        detection["keypoints"] = {
            "keypoints": _jitter_points(
                rng,
                instance.keypoints,
                0.04 * instance.box[3],
                swap=rng.random() < 0.15,
            )
        }
    if instance.plate_box is not None:
        plate = _scored(rng, "plate", instance.plate_box, score)
        detection["sub_detections"] = {"plate": plate}
    if instance.face_box is not None:
        face = _scored(rng, None, instance.face_box, score)
        face["keypoints"] = {
            "keypoints": _jitter_points(
                rng, instance.face_keypoints, 0.08 * instance.face_box[3]
            )
        }
        detection["sub_detections"] = {"face": face}
    return detection


def _scored(
    rng: np.random.Generator,
    label: str | None,
    box: Box,
    score: float,
    *,
    loose: bool = False,
) -> _Raw:
    """Make a scored detection around a box that moves a little, or a lot."""
    moved = _jitter(rng, box, 0.4 if loose else 0.06)
    detection: _Raw = {
        "boundingbox": _bbox(moved),
        "metadata": {"score": round(score, 3)},
    }
    if label is not None:
        detection["class"] = label
    return detection


def _jitter(rng: np.random.Generator, box: Box, amount: float) -> Box:
    """Move and scale a box by a fraction of its size, inside the image."""
    x, y, w, h = box
    w2 = w * float(np.clip(rng.normal(1, amount), 0.5, 1.5))
    h2 = h * float(np.clip(rng.normal(1, amount), 0.5, 1.5))
    x2 = float(np.clip(x + rng.normal(0, amount) * w, 0, 1 - min(w2, 1)))
    y2 = float(np.clip(y + rng.normal(0, amount) * h, 0, 1 - min(h2, 1)))
    return x2, y2, min(w2, 1 - x2), min(h2, 1 - y2)


def _jitter_points(
    rng: np.random.Generator,
    points: dict[str, LabeledPoint],
    spread: float,
    *,
    swap: bool = False,
) -> dict[str, LabeledPoint]:
    """Move each labeled point, and swap left and right when asked."""
    moved: dict[str, LabeledPoint] = {}
    for name, (x, y, visibility) in points.items():
        moved[name] = (
            (0.0, 0.0, 0)
            if visibility == 0
            else (
                float(np.clip(x + rng.normal(0, spread), 0, 1)),
                float(np.clip(y + rng.normal(0, spread), 0, 1)),
                visibility,
            )
        )
    if swap:
        for left, right in POSE_FLIP_PAIRS:
            moved[left], moved[right] = moved[right], moved[left]
    return moved


def _stereo_records(
    rng: np.random.Generator, root: Path, camera: Camera, frames: int
) -> Iterator[_Raw]:
    """Render a sequence twice, from two cameras, with its disparity."""
    baseline = 0.3
    street = random_street(rng, vehicles=5, people=5, signs=2)
    right_camera = replace(camera, offset=camera.offset + baseline)
    for step in range(frames):
        moment = street.at(step)
        left, right = moment.render(camera), moment.render(right_camera)
        disparity = camera.focal * baseline / left.depth
        columns = np.arange(camera.width)[None, :] + 0.5
        disparity[~np.isfinite(left.depth) | (columns - disparity < 0)] = 0
        stem = f"stereo_{step:03d}"
        array = root / "stereo" / f"{stem}_disparity.npy"
        np.save(array, disparity.astype(np.float16))
        detections = _detections(left)
        yield {
            "media": {
                "left": _write_image(
                    root / "stereo" / f"{stem}_left.jpg", left.image
                ),
                "right": _write_image(
                    root / "stereo" / f"{stem}_right.jpg", right.image
                ),
            },
            "annotation": {
                "vehicles": detections["vehicles"],
                "people": detections["people"],
                "disparity": [
                    {"class": "disparity", "array": {"data": str(array)}}
                ],
            },
            "sample_metadata": {
                "baseline_m": baseline,
                "focal_px": round(camera.focal, 1),
                "frame": step,
            },
        }
