import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import pycocotools.mask as mask_util
from loguru import logger
from typing_extensions import override

from luxonis_ml.data import DatasetIterator
from luxonis_ml.data.utils import COCOFormat
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import (
    Layout,
    ParseResult,
    SplitParserPlugin,
    SplitRecord,
)


def _load_annotations(annotation_path: Path) -> dict[str, Any]:
    """Decode a COCO annotation file.

    Args:
        annotation_path: Annotation JSON file.

    Returns:
        Decoded contents of the file.

    """
    with open(annotation_path) as f:
        data: dict[str, Any] = json.load(f)
    return data


class COCOParser(SplitParserPlugin):
    """Parse a directory with COCO annotations into LDF.

    Expected formats::

        dataset_dir/
        ├── train/
        │   ├── data/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── labels.json
        ├── validation/
        │   ├── data/
        │   └── labels.json
        └── test/
            ├── data/
            └── labels.json

        This is default format returned when using FiftyOne package.

    or::

        dataset_dir/
            ├── train/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            │   └── _annotations.coco.json
            ├── valid/
            └── test/

        This is one of the formats that Roboflow can generate.
    """

    dataset_types = ("coco",)

    @staticmethod
    def _detect_dataset_dir_format(
        dataset_dir: Path,
    ) -> tuple[COCOFormat | None, list[str]]:
        """Detect whether a dataset uses FiftyOne or Roboflow layout."""
        if not dataset_dir.is_dir():
            return None, []

        fiftyone_splits = ["train", "validation", "test"]
        roboflow_splits = ["train", "valid", "test"]

        existing = [d.name for d in dataset_dir.iterdir() if d.is_dir()]

        # Clash with NATIVE format
        if "val" in existing:
            return None, []

        fiftyone_splits = [s for s in fiftyone_splits if s in existing]
        roboflow_splits = [s for s in roboflow_splits if s in existing]

        if len(fiftyone_splits) != 0 and len(fiftyone_splits) == len(
            roboflow_splits
        ):
            for split_name in ("validation", "valid"):
                if split_name in existing:
                    return (
                        (COCOFormat.FIFTYONE, fiftyone_splits)
                        if split_name == "validation"
                        else (COCOFormat.ROBOFLOW, roboflow_splits)
                    )

            # Partial layouts like train-only are ambiguous by directory
            # names alone, so inspect the annotation filename inside any
            # present split to distinguish Roboflow from FiftyOne.
            for split_name in fiftyone_splits:
                split_info = COCOParser.validate_split(
                    dataset_dir / split_name
                )
                if split_info is not None:
                    annotation_name = split_info["annotation_path"].name
                    # ROBOFLOW has _annotations.coco.json while FIFTYONE has labels.json
                    if annotation_name == "_annotations.coco.json":
                        return COCOFormat.ROBOFLOW, roboflow_splits
                    return COCOFormat.FIFTYONE, fiftyone_splits

        if len(fiftyone_splits) != 0 and len(fiftyone_splits) >= len(
            roboflow_splits
        ):
            return COCOFormat.FIFTYONE, fiftyone_splits
        if len(roboflow_splits) != 0:
            return COCOFormat.ROBOFLOW, roboflow_splits
        return None, []

    @staticmethod
    def _load_coco_json(json_path: Path) -> dict[str, Any] | None:
        """Decode ``json_path`` if it has the required COCO fields."""
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        # images is required, annotations is optional (test sets don't have them)
        if "images" not in data:
            return None
        # Categories can be at top level or nested inside info
        if "categories" in data or (
            "info" in data
            and isinstance(data["info"], dict)
            and "categories" in data["info"]
        ):
            return data
        return None

    @staticmethod
    def _is_coco_json(json_path: Path) -> bool:
        """Check if JSON file has required COCO format fields."""
        return COCOParser._load_coco_json(json_path) is not None

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Return the parse arguments when ``split_path`` holds COCO labels.

        Recognizing a split decodes its whole annotation file, so the
        decoded contents are handed back with the arguments. They travel
        to the parse inside the layout, which is what keeps an annotation
        file from being read a second time.

        Args:
            split_path: Directory expected to hold one split.

        Returns:
            Arguments for `_split_records`, or ``None`` when the
            directory does not hold a COCO split.

        """
        if not split_path.exists():
            return None
        json_path = next(split_path.glob("*.json"), None)
        if not json_path:
            return None
        if json_path.name == "_annotations.coco.json":
            data = COCOParser._load_coco_json(json_path)
            if data is None:
                return None
            logger.info("Identified Roboflow format")
            return {
                "image_dir": split_path,
                "annotation_path": json_path,
                "annotation_data": data,
            }
        data = COCOParser._load_coco_json(json_path)
        if data is None:
            return None
        logger.info("Identified FiftyOne format")
        dirs = [d for d in split_path.iterdir() if d.is_dir()]
        if len(dirs) != 1:
            return None
        return {
            "image_dir": dirs[0],
            "annotation_path": json_path,
            "annotation_data": data,
        }

    @classmethod
    @override
    def detect(cls, source: Path) -> Layout | None:
        # Which directories hold splits depends on the layout variant, so
        # the FiftyOne / Roboflow detection replaces the inherited one.
        dir_format, splits = cls._detect_dataset_dir_format(source)

        discovered: dict[str | None, dict[str, Any]] = {}
        if dir_format is not None:
            for split_name in splits:
                split_kwargs = cls.validate_split(source / split_name)
                if split_kwargs is None:
                    continue
                discovered[cls._canonicalize_split_name(split_name)] = (
                    split_kwargs
                )
        if discovered:
            return Layout(discovered)

        split_kwargs = cls.validate_split(source)
        if split_kwargs is None:
            return None
        return Layout({None: split_kwargs})

    def _resolve_dir_format_and_keypoint_paths(
        self,
        dataset_dir: Path,
        use_keypoint_ann: bool,
        keypoint_ann_paths: dict[str, str] | None,
    ) -> tuple[COCOFormat, list[str], dict[str, str] | None]:
        dir_format, splits = COCOParser._detect_dataset_dir_format(dataset_dir)
        if dir_format is None:
            raise ValueError("Dataset is not in any expected format.")

        if dir_format is COCOFormat.ROBOFLOW:
            logger.warning(
                "Roboflow dataset format detected, following arguments won't be taken "
                "into account: ['use_keypoint_ann', 'keypoint_ann_paths', 'split_val_to_test']."
            )
        elif (
            dir_format is COCOFormat.FIFTYONE
            and use_keypoint_ann
            and not keypoint_ann_paths
        ):
            keypoint_ann_paths = {
                "train": "raw/person_keypoints_train2017.json",
                "val": "raw/person_keypoints_val2017.json",
                # NOTE: this file is not present by default
                "test": "raw/person_keypoints_test2017.json",
            }
        return dir_format, splits, keypoint_ann_paths

    def _resolve_split_inputs(
        self,
        dataset_dir: Path,
        layout: Layout,
        *,
        use_keypoint_ann: bool,
        keypoint_ann_paths: dict[str, str] | None,
    ) -> dict[str, dict[str, Any]]:
        """Return the parse arguments of every split, in split order.

        The layout already holds what recognizing the source revealed,
        including the annotations it decoded. Only the keypoint
        annotation files and the cleaned train annotations replace what
        it carries, and neither can be known before the parse arguments
        are.

        Args:
            dataset_dir: Root of the dataset.
            layout: Layout returned by `detect`.
            use_keypoint_ann: Whether the official COCO keypoint
                annotation files replace the recognized ones.
            keypoint_ann_paths: Keypoint annotation file per split,
                relative to ``dataset_dir``.

        Returns:
            Arguments for `_split_records` per canonical split name.

        Raises:
            ValueError: If a split directory is not in the expected
                format.

        """
        dir_format, splits, keypoint_ann_paths = (
            self._resolve_dir_format_and_keypoint_paths(
                dataset_dir,
                use_keypoint_ann=use_keypoint_ann,
                keypoint_ann_paths=keypoint_ann_paths,
            )
        )

        resolved: dict[str, dict[str, Any]] = {}
        for split_name in splits:
            canonical_name = self._canonicalize_split_name(split_name)
            split_kwargs = layout.splits.get(canonical_name)
            if split_kwargs is None:
                raise ValueError(
                    f"{split_name.title()} split not in expected format"
                )

            annotation_path = split_kwargs["annotation_path"]
            annotation_data = split_kwargs["annotation_data"]

            if (
                keypoint_ann_paths
                and use_keypoint_ann
                and dir_format is COCOFormat.FIFTYONE
            ):
                if canonical_name == "test":
                    kp_path = dataset_dir / keypoint_ann_paths["test"]
                    if not kp_path.exists():
                        logger.warning(
                            f"Keypoint annotation file not found: {kp_path}. "
                            "Skipping test split."
                        )
                        continue
                    annotation_path = kp_path
                else:
                    annotation_path = (
                        dataset_dir / keypoint_ann_paths[canonical_name]
                    )
                # A keypoint annotation file replaces the one the split
                # was recognized by, so it is not the decoded one.
                annotation_data = None

            if canonical_name == "train":
                if annotation_data is None:
                    annotation_data = _load_annotations(annotation_path)
                # Cleaning filters in place, so the decoded annotations
                # stay the contents of the path it hands back.
                annotation_path = clean_annotations(
                    annotation_path, annotation_data
                )

            resolved[canonical_name] = {
                "image_dir": split_kwargs["image_dir"],
                "annotation_path": annotation_path,
                "annotation_data": annotation_data,
            }
        return resolved

    @staticmethod
    def _val_files_moved_to_test(
        val_files: Sequence[Path],
        *,
        has_test_files: bool,
        split_val_to_test: bool,
    ) -> list[Path]:
        """Return the validation images imported as the test split.

        A COCO test split ships without annotations and then contributes
        no image at all, in which case the validation images are halved
        and the second half stands in for the test split.

        Args:
            val_files: Images of the validation split, in the order the
                records report them. Empty when the split does not have
                to be halved.
            has_test_files: Whether the test split contributes an image.
            split_val_to_test: Whether halving is allowed at all.

        Returns:
            The validation images imported as the test split.

        """
        if has_test_files:
            return []
        if not split_val_to_test:
            logger.warning(
                "Sampling from the test set cannot be done since the "
                "labels are missing. This is expected for COCO datasets "
                "where the test set annotations are not publicly available."
            )
            return []
        split_point = round(len(val_files) * 0.5)
        return list(val_files[split_point:])

    def _iter_kept_annotations(
        self,
        annotations: list[dict[str, Any]],
        *,
        categories: dict[Any, str],
        file: Path,
        annotation_path: Path,
    ) -> Iterator[
        tuple[dict[str, Any], str, tuple[float, float, float, float]]
    ]:
        """Yield the annotations of one image that produce a record.

        Listing a split's files and streaming its records have to agree
        on exactly which annotations survive, so the checks deciding
        that live in one place. Reporting a skipped annotation twice is
        free: the collector keeps every distinct issue once.

        Args:
            annotations: Annotations referencing one image.
            categories: Class name per COCO category id.
            file: Resolved path of the image, used for reporting.
            annotation_path: Annotation file, used for reporting.

        Yields:
            The annotation, its class name and its bounding box.

        """
        for ann in annotations:
            if ann.get("iscrowd"):
                self._warn_skipped_annotation(
                    ParserIssue.COCO_ISCROWD,
                    "COCO annotation has iscrowd=1",
                    source=annotation_path,
                    image=file,
                    annotation_id=ann.get("id"),
                )
                continue
            class_name = categories[ann["category_id"]]

            try:
                # Unpacked one by one rather than through a generator: a
                # box that is not four numbers still leaves through
                # `TypeError` or `ValueError`, whichever of the two steps
                # raises it.
                raw_x, raw_y, raw_w, raw_h = ann["bbox"]
                x = float(raw_x)
                y = float(raw_y)
                w = float(raw_w)
                h = float(raw_h)
                valid_bbox = (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and math.isfinite(w)
                    and math.isfinite(h)
                )
            except (TypeError, ValueError):
                valid_bbox = False

            if not valid_bbox:
                self._warn_skipped_annotation(
                    ParserIssue.NON_NUMERIC_ANNOTATION,
                    "Annotation contains non-numeric bbox values",
                    source=annotation_path,
                    image=file,
                    annotation_id=ann.get("id"),
                )
                continue

            yield ann, class_name, (x, y, w, h)

    def _iter_split_files(
        self,
        image_dir: Path,
        annotation_path: Path,
        annotation_data: dict[str, Any] | None = None,
    ) -> Iterator[Path]:
        """Yield the images of one split that contribute a record.

        The image table is the index the parse walks anyway, so a split
        can be listed without decoding a single segmentation: a path
        resolution, an existence check, and the very same annotation
        filter the records are built from.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.
            annotation_data: Decoded contents of ``annotation_path``, as
                carried by the layout. Read from disk when not given.

        Yields:
            The images of the split, in the order the records report
            them.

        """
        if annotation_data is None:
            annotation_data = _load_annotations(annotation_path)

        coco_categories = annotation_data.get("categories", [])
        categories = {cat["id"]: cat["name"] for cat in coco_categories}
        # Only whether the split has skeletons matters here: it decides
        # whether an image without annotations contributes a record.
        has_skeletons = any(
            "keypoints" in cat and "skeleton" in cat for cat in coco_categories
        )

        ann_dict: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for ann in annotation_data.get("annotations", []):
            ann_dict[ann["image_id"]].append(ann)

        base_dir = image_dir.absolute().resolve()
        img_dict = {img["id"]: img for img in annotation_data["images"]}
        for img_id, img in img_dict.items():
            file = resolve_manifest_path(base_dir, img["file_name"])
            if not file.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=annotation_path,
                    image=file,
                )
                continue

            img_anns = ann_dict.get(img_id)
            if not img_anns:
                if not has_skeletons:
                    yield file
                continue

            # The records report every skipped annotation of the image,
            # so the first kept one already answers whether it is listed.
            kept = self._iter_kept_annotations(
                img_anns,
                categories=categories,
                file=file,
                annotation_path=annotation_path,
            )
            if next(kept, None) is not None:
                yield file

    def _split_files(
        self,
        image_dir: Path,
        annotation_path: Path,
        annotation_data: dict[str, Any] | None = None,
    ) -> list[Path]:
        """List the images of one split from its annotation index.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.
            annotation_data: Decoded contents of ``annotation_path``, as
                carried by the layout. Read from disk when not given.

        Returns:
            The images of the split, in the order the records report
            them.

        """
        return list(
            self._iter_split_files(image_dir, annotation_path, annotation_data)
        )

    def _split_records(
        self,
        image_dir: Path,
        annotation_path: Path,
        annotation_data: dict[str, Any] | None = None,
    ) -> DatasetIterator:
        """Stream COCO annotations of one split as LDF records.

        Annotations include classification, segmentation, object detection,
        and keypoints when present.

        Args:
            image_dir: Directory with images.
            annotation_path: Annotation JSON file.
            annotation_data: Decoded contents of ``annotation_path``, as
                carried by the layout. Read from disk when not given.

        Returns:
            One record per kept annotation, and one per unannotated
            image while the split declares no skeleton.

        """
        if annotation_data is None:
            annotation_data = _load_annotations(annotation_path)

        coco_images = annotation_data["images"]
        coco_annotations = annotation_data.get("annotations", [])
        coco_categories = annotation_data.get("categories", [])
        categories = {cat["id"]: cat["name"] for cat in coco_categories}

        skeletons = {}
        for cat in coco_categories:
            if "keypoints" in cat and "skeleton" in cat:
                skeletons[categories[cat["id"]]] = {
                    "labels": cat["keypoints"],
                    "edges": list(
                        map(tuple, (np.array(cat["skeleton"]) - 1).tolist())
                    ),
                }
        self._skeletons.update(skeletons)

        ann_dict: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco_annotations:
            ann_dict[ann["image_id"]].append(ann)

        # Keying the image table by id keeps the last entry of a
        # duplicated id while preserving the order of the first. Listing
        # the split does the same, so the two agree on which images the
        # split holds and in which order.
        img_dict = {img["id"]: img for img in coco_images}
        base_dir = image_dir.absolute().resolve()

        def generator() -> DatasetIterator:
            existing_ids: set[Any] | None = None
            next_fallback_id = 0

            for img_id, img in img_dict.items():
                file = resolve_manifest_path(base_dir, img["file_name"])
                if not file.exists():
                    self._warn_skipped_annotation(
                        ParserIssue.MISSING_IMAGE,
                        "referenced image file does not exist",
                        source=annotation_path,
                        image=file,
                    )
                    continue

                img_anns = ann_dict.get(img_id)

                if not img_anns:
                    if skeletons:
                        # Keypoint annotations: skip images with no labels
                        continue
                    # Register image with no annotations (valid COCO case)
                    yield {"file": file, "annotation": None}
                    continue

                img_h = img["height"]
                img_w = img["width"]

                for ann, class_name, box in self._iter_kept_annotations(
                    img_anns,
                    categories=categories,
                    file=file,
                    annotation_path=annotation_path,
                ):
                    x, y, w, h = box

                    segmentation = None

                    coco_seg = ann.get("segmentation", [])

                    if isinstance(coco_seg, list) and coco_seg:
                        rles = mask_util.frPyObjects(coco_seg, img_h, img_w)
                        rle = mask_util.merge(rles)
                        segmentation = {
                            "height": rle["size"][0],
                            "width": rle["size"][1],
                            "counts": rle["counts"],
                        }
                    elif isinstance(coco_seg, dict):
                        segmentation = {
                            "height": coco_seg["size"][0],
                            "width": coco_seg["size"][1],
                            "counts": coco_seg["counts"],
                        }

                    if "id" in ann:
                        instance_id = ann["id"]
                    else:
                        if existing_ids is None:
                            # Only a missing id needs to know which ids
                            # are taken, so annotation files that always
                            # carry one never pay for collecting them.
                            existing_ids = {
                                other["id"]
                                for other in coco_annotations
                                if "id" in other
                            }
                        while next_fallback_id in existing_ids:
                            next_fallback_id += 1
                        instance_id = next_fallback_id
                        existing_ids.add(instance_id)
                        next_fallback_id += 1

                    record = {
                        "file": file,
                        "annotation": {
                            "class": class_name,
                            "instance_id": instance_id,
                            "boundingbox": {
                                "x": x / img_w,
                                "y": y / img_h,
                                "w": w / img_w,
                                "h": h / img_h,
                            },
                        },
                    }

                    if segmentation is not None:
                        record["annotation"]["segmentation"] = segmentation
                        record["annotation"]["instance_segmentation"] = (
                            segmentation
                        )

                    if "keypoints" in ann:
                        kpts = np.array(ann["keypoints"]).reshape(-1, 3)

                        np.clip(kpts[:, 0], 0, img_w, out=kpts[:, 0])
                        np.clip(kpts[:, 1], 0, img_h, out=kpts[:, 1])

                        keypoints = []
                        for kp in kpts:
                            keypoints.append(
                                (kp[0] / img_w, kp[1] / img_h, int(kp[2]))
                            )

                        record["annotation"]["keypoints"] = {
                            "keypoints": keypoints
                        }

                    yield record

        return generator()

    def _reject_single_split_options(
        self,
        source: Path,
        use_keypoint_ann: bool,
        keypoint_ann_paths: dict[str, str] | None,
    ) -> None:
        """Reject the options that only a split-based source supports.

        ``split_val_to_test`` is meaningless for a single split, but the
        keypoint options would be silently ignored, so they are refused
        rather than dropped.

        Raises:
            ValueError: If a keypoint option was given.

        """
        unsupported = [
            name
            for name, value in (
                ("use_keypoint_ann", use_keypoint_ann),
                ("keypoint_ann_paths", keypoint_ann_paths),
            )
            if value
        ]
        if unsupported:
            raise ValueError(
                f"COCO options {unsupported} are only supported for "
                "sources containing split directories, not for the "
                f"single split '{source}'."
            )

    @override
    def parse(
        self,
        source: Path,
        layout: Layout,
        *,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: dict[str, str] | None = None,
        split_val_to_test: bool = True,
        **kwargs: Any,
    ) -> ParseResult:
        """Stream the records of every split of a COCO source.

        Args:
            source: Root of the dataset.
            layout: Layout returned by `detect`.
            use_keypoint_ann: Whether the official COCO keypoint
                annotation files replace the recognized ones.
            keypoint_ann_paths: Keypoint annotation file per split,
                relative to ``source``. Defaults to where the official
                COCO release keeps them.
            split_val_to_test: Whether the second half of the validation
                split is imported as the test split when the test split
                contributes no image.
            kwargs: Arguments forwarded to a single-split parse.

        Returns:
            Records tagged with the split they belong to.

        """
        if not layout.split_names:
            self._reject_single_split_options(
                source, use_keypoint_ann, keypoint_ann_paths
            )
            return super().parse(source, layout, **kwargs)

        split_inputs = self._resolve_split_inputs(
            source,
            layout,
            use_keypoint_ann=use_keypoint_ann,
            keypoint_ann_paths=keypoint_ann_paths,
        )

        test_kwargs = split_inputs.get("test")
        # The first image the test split would report already answers
        # whether it contributes anything, so a test split that has
        # images is never listed in full.
        has_test_files = test_kwargs is not None and (
            next(self._iter_split_files(**test_kwargs), None) is not None
        )
        val_kwargs = split_inputs.get("val")
        val_files = (
            self._split_files(**val_kwargs)
            if val_kwargs is not None
            and split_val_to_test
            and not has_test_files
            else []
        )
        moved_to_test = set(
            self._val_files_moved_to_test(
                val_files,
                has_test_files=has_test_files,
                split_val_to_test=split_val_to_test,
            )
        )

        def records() -> Iterator[SplitRecord]:
            for split_name, split_kwargs in split_inputs.items():
                if split_name == "val" and moved_to_test:
                    for record in self._split_records(**split_kwargs):
                        file = cast(dict[str, Any], record)["file"]
                        yield (
                            "test" if file in moved_to_test else "val",
                            record,
                        )
                else:
                    for record in self._split_records(**split_kwargs):
                        yield split_name, record

        return ParseResult(records(), self._skeletons)

    @override
    def enumerate_files(
        self,
        source: Path,
        layout: Layout,
        *,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: dict[str, str] | None = None,
        split_val_to_test: bool = True,
        **kwargs: Any,
    ) -> dict[str | None, list[Path]] | None:
        """List the images of every split from the annotation index.

        Args:
            source: Root of the dataset.
            layout: Layout returned by `detect`.
            use_keypoint_ann: Whether the official COCO keypoint
                annotation files replace the recognized ones.
            keypoint_ann_paths: Keypoint annotation file per split,
                relative to ``source``.
            split_val_to_test: Whether the second half of the validation
                split is imported as the test split when the test split
                contributes no image.
            kwargs: Arguments forwarded to a single-split enumeration.

        Returns:
            The images of each split, split exactly the way the records
            are tagged.

        """
        if not layout.split_names:
            self._reject_single_split_options(
                source, use_keypoint_ann, keypoint_ann_paths
            )
            return super().enumerate_files(source, layout, **kwargs)

        split_inputs = self._resolve_split_inputs(
            source,
            layout,
            use_keypoint_ann=use_keypoint_ann,
            keypoint_ann_paths=keypoint_ann_paths,
        )
        enumerated: dict[str | None, list[Path]] = {
            split_name: self._split_files(**split_kwargs)
            for split_name, split_kwargs in split_inputs.items()
        }

        # Reported the way `parse` tags the records; the warning a
        # forbidden halving logs belongs to the parse, not to a listing.
        if split_val_to_test:
            moved = self._val_files_moved_to_test(
                enumerated.get("val", []),
                has_test_files=bool(enumerated.get("test")),
                split_val_to_test=True,
            )
            if moved:
                moved_set = set(moved)
                enumerated["val"] = [
                    file for file in enumerated["val"] if file not in moved_set
                ]
                enumerated["test"] = [*enumerated.get("test", []), *moved]
        return enumerated


#: Containers with more children than this are written one child at a
#: time, so that encoding a large annotation file never holds its whole
#: encoded text in memory.
_JSON_CHUNK_LIMIT = 256


def _write_json(file: TextIO, value: Any) -> None:
    """Write ``value`` the way ``json.dump`` would, in chunks.

    ``json.dump`` asks its encoder for an iterator of chunks, the one
    mode the C encoder does not implement, so it falls back to the
    Python encoder and takes several times longer than ``json.dumps``.
    Encoding one child at a time reaches the C encoder while keeping the
    text held at once proportional to a single child rather than to the
    whole file. Every chunk is produced by the same encoder with the
    same defaults, so the file is byte for byte what ``json.dump``
    writes.

    Args:
        file: Text file to write to.
        value: Value to encode.

    """
    if isinstance(value, dict) and value:
        separator = "{"
        for key, child in value.items():
            if _is_large(child):
                # `json.dumps({key: 0})` is `{<key>: 0}`, so dropping the
                # brace and the placeholder leaves the key exactly as
                # `json.dump` writes it, including how it coerces keys
                # that are not strings.
                file.write(separator + json.dumps({key: 0})[1:-2])
                _write_json(file, child)
            else:
                file.write(separator + json.dumps({key: child})[1:-1])
            separator = ", "
        file.write("}")
    elif isinstance(value, list) and value:
        separator = "["
        for child in value:
            file.write(separator)
            if _is_large(child):
                _write_json(file, child)
            else:
                file.write(json.dumps(child))
            separator = ", "
        file.write("]")
    else:
        file.write(json.dumps(value))


def _is_large(value: Any) -> bool:
    """Return whether ``value`` is worth writing one child at a time."""
    return isinstance(value, (dict, list)) and len(value) > _JSON_CHUNK_LIMIT


def clean_annotations(
    annotation_path: Path,
    annotation_data: dict[str, Any] | None = None,
) -> Path:
    """Remove COCO images that are known to cause parsing issues.

    Args:
        annotation_path: Annotation JSON file.
        annotation_data: Decoded contents of ``annotation_path``. Read
            from disk when not given. The images and annotations are
            filtered in place, so a caller passing them in is left with
            the cleaned data.

    Returns:
        Path to the cleaned annotation JSON file.

    """
    files_to_avoid = {
        "000000341448.jpg",
        "000000279522.jpg",
        "000000090169.jpg",
        "000000321238.jpg",
        "000000242807.jpg",
        "000000297126.jpg",
        "000000411274.jpg",
        "000000407259.jpg",
        "000000446141.jpg",
        "000000373199.jpg",
        "000000410810.jpg",
        "000000397819.jpg",
        "000000578492.jpg",
        "000000531721.jpg",
    }
    if annotation_data is None:
        annotation_data = _load_annotations(annotation_path)

    filtered_images = [
        img
        for img in annotation_data["images"]
        if img["file_name"] not in files_to_avoid
    ]

    if len(filtered_images) == len(annotation_data["images"]):
        return annotation_path

    filtered_image_ids = {img["id"] for img in filtered_images}
    filtered_annotations = [
        ann
        for ann in annotation_data["annotations"]
        if ann["image_id"] in filtered_image_ids
    ]

    annotation_data["images"] = filtered_images
    annotation_data["annotations"] = filtered_annotations

    cleaned_annotation_path = annotation_path.with_name("labels_fixed.json")
    with open(cleaned_annotation_path, "w") as f:
        _write_json(f, annotation_data)

    return cleaned_annotation_path
