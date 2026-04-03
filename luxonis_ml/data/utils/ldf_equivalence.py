import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import polars as pl
from pycocotools import mask as mask_utils

from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.utils.environ import environ

if TYPE_CHECKING:
    from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset

LDFCollector = Callable[[PreparedLDF], Any]


def ldf_equivalent(
    previous_dataset: Union[str, "LuxonisDataset"],
    new_dataset: Union[str, "LuxonisDataset"],
) -> bool:
    return LDFEquivalence.ldf_equivalent(previous_dataset, new_dataset)


class LDFEquivalence:
    @staticmethod
    def file_sha256(path: Path) -> str:
        """The image's hash is used to order annotations to survive
        renaming."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def ldf_equivalent(
        cls,
        previous_dataset: Union[str, "LuxonisDataset"],
        new_dataset: Union[str, "LuxonisDataset"],
    ) -> bool:
        try:
            previous_ldf = cls._prepared_ldf_from_public_input(
                previous_dataset
            )
            new_ldf = cls._prepared_ldf_from_public_input(new_dataset)
            cls.assert_equivalence(previous_ldf, new_ldf)
        except AssertionError:
            return False
        return True

    @classmethod
    def equivalent(
        cls,
        previous_dataset: Union[str, "LuxonisDataset"],
        new_dataset: Union[str, "LuxonisDataset"],
    ) -> bool:
        return cls.ldf_equivalent(previous_dataset, new_dataset)

    @classmethod
    def assert_equivalence(
        cls,
        previous: Any,
        new: Any,
        collector: LDFCollector | None = None,
    ) -> None:
        previous_ldf = cls._to_prepared_ldf(previous)
        new_ldf = cls._to_prepared_ldf(new)

        if collector is not None:
            cls._assert_collected_equivalence(previous_ldf, new_ldf, collector)
            return

        assert cls.collect_image_multiset(
            previous_ldf
        ) == cls.collect_image_multiset(new_ldf), "Different image sets"

        for task_type in sorted(
            cls._task_types(previous_ldf) | cls._task_types(new_ldf)
        ):
            if task_type == "boundingbox":
                cls._assert_collected_equivalence(
                    previous_ldf, new_ldf, cls.collect_bbox_multiset
                )
            elif task_type == "classification":
                cls._assert_collected_equivalence(
                    previous_ldf, new_ldf, cls.collect_classification_multiset
                )
            elif task_type == "instance_segmentation":
                cls._assert_collected_equivalence(
                    previous_ldf,
                    new_ldf,
                    cls.collect_instance_segmentation_mask_overlap_multiset,
                )
            elif task_type == "keypoints":
                cls._assert_collected_equivalence(
                    previous_ldf, new_ldf, cls.collect_keypoint_multiset
                )
            elif task_type == "segmentation":
                cls._assert_collected_equivalence(
                    previous_ldf,
                    new_ldf,
                    cls.collect_segmentation_mask_overlap_multiset,
                )
            else:
                assert cls.collect_annotation_multiset(
                    previous_ldf, task_type
                ) == cls.collect_annotation_multiset(new_ldf, task_type), (
                    f"Different annotations for task type '{task_type}'"
                )

    @staticmethod
    def multiset_equal_with_tolerance(
        prev_map: dict[tuple[str], Counter],
        new_map: dict[tuple[str], Counter],
        tol: float,
    ) -> None:
        assert prev_map.keys() == new_map.keys(), (
            f"Different image sets:\nprev-only={set(prev_map) - set(new_map)}\n"
            f"new-only={set(new_map) - set(prev_map)}"
        )

        for key, prev_counter in prev_map.items():
            new_counter = new_map[key]

            if prev_counter == new_counter:
                continue

            prev_list = list(prev_counter.elements())
            new_list = list(new_counter.elements())

            assert len(prev_list) == len(new_list), (
                f"Different number of boxes for {key}: "
                f"{len(prev_list)} vs {len(new_list)}"
            )

            used = [False] * len(new_list)

            def within_tol(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
                return all(
                    abs(aa - bb) <= tol for aa, bb in zip(a, b, strict=True)
                )

            for box in prev_list:
                found = False
                for j, cand in enumerate(new_list):
                    if not used[j] and within_tol(box, cand):
                        used[j] = True
                        found = True
                        break
                assert found, (
                    f"No match within tol={tol} for box {box} in image {key}. "
                    f"Unmatched candidates: "
                    f"{[c for u, c in zip(used, new_list, strict=True) if not u]}"
                )

            assert all(used), (
                f"Extra unmatched boxes in new for {key}: "
                f"{[c for u, c in zip(used, new_list, strict=True) if not u]}"
            )

    @classmethod
    def collect_image_multiset(cls, prepared_ldf: PreparedLDF) -> Counter[str]:
        out: Counter[str] = Counter()
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, _entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            out.update((cls.file_sha256(Path(file_path)),))
        return out

    @classmethod
    def collect_annotation_multiset(
        cls,
        prepared_ldf: PreparedLDF,
        task_type: str,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            annotations = []
            for row in entry.iter_rows(named=True):
                if row["task_type"] != task_type:
                    continue
                annotations.append(
                    (
                        row["task_name"],
                        row["class_name"],
                        row["instance_id"],
                        cls._canonicalize_annotation(row["annotation"]),
                    )
                )
            if annotations:
                out.setdefault(hashed_key, Counter()).update(annotations)
        return out

    @classmethod
    def collect_bbox_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            boxes = []
            for row in entry.iter_rows(named=True):
                if row["task_type"] == "boundingbox":
                    d = json.loads(row["annotation"])
                    boxes.append(
                        (
                            round(d["x"], 2),
                            round(d["y"], 2),
                            round(d["w"], 2),
                            round(d["h"], 2),
                        )
                    )
            if boxes:
                out.setdefault(hashed_key, Counter()).update(boxes)
        return out

    @classmethod
    def collect_keypoint_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            keypoints = []
            for row in entry.iter_rows(named=True):
                if row["task_type"] == "keypoints":
                    d = json.loads(row["annotation"])
                    for kp in d["keypoints"]:
                        rounded_kp = tuple(
                            round(v, 3) if i < 2 else v
                            for i, v in enumerate(kp)
                        )
                        keypoints.append(rounded_kp)
            if keypoints:
                out.setdefault(hashed_key, Counter()).update(keypoints)
        return out

    @classmethod
    def collect_instance_segmentation_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            counts = []
            for row in entry.iter_rows(named=True):
                if row["task_type"] == "instance_segmentation":
                    d = json.loads(row["annotation"])
                    counts.extend(d["counts"])
            if counts:
                out.setdefault(hashed_key, Counter()).update(counts)
        return out

    @classmethod
    def collect_instance_segmentation_mask_overlap_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], list[np.ndarray]]:
        out: dict[tuple[str], list[np.ndarray]] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            masks: list[np.ndarray] = []

            for row in entry.iter_rows(named=True):
                if row["task_type"] == "instance_segmentation":
                    d = json.loads(row["annotation"])
                    if isinstance(d.get("counts"), str):
                        rle = {
                            "counts": d["counts"].encode("utf-8"),
                            "size": [d["height"], d["width"]],
                        }
                        mask = mask_utils.decode(rle).astype(bool)  # type: ignore
                        masks.append(mask)

            if masks:
                out.setdefault(hashed_key, []).extend(masks)

        return out

    @classmethod
    def collect_classification_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            classes = []
            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "classification"
                    and row["instance_id"] == -1
                ):
                    classes.append(row["class_name"])
            if classes:
                out.setdefault(hashed_key, Counter()).update(classes)
        return out

    @classmethod
    def collect_segmentation_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (cls.file_sha256(Path(file_path)),)
            classes = []
            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "segmentation"
                    and row["instance_id"] == -1
                ):
                    classes.append(row["class_name"])
            if classes:
                out.setdefault(hashed_key, Counter()).update(classes)
        return out

    @classmethod
    def collect_segmentation_mask_overlap_multiset(
        cls,
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str, str], list[np.ndarray]]:
        out: dict[tuple[str, str], list[np.ndarray]] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            img_hash = cls.file_sha256(Path(file_path))

            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "segmentation"
                    and row["instance_id"] == -1
                ):
                    d = json.loads(row["annotation"])
                    class_name = row["class_name"]
                    if isinstance(d.get("counts"), str):
                        rle = {
                            "counts": d["counts"].encode("utf-8"),
                            "size": [d["height"], d["width"]],
                        }
                        mask = mask_utils.decode(rle).astype(bool)  # type: ignore
                        out.setdefault((img_hash, class_name), []).append(mask)

        return out

    @staticmethod
    def _canonicalize_annotation(annotation: str | None) -> str | None:
        if annotation is None:
            return None
        try:
            decoded = json.loads(annotation)
        except json.JSONDecodeError:
            return annotation
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _combine_masks(masks: list[np.ndarray]) -> np.ndarray:
        if not masks:
            return np.zeros((0, 0), dtype=bool)
        combined = masks[0].astype(bool).copy()
        for mask in masks[1:]:
            if mask.shape != combined.shape:
                raise AssertionError(
                    f"Mask shape mismatch: {mask.shape} vs {combined.shape}"
                )
            combined |= mask.astype(bool)
        return combined

    @classmethod
    def _assert_collected_equivalence(
        cls,
        previous_ldf: PreparedLDF,
        new_ldf: PreparedLDF,
        collector: LDFCollector,
    ) -> None:
        prev = collector(previous_ldf)
        new = collector(new_ldf)

        if collector.__name__ == "collect_bbox_multiset":
            cls.multiset_equal_with_tolerance(prev, new, tol=0.02)
        elif collector.__name__ in (
            "collect_instance_segmentation_mask_overlap_multiset",
            "collect_segmentation_mask_overlap_multiset",
        ):
            assert prev.keys() == new.keys(), (
                "Different image sets:\n"
                f"prev-only={set(prev) - set(new)}\n"
                f"new-only={set(new) - set(prev)}"
            )

            for key in prev:
                comb_prev = cls._combine_masks(prev[key])
                comb_new = cls._combine_masks(new[key])

                if comb_prev.size == 0 or comb_new.size == 0:
                    assert comb_prev.size == comb_new.size, (
                        f"Mask presence differs for {key}"
                    )
                    continue

                assert comb_prev.shape == comb_new.shape, (
                    f"Combined mask shape differs for {key}"
                )

                intersection = np.logical_and(comb_prev, comb_new).sum()
                union = np.logical_or(comb_prev, comb_new).sum()
                if union == 0:
                    continue

                iou = intersection / union

                assert iou > 0.75, f"Low IoU in combined masks for {key}"
        else:
            assert prev == new

    @staticmethod
    def _task_types(prepared_ldf: PreparedLDF) -> set[str]:
        return set(
            prepared_ldf.processed_df.select("task_type")
            .drop_nulls()
            .to_series()
            .to_list()
        )

    @staticmethod
    def _to_prepared_ldf(ldf_like: Any) -> PreparedLDF:
        if isinstance(ldf_like, PreparedLDF):
            return ldf_like

        from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset

        if isinstance(ldf_like, LuxonisDataset):
            return PreparedLDF.from_dataset(ldf_like)

        raise TypeError(
            "Expected a PreparedLDF or LuxonisDataset for LDF comparison."
        )

    @staticmethod
    def _prepared_ldf_from_name(dataset_name: str) -> PreparedLDF:
        dataset_path = LDFEquivalence._local_dataset_path(dataset_name)
        if dataset_path is None:
            raise FileNotFoundError(
                f"Local dataset '{dataset_name}' does not exist."
            )
        return LDFEquivalence._prepared_ldf_from_local_path(dataset_path)

    @staticmethod
    def _prepared_ldf_from_public_input(
        dataset: Union[str, "LuxonisDataset"],
    ) -> PreparedLDF:
        if isinstance(dataset, str):
            return LDFEquivalence._prepared_ldf_from_name(dataset)

        from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset

        if isinstance(dataset, LuxonisDataset):
            return PreparedLDF.from_dataset(dataset)

        raise TypeError(
            "ldf_equivalent() expects dataset names or LuxonisDataset instances."
        )

    @staticmethod
    def _local_dataset_path(dataset_name: str) -> Path | None:
        datasets_root = environ.LUXONISML_BASE_PATH / "data"
        candidates = sorted(datasets_root.glob(f"*/datasets/{dataset_name}"))
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple local datasets named '{dataset_name}' were found: "
                f"{[str(path) for path in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _prepared_ldf_from_local_path(dataset_path: Path) -> PreparedLDF:
        splits_path = dataset_path / "metadata" / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_path.name}' does not have splits metadata."
            )
        splits = json.loads(splits_path.read_text())

        annotation_files = sorted(
            (dataset_path / "annotations").glob("*.parquet")
        )
        if not annotation_files:
            raise FileNotFoundError(
                f"Dataset '{dataset_path.name}' does not contain annotations."
            )

        df = pl.scan_parquet(
            [str(path) for path in annotation_files]
        ).collect()
        if df.is_empty():
            raise FileNotFoundError(f"Dataset '{dataset_path.name}' is empty.")

        df = df.with_row_count("row_idx")
        media_path = dataset_path / "media"

        def resolve_path(
            img_path: str | Path, uuid: str, local_media_path: Path
        ) -> str:
            path = Path(img_path)
            if path.exists():
                return str(path)
            fallback = local_media_path / f"{uuid}{path.suffix}"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing image: {fallback}")
            return str(fallback)

        df = df.with_columns(
            pl.struct(["file", "uuid"])
            .map_elements(
                lambda row: resolve_path(row["file"], row["uuid"], media_path),
                return_dtype=pl.Utf8,
            )
            .alias("file")
        )

        grouped_image_sources = df.select(
            "group_id", "source_name", "file"
        ).unique()

        annotated = df.filter(pl.col("annotation").is_not_null())
        annotated_groups = annotated.select("group_id").unique()["group_id"]
        noann_first = df.filter(
            ~pl.col("group_id").is_in(annotated_groups)
        ).unique(subset=["group_id"], keep="first")

        processed_df = (
            pl.concat([annotated, noann_first]).sort("row_idx").drop("row_idx")
        )

        return PreparedLDF(
            splits=splits,
            processed_df=processed_df,
            grouped_image_sources=grouped_image_sources,
        )
