import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from luxonis_ml.data import LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset
from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.data.utils.enums import BucketStorage


def gather_tasks(dataset: LuxonisDataset) -> set[str]:
    return {
        f"{task_name}/{task_type}"
        for task_name, task_types in dataset.get_tasks().items()
        for task_type in task_types
    }


def create_image(i: int, dir: Path) -> Path:
    path = dir / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


def get_loader_output(loader: LuxonisLoader) -> set[str]:
    all_labels = set()
    for _, labels in loader:
        all_labels.update(labels.keys())
    return all_labels


def create_dataset(
    dataset_name: str,
    generator: DatasetIterator,
    bucket_storage: BucketStorage = BucketStorage.LOCAL,
    *,
    splits: bool | dict[str, float] | tuple = True,
    delete_local: bool = True,
    delete_remote: bool = True,
    **kwargs,
) -> LuxonisDataset:
    dataset = LuxonisDataset(
        dataset_name,
        delete_local=delete_local,
        delete_remote=delete_remote,
        bucket_storage=bucket_storage,
        **kwargs,
    ).add(generator)
    if splits is True:
        dataset.make_splits()
    elif splits:
        dataset.make_splits(splits)
    return dataset


class LDFEquivalence:
    @staticmethod
    def file_sha256(path: Path) -> str:
        """The image's hash is used to order annotations to survive
        renaming."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

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

    @staticmethod
    def assert_equivalence(
        dataset: LuxonisDataset,
        new_dataset: LuxonisDataset,
        collector: Callable,
    ) -> None:
        previous_ldf = PreparedLDF.from_dataset(dataset)
        new_ldf = PreparedLDF.from_dataset(new_dataset)
        prev = collector(previous_ldf)
        new = collector(new_ldf)

        if collector.__name__ == "collect_bbox_multiset":
            LDFEquivalence.multiset_equal_with_tolerance(prev, new, tol=0.02)
        elif (
            collector.__name__
            == "collect_instance_segmentation_mask_overlap_multiset"
        ):
            # Combine all masks per image and assert there is spatial overlap.
            def _combine_masks(masks: list[np.ndarray]) -> np.ndarray:
                if not masks:
                    return np.zeros((0, 0), dtype=bool)
                combined = masks[0].astype(bool).copy()
                for m in masks[1:]:
                    if m.shape != combined.shape:
                        raise AssertionError(
                            f"Mask shape mismatch: {m.shape} vs {combined.shape}"
                        )
                    combined |= m.astype(bool)
                return combined

            assert prev.keys() == new.keys(), (
                f"Different image sets:\nprev-only={set(prev) - set(new)}\nnew-only={set(new) - set(prev)}"
            )

            for key in prev:
                comb_prev = _combine_masks(prev[key])
                comb_new = _combine_masks(new[key])

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
                iou = intersection / union if union > 0 else 0.0

                assert iou > 0.75, f"Low IoU in combined masks for {key}"
        else:
            assert prev == new

    @staticmethod
    def collect_bbox_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
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

    @staticmethod
    def collect_keypoint_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
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

    @staticmethod
    def collect_instance_segmentation_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
            counts = []
            for row in entry.iter_rows(named=True):
                if row["task_type"] == "instance_segmentation":
                    d = json.loads(row["annotation"])
                    counts.extend(d["counts"])
            if counts:
                out.setdefault(hashed_key, Counter()).update(counts)
        return out

    @staticmethod
    def collect_instance_segmentation_mask_overlap_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], list[np.ndarray]]:
        out: dict[tuple[str], list[np.ndarray]] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
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

    @staticmethod
    def collect_classification_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
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

    @staticmethod
    def collect_segmentation_multiset(
        prepared_ldf: PreparedLDF,
    ) -> dict[tuple[str], Counter]:
        out: dict[tuple[str], Counter] = {}
        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )
        for key, entry in grouped:
            file_path, _gid = cast(tuple[str, str], key)
            hashed_key = (LDFEquivalence.file_sha256(Path(file_path)),)
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
