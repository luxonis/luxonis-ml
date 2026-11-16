"""Export to a classification directory layout, one folder per class."""

from pathlib import Path
from typing import Any, cast

from typing_extensions import override

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
    whole_image_classes,
)
from luxonis_ml.enums import DatasetType


class ClassificationDirectoryExporter(BaseExporter):
    """Export whole-image classes as one directory per class.

    Output structure::

        <dataset_name>/
            train/
                cat/
                    0.jpg
                    1.jpg
                dog/
                    0.jpg
            valid/
            test/

    An image with more than one class gets a copy with the same name in
    each of its class directories. ``0.jpg`` above is a ``cat`` and a
    ``dog``. `ClassificationDirectoryParser` reads these copies back as
    one image with all of its classes. An image without a class is
    skipped.

    Only classes with the instance ID ``-1`` are exported. The classes of
    boxes, masks and keypoints are not.
    """

    def get_split_names(self) -> dict[str, str]:
        """Return the directory name of each split.

        Returns:
            Split names mapped to directory names. The ``val`` split
            becomes ``valid``.

        """
        return {"train": "train", "val": "valid", "test": "test"}

    @override
    def supported_ann_types(self) -> list[str]:
        return DatasetType.CLSDIR.supported_annotation_formats

    @override
    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        copied_pairs: set[tuple[Path, str]] = (
            set()
        )  # (src, dest_abs_str) to avoid duplicate write in same pass

        for key, entry in grouped:
            file_name, group_id = cast("tuple[str, Any]", key)
            file_path = Path(str(file_name))

            split = split_of_group(prepared_ldf, group_id)

            class_names = set(whole_image_classes(entry))
            if not class_names:
                continue

            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"

            for cname in sorted(class_names):
                target_dir = (
                    self._get_data_path(self.output_path, split, self.part)
                    / cname
                )
                target_dir.mkdir(parents=True, exist_ok=True)

                dest = target_dir / new_name
                pair_key = (file_path, str(dest))

                if pair_key in copied_pairs:
                    continue
                copied_pairs.add(pair_key)

                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())

    @override
    def _dump_annotations(
        self,
        annotation_splits: dict[str, dict[str, Any]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        return

    @override
    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        split_name = self.get_split_names().get(split, split)
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split_name
