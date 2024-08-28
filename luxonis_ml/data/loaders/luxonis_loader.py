import json
import logging
from typing import Optional

import numpy as np

from ..augmentations import Augmentations
from ..datasets import LuxonisDataset
from ..utils.enums import LabelType
from .base_loader import BaseLoader, LuxonisLoaderOutput


class LuxonisLoader(BaseLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str = "train",
        stream: bool = False,
        augmentations: Optional[Augmentations] = None,
        *,
        force_resync: bool = False,
    ) -> None:
        """A loader class used for loading data from L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: LuxonisDataset to use
        @type view: str
        @param view: View of the dataset. Defaults to "train".
        @type stream: bool
        @param stream: Flag for data streaming. Defaults to C{False}.
        @type augmentations: Optional[luxonis_ml.loader.Augmentations]
        @param augmentations: Augmentation class that performs augmentations. Defaults
            to C{None}.
        @type force_resync: bool
        @param force_resync: Flag to force resync from cloud. Defaults to C{False}.
        """

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.stream = stream
        self.sync_mode = self.dataset.is_remote and not self.stream

        if self.sync_mode:
            self.dataset.sync_from_cloud(force=force_resync)

        self.view = view

        df = self.dataset._load_df_offline()
        if df is None:
            raise FileNotFoundError("Cannot find dataframe")
        self.df = df

        if not self.dataset.is_remote or not self.stream:
            file_index = self.dataset._get_file_index()
            if file_index is None:
                raise FileNotFoundError("Cannot find file index")
            self.df = self.df.join(file_index, on="uuid").drop("file_right")
        else:
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        self.classes, self.classes_by_task = self.dataset.get_classes()
        self.augmentations = augmentations
        if self.view in ["train", "val", "test"]:
            splits_path = self.dataset.metadata_path / "splits.json"
            if not splits_path.exists():
                raise RuntimeError(
                    "Cannot find splits! Ensure you call dataset.make_splits()"
                )
            with open(splits_path, "r") as file:
                splits = json.load(file)
            self.instances = splits[self.view]
        else:
            raise NotImplementedError

        self.idx_to_df_row = []
        for uuid in self.instances:
            boolean_mask = df["uuid"] == uuid
            row_indexes = boolean_mask.arg_true().to_list()
            self.idx_to_df_row.append(row_indexes)

        self.class_mappings = {}
        for task in df["task"].unique():
            class_mapping = {
                class_: i
                for i, class_ in enumerate(
                    sorted(
                        self.classes_by_task[task],
                        key=lambda x: {"background": -1}.get(x, 0),
                    )
                )
            }
            self.class_mappings[task] = class_mapping
        self.img = np.random.rand(416, 416, 3).astype(np.float32)
        self.labels = {}
        self.labels["boundingbox"] = (
            np.array(
                [
                    [
                        3.00000000e00,
                        2.98437500e-03,
                        7.64000000e-02,
                        9.52796875e-01,
                        9.10117647e-01,
                    ],
                    [
                        3.00000000e00,
                        3.35765625e-01,
                        3.41576471e-01,
                        2.85015625e-01,
                        4.71905882e-01,
                    ],
                    [
                        3.00000000e00,
                        7.44640625e-01,
                        2.53929412e-01,
                        2.55359375e-01,
                        5.64047059e-01,
                    ],
                    [
                        5.60000000e01,
                        6.90875000e-01,
                        6.81858824e-01,
                        3.36718750e-02,
                        3.91529412e-02,
                    ],
                    [
                        5.60000000e01,
                        8.44843750e-01,
                        7.95505882e-01,
                        9.00937500e-02,
                        1.28823529e-01,
                    ],
                    [
                        5.50000000e01,
                        5.34234375e-01,
                        2.89200000e-01,
                        4.16343750e-01,
                        4.83152941e-01,
                    ],
                    [
                        5.60000000e01,
                        3.11000000e-01,
                        2.85270588e-01,
                        8.31093750e-02,
                        3.95058824e-02,
                    ],
                    [
                        5.60000000e01,
                        7.76765625e-01,
                        7.26800000e-01,
                        4.75468750e-02,
                        4.99764706e-02,
                    ],
                ]
            ),
            LabelType.BOUNDINGBOX,
        )

    def __len__(self) -> int:
        """Returns length of the dataset.

        @rtype: int
        @return: Length of dataset.
        """
        return len(self.instances)

    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Function to load a sample consisting of an image and its annotations.

        @type idx: int
        @param idx: The (often random) integer index to retrieve a sample from the
            dataset.
        @rtype: LuxonisLoaderOutput
        @return: The loader ouput consisting of the image and a dictionary defining its
            annotations.
        """

        return self.img, self.labels
