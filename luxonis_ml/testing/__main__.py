"""The ``luxonis_ml test`` commands, which open synthetic test data."""

from typing import Annotated

from cyclopts import App, Parameter
from rich import print

app = App(help="Synthetic data for testing luxonis-ml by eye.")

_MORE_VIEWS = """\
[dim]More views of the same data:
  luxonis_ml data inspect {main} --array-viz
  luxonis_ml data inspect {main} --blend-all --skeletons --keypoint-labels names
  luxonis_ml data inspect {main} --color-by instance -v val
  luxonis_ml data inspect {stereo} --array-viz --array-mode overlay
  luxonis_ml data compare {main} {predictions} --layout triple --summary[/dim]"""


@app.command
def viz(
    *,
    compare: Annotated[bool, Parameter(negative=())] = False,
    stereo: Annotated[bool, Parameter(negative=())] = False,
    augment: Annotated[bool, Parameter(negative=())] = False,
    rebuild: Annotated[bool, Parameter(negative=())] = False,
    prefix: str = "vizlab_features",
):
    """Open the feature datasets in the vizlab viewer.

    The first run paints three local datasets and stores them, which takes
    a minute or two. Later runs reuse them. The datasets hold every
    annotation kind that LDF stores and that vizlab draws; see
    `luxonis_ml.testing` for their contents.

    Without an option, the command inspects the main dataset.

    Args:
        compare: Compare the main dataset with its predictions dataset.
        stereo: Inspect the stereo dataset, with its disparity maps.
        augment: Inspect the main dataset with augmentations that move every
            kind of label.
        rebuild: Paint and store the datasets again first.
        prefix: The name of the main dataset, and the start of the names of
            the other two.

    """
    if compare + stereo + augment > 1:
        raise ValueError("Pass at most one of --compare, --stereo, --augment.")

    from luxonis_ml.data.__main__ import compare as compare_datasets
    from luxonis_ml.data.__main__ import inspect

    from .feature_datasets import FeatureDatasets, build_feature_datasets

    names = FeatureDatasets.named(prefix)
    if rebuild or not names.exist():
        print(f"Painting the feature datasets into '{names.root}'...")
        names = build_feature_datasets(prefix)
    print(
        _MORE_VIEWS.format(
            main=names.main,
            predictions=names.predictions,
            stereo=names.stereo,
        )
    )
    if compare:
        compare_datasets(names.main, names.predictions)
    elif stereo:
        inspect(names.stereo, array_viz=True, array_ignore=0.0)
    elif augment:
        inspect(names.main, aug_config=names.augmentations)
    else:
        inspect(names.main)
