"""Synthetic data for testing luxonis-ml by eye.

The package paints street scenes and stores them as LDF datasets. The
datasets hold every annotation kind that LDF stores and that vizlab draws,
so a change to the loader or to the renderer shows up in one viewer session.

Usage
=====

Open the datasets from the command line. The first run paints and stores
them, which takes a minute or two. Later runs reuse them:

.. code-block:: bash

    luxonis_ml test viz              # the main dataset
    luxonis_ml test viz --compare    # ground truth against predictions
    luxonis_ml test viz --stereo     # left and right images, disparity
    luxonis_ml test viz --augment    # the main dataset, augmented
    luxonis_ml test viz --rebuild    # paint the datasets again first

The datasets are ordinary local datasets, so every option of ``data inspect``
and ``data compare`` works on them:

.. code-block:: bash

    luxonis_ml data inspect vizlab_features --array-viz
    luxonis_ml data inspect vizlab_features --blend-all --skeletons
    luxonis_ml data inspect vizlab_features --color-by instance -v val

Build them from Python with `build_feature_datasets`:

.. code-block:: python

    from luxonis_ml.testing import build_feature_datasets

    names = build_feature_datasets("my_features", frames=4)
    print(names.main, names.predictions, names.stereo)

Contents
========

The main dataset starts with focused cases, one sample each, all in the
``train`` split:

=====================  ======================================================
Case                   What it checks
=====================  ======================================================
``everything``         Every class, every annotation kind and every array.
``true_negative``      A sample without any annotation.
``declared_negative``  Empty detection tasks next to semantic and scene labels.
``crowd``              A dense scene, for the decluttering of ``d``.
``far_away``           Tiny objects near the horizon.
``image_border``       Objects and joints cut by the image border.
``occlusion``          Masks that a nearer object cuts, occluded joints.
``lone_object``        One object, whose metadata the panel shows.
``facing_away``        People without a face sub-detection, so the
                       ``people/face`` task stays apart from ``people``.
``markup_text``        Markup characters, CJK and right-to-left text in the
                       metadata.
``night_rain``         Night and rain.
``dusk_fog``           Dusk and fog.
``close_pose``         One large pose, for ``--keypoint-labels``.
=====================  ======================================================

Four sequences of moving traffic follow. Day and night are in ``train``,
dusk with rain is in ``val``, and day with fog is in ``test``. The flow of
one frame points to the next frame of its sequence.

The tasks of the main dataset:

=================  ==========================================================
Task               Labels
=================  ==========================================================
``vehicles``       Boxes and instance masks of cars, trucks and buses, with
                   categorical, float and string metadata. A ``plate``
                   sub-detection with the plate text as metadata.
``people``         Boxes, instance masks and the 17 COCO keypoints with
                   edges, flip pairs and sigmas. Visibility 1 marks a joint
                   behind a nearer object, 0 a joint outside the image. A
                   ``face`` sub-detection, without a class, with five
                   landmarks whose flip pairs LDF infers from their names.
``signs``          Boxes and instance masks given as polygons, with Unicode
                   text in the metadata.
``semantic``       Semantic masks of nine classes. The curbs, the poles and
                   a traffic cone stay unlabeled, which is the background.
``scene``          Two image-level classes, the light and the weather, and a
                   detection with metadata only.
array tasks        See `feature_datasets`.
=================  ==========================================================

Each sample also carries nested sample metadata for the side panel.

See:
    `feature_datasets` for the predictions, the stereo dataset and the
    arrays, and `street` for the painter.
"""

from .feature_datasets import (
    FeatureDatasets,
    augmentation_config,
    build_feature_datasets,
)

__all__ = ["FeatureDatasets", "augmentation_config", "build_feature_datasets"]
