# vizlab examples

Runnable examples for `luxonis_ml.vizlab`, the visualization layer of
**luxonis-ml**.

## Start here

**[`vizlab.ipynb`](vizlab.ipynb)** is the tour, and the only thing you need to
read. It renders every annotation type beside the LDF data that produces it,
then covers themes, palettes, inline markup, composition, prediction
comparison, the interactive HTML export, video, and the in-notebook viewer.

Every frame in it is painted with numpy, so it needs no dataset download and
runs offline.

```bash
pip install "luxonis-ml[viz]"       # rendering
pip install "luxonis-ml[data]"      # + the LuxonisDataset round-trip (§1.9)
pip install "luxonis-ml[notebook]"  # + the interactive viewer (§8)

jupyter lab vizlab_examples/vizlab.ipynb
```

The notebook ships with its outputs, so it reads as a gallery without running
it — except for the two live sections (the embedded HTML page and the viewer
widget), which need a running kernel.

## The rest

| Path                                         | What it is                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| [`vizlab.ipynb`](vizlab.ipynb)               | **The tour.** Every feature, with output.                                              |
| [`gallery.py`](gallery.py)                   | Builds the figures under `output/` that the API docs embed. Run it to regenerate them. |
| [`showcase_scene.py`](showcase_scene.py)     | One dense scene in a desktop window — a smoke test to eyeball while developing.        |
| [`camera_3d/`](camera_3d/)                   | Projecting 3-D geometry, depth, and point clouds into a camera.                        |
| [`architecture_mocks/`](architecture_mocks/) | Diagrams of vizlab's own internals, for the design docs.                               |
| `output/`                                    | Generated figures. Written by `gallery.py`; not edited by hand.                        |

```bash
python vizlab_examples/gallery.py          # rewrite output/*.png
python vizlab_examples/showcase_scene.py   # interactive window
python -m vizlab_examples.camera_3d        # writes camera_3d/output/
```

`gallery.py`'s augmentation figures additionally need the `data` extra and the
`D2_ParkingLot_Native` dataset (downloaded from the GCS test bucket, so they
need credentials); they skip themselves with a hint when either is missing.

## The CLI

Everything the notebook draws is also reachable from a shell, over a real
dataset:

```bash
luxonis_ml data inspect <dataset>
luxonis_ml data inspect <dataset> --color-by instance --skeletons --array-viz
luxonis_ml data inspect <dataset> --class-name person --min-instances 3 --save contact.png
```

See `luxonis_ml data inspect --help` for the filters, array-reading options, and
the `--save` targets (PNG, HTML, or a video clip).
