"""The demo's content: one `Slide` per thing worth showing.

Hand-written, not lifted from anywhere. Each slide is a sentence or two of
explanation and a snippet short enough to read in one glance — where the
notebook's version of an example is too long for that, the example here is a
smaller one that makes the same point.

`SETUP` runs once before the first slide and is never shown: it holds the
imports and the synthetic imagery every slide draws on, so no slide has to
spend its lines on scaffolding. Every snippet is executed by
`vizlab_examples.demo.show`, and whatever its last expression evaluates to
becomes the picture beside it — so the code on a slide is always the code that
drew the picture on it.

Two rules the content follows, both learned from looking at the rendered deck:

- **Annotate something.** A box around empty gradient demonstrates a rectangle.
  Boxes here land on the car and the pedestrian that `demo.scene` draws, via
  the `CAR` and `PERSON` constants, so the annotation is always around a thing.
- **Show the claim.** If the prose says a feature does something, the picture
  has to show it doing that — a `curvature` that bows around nothing, or a
  colourblind-safe palette next to no simulation, is an assertion rather than a
  demonstration.
"""

from dataclasses import dataclass

from luxonis_ml.vizlab import Tooltip

#: Run once, shown to nobody: imports and the synthetic imagery every slide
#: draws on, so a slide's own snippet is only ever about the annotation.
SETUP = '''
import numpy as np

from luxonis_ml.ldf import DatasetRecord
from luxonis_ml.utils.color.cvd import min_separation, simulate
from luxonis_ml.vizlab import (
    DARK_THEME, LIGHT_THEME, Arrow, BBox, Caption, ClassDistribution,
    Classification, ColorBar, Corner, FlowField, FlowWheel, Heatmap,
    InfoCard, Image, Keypoints, Legend, Mask, Palette, Polyline,
    RenderOptions, Ruler, ScalarField, ScaleBar, SemanticMask, Tooltip,
    compare, grid, hstack, visualize_record, vstack, with_panel,
)
from luxonis_ml.vizlab.comparison.match import (
    CLASS_ERROR_COLOR, FN_COLOR, FP_COLOR, TP_COLOR,
)
from vizlab_examples.demo.scene import (
    CAR, CLASSES, PERSON, SIGN, label_map, traffic,
)
from vizlab_examples.gallery import gradient

#: The verdict colours `compare` paints with, so a slide can key them.
VERDICTS = [("hit", TP_COLOR), ("miss", FN_COLOR),
            ("false alarm", FP_COLOR), ("wrong class", CLASS_ERROR_COLOR)]

# The slide's picture frame, so a scene is authored at the size it is shown
# at: nothing is scaled to fit, and the deck keeps one picture footprint.
SCENE_W = min(FRAME_W, FRAME_H * 4 // 3)
SCENE_H = SCENE_W * 3 // 4
# Named scene sizes, so a snippet stays short: a wide snippet steals width
# from the picture frame it is sizing, which is a loop worth breaking.
SCENE = (SCENE_W, SCENE_H)                      # fills the frame
STRIP = (FRAME_W, FRAME_H // 2 - 52)            # one of two, stacked
THIRD = (FRAME_W // 3 - 12, FRAME_H * 2 // 3)   # one of three, across
SIXTH = (FRAME_W // 4, FRAME_H // 3 - 16)       # one of six, 3 x 2
PANE = (FRAME_W // 2 - 30, FRAME_H // 2 - 46)   # one of a 2x2
COL = (FRAME_W // 3, FRAME_H * 3 // 4)          # a column in a composite


def street(width=None, height=None):
    """The traffic frame — a car, a pedestrian, and a sign to annotate."""
    return Image(traffic(width or SCENE_W, height or SCENE_H))


def cell(hue=0.58, width=None, height=None):
    """An empty scene on a gradient backdrop, for the layout examples."""
    return Image(
        gradient(width or PANE[0], height or PANE[1], hue=hue)
    )


def dusk(hue):
    """The deck's usual dark backdrop, at pane size."""
    return gradient(*PANE, hue=hue)


def pale(hue):
    """The same backdrop lightened — a light theme wants a light frame."""
    return np.clip(dusk(hue).astype(np.int16) + 168, 0, 255).astype(np.uint8)


def plate(width, height):
    """A flat, near-black backdrop.

    A gradient is scenery, and scenery is dead weight behind a widget that
    carries its own card: the chart is the subject here, so the backdrop gets
    out of its way and the space goes to more of them.
    """
    return np.full((height, width, 3), 20, np.uint8)


def blobs(width, height, spots):
    """A scalar field: a sum of soft gaussian blobs in [0, 1]."""
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    field = np.zeros((height, width))
    for cx, cy, radius in spots:
        field += np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / radius**2))
    return np.clip(field, 0.0, 1.0)


def disparity(width, height):
    """A stereo disparity map: large up close, small far away.

    Near is the *bottom* of a forward-facing frame, so the ramp runs that way;
    an inverted depth map is the first thing a CV reader notices. The car and
    the pedestrian stand out of the ground plane at their own distances.
    """
    ground = np.linspace(0.0, 1.0, height)[:, None] ** 1.6 * 58.0
    field = np.tile(ground, (1, width))
    field[
        int(0.44 * height) : int(0.70 * height),
        int(0.52 * width) : int(0.82 * width),
    ] = 33.0
    field[
        int(0.45 * height) : int(0.76 * height),
        int(0.16 * width) : int(0.25 * width),
    ] = 47.0
    return field


def flow(width, height):
    """Optical flow: the scene streaming past, the car crossing it.

    Forward motion streams outward from the vanishing point, so every
    direction is present at a magnitude worth colouring. A field near zero
    everywhere is technically motion and renders as a white hole.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    field = np.stack(
        [(xs / width - 0.52) * 13.0, (ys / height - 0.44) * 13.0], axis=-1
    )
    field[
        int(0.42 * height) : int(0.72 * height),
        int(0.50 * width) : int(0.84 * width),
    ] = (11.0, 1.5)
    return field
'''


def _term(
    summary: str,
    *rows: tuple[str, str],
    signature: str | None = None,
    data: object = None,
) -> Tooltip:
    """One glossary entry: what a name is, how it is called, what it holds.

    ``signature`` leads, because the first question about a name in a snippet
    is how to call it; ``data`` rides in the tooltip's JSON block, so a
    constant that *is* a dict is shown as the dict rather than described.
    """
    lead = (("signature", signature),) if signature else ()
    return Tooltip(title=summary, rows=lead + rows, data=data)


#: The names a snippet leans on that are never defined on screen — they live in
#: `SETUP`, which no slide shows — and the API the prose names. Each is painted
#: as a gradient term wherever it appears and explains itself under the cursor,
#: so the code stays short without becoming a code with no key.
GLOSSARY: dict[str, Tooltip] = {
    "CAR": _term(
        "the car's box in the scene",
        ("units", "normalized to the frame"),
        signature="dict[str, float]",
        data={"x": 0.52, "y": 0.44, "w": 0.30, "h": 0.26},
    ),
    "PERSON": _term(
        "the pedestrian's box in the scene",
        ("units", "normalized to the frame"),
        signature="dict[str, float]",
        data={"x": 0.16, "y": 0.45, "w": 0.09, "h": 0.31},
    ),
    "SIGN": _term(
        "the road sign's box in the scene",
        ("units", "normalized to the frame"),
        signature="dict[str, float]",
        data={"x": 0.30, "y": 0.36, "w": 0.05, "h": 0.26},
    ),
    "CLASSES": _term(
        "the label map's class ids",
        signature="dict[int, str]",
        data={
            "0": "void",
            "1": "sky",
            "2": "building",
            "3": "road",
            "4": "car",
            "5": "person",
            "6": "sign",
        },
    ),
    "VERDICTS": _term(
        "the colours compare paints each match with",
        ("from", "vizlab.comparison.match"),
        signature="list[tuple[str, Color]]",
        data={
            "hit": "#009e73",
            "miss": "#f0e442",
            "false alarm": "#d55e00",
            "wrong class": "#cc79a7",
        },
    ),
    "SCENE": _term(
        "a scene sized to fill the picture frame",
        ("why", "authored at the size it is shown"),
        signature="tuple[int, int]  # width, height",
    ),
    "STRIP": _term(
        "one of two strips stacked down the frame",
        signature="tuple[int, int]",
    ),
    "THIRD": _term(
        "one of three cells across the frame",
        signature="tuple[int, int]",
    ),
    "PANE": _term(
        "one cell of a 2x2 grid",
        signature="tuple[int, int]",
    ),
    "COL": _term(
        "one column of a composite",
        signature="tuple[int, int]",
    ),
    "street": _term(
        "the traffic frame, ready to annotate",
        ("draws", "sky, skyline, road, car, person, sign"),
        signature="street(width=None, height=None) -> Image",
    ),
    "traffic": _term(
        "the street as raw pixels",
        ("drawn with", "Canvas.polygon / circle"),
        signature="traffic(width, height) -> np.ndarray  # (H, W, 4)",
    ),
    "cell": _term(
        "an empty gradient backdrop",
        signature="cell(hue=.58, width=None, height=None) -> Image",
    ),
    "gradient": _term(
        "a plain diagonal backdrop",
        ("from", "vizlab_examples.gallery"),
        signature="gradient(width, height, hue) -> np.ndarray",
    ),
    "label_map": _term(
        "the street as a semantic label map",
        ("keyed by", "CLASSES"),
        signature="label_map(width, height) -> np.ndarray  # (H, W) int32",
    ),
    "dusk": _term(
        "the deck's usual dark backdrop, at pane size",
        signature="dusk(hue) -> np.ndarray",
    ),
    "pale": _term(
        "the same backdrop lightened, for the light theme",
        signature="pale(hue) -> np.ndarray",
    ),
    "blobs": _term(
        "a scalar field of soft gaussian blobs",
        ("range", "0 to 1"),
        signature="blobs(width, height, spots) -> np.ndarray",
    ),
    "disparity": _term(
        "a synthetic stereo disparity map",
        ("near", "large, at the bottom of the frame"),
        signature="disparity(width, height) -> np.ndarray",
    ),
    "flow": _term(
        "synthetic optical flow for forward motion",
        ("content", "radial, plus the car crossing it"),
        signature="flow(width, height) -> np.ndarray  # (H, W, 2)",
    ),
    "visualize_record": _term(
        "render a whole LDF record in one call",
        ("reads", "boxes, keypoints, masks, arrays, tags"),
        signature="visualize_record(record, image, *, options) -> Image",
    ),
    "hover_metadata": _term(
        "turn each detection's metadata into a tooltip",
        signature="RenderOptions(hover_metadata: bool = False)",
    ),
    "BBox": _term(
        "a box, oriented or nested",
        ("units", "normalized to the frame"),
        signature="BBox(x, y, w, h, *, angle, label, score, payload)",
    ),
    "Keypoints": _term(
        "points joined into a skeleton",
        ("visibility", "COCO 0 hidden, 1 occluded, 2 visible"),
        signature="Keypoints(keypoints=[(x, y, v)], edges=[(i, j)])",
    ),
    "Mask": _term(
        "a polygon or a binary array",
        signature="Mask(points=[(x, y)] | mask=array, width, height)",
    ),
    "SemanticMask": _term(
        "a whole label map, coloured by class",
        signature="SemanticMask(labels, names, ignore_index=None)",
    ),
    "ClassDistribution": _term(
        "the whole prediction vector, not just the winner",
        signature="ClassDistribution(probabilities, ground_truth, mode)",
    ),
    "Heatmap": _term(
        "a scalar field laid over the frame",
        signature="Heatmap(values, gradient, vmin, vmax, alpha)",
    ),
    "ColorBar": _term(
        "which value a colour stands for",
        signature="ColorBar.for_heatmap(field, *, title, corner)",
    ),
    "ScalarField": _term(
        "an array read as depth or disparity",
        signature="ScalarField(values, gradient, vmin, vmax)",
    ),
    "FlowField": _term(
        "an array read as two-channel motion",
        signature="FlowField(values)  # values: (H, W, 2)",
    ),
    "FlowWheel": _term(
        "the key that decodes a flow field's colours",
        signature="FlowWheel.for_field(field, *, title, corner)",
    ),
    "Polyline": _term(
        "an open or closed run of points",
        signature="Polyline(points, values, gradient, widths, arrows)",
    ),
    "Arrow": _term(
        "relates two things in the scene",
        ("ends", "coordinates, or other annotations"),
        signature="Arrow(start, end, *, curvature, label)",
    ),
    "ScaleBar": _term(
        "a round length, for reading distances off",
        signature="ScaleBar(pixels_per_unit, unit, reference_width)",
    ),
    "Ruler": _term(
        "measures one span",
        signature="Ruler(start, end, pixels_per_unit, unit)",
    ),
    "Legend": _term(
        "the class colour key",
        signature="Legend(entries, *, title, corner)",
    ),
    "InfoCard": _term(
        "free rows in a corner",
        signature="InfoCard(rows, *, title, corner)",
    ),
    "Caption": _term(
        "one line in a corner",
        signature="Caption(text, *, corner)",
    ),
    "Classification": _term(
        "whole-frame tags",
        signature="Classification(tags=[(name, score)], corner)",
    ),
    "Tooltip": _term(
        "plain hover data an annotation carries",
        ("drawn by", "the viewer, not the renderer"),
        signature="Tooltip(title, rows, data, tint)",
    ),
    "Theme": _term(
        "the default style, palette and background",
        signature="Theme(style, palette, background)",
    ),
    "RenderOptions": _term(
        "what a render inherits when nothing overrides",
        signature="RenderOptions(theme, gradient, hover_metadata)",
    ),
    "compare": _term(
        "match predictions to truth, colour by verdict",
        ("verdicts", "hit, miss, false alarm, wrong class"),
        signature="compare(image, *, gt, pred, iou_threshold, panel)",
    ),
    "with_panel": _term(
        "attach a metadata panel beside a scene",
        signature="with_panel(image, data, *, side, title) -> Renderable",
    ),
    "grid": _term(
        "uniform cells, row major",
        signature="grid(images, *, ncols, pad, bg, titles)",
    ),
    "hstack": _term(
        "side by side, one row",
        signature="hstack(images, *, pad, bg, titles)",
    ),
    "vstack": _term(
        "stacked, one column",
        signature="vstack(images, *, pad, bg, titles)",
    ),
    "Palette": _term(
        "assigns a colour to a class name, and keeps it",
        signature="Palette(classes=None, *, generator, colors)",
    ),
    "simulate": _term(
        "how a colour looks to a colourblind viewer",
        ("models", "Machado, Oliveira & Fernandes (2009)"),
        signature="simulate(color, deficiency) -> Color",
    ),
    "min_separation": _term(
        "the closest pair in a palette, as CIEDE2000 ΔE",
        ("under", "normal and simulated colour vision"),
        signature="min_separation(colors, vision) -> float",
    ),
    "escape": _term(
        "make caller text safe to use as markup",
        signature="escape(text) -> str",
    ),
}


@dataclass(frozen=True)
class Slide:
    """One screen of the demo: what it says, and the code that draws it."""

    title: str
    body: str
    source: str


SLIDES: list[Slide] = [
    Slide(
        title="Data in, picture out",
        body=(
            "A whole LDF record renders in one call. `visualize_record` reads "
            "what the data carries — boxes, keypoints, masks, tags — and "
            "decides what to draw, so nothing here builds a vizlab object by "
            "hand. `hover_metadata` turns each detection's metadata into "
            "hover content: **try the cursor on a box**."
        ),
        source="""
record = DatasetRecord.model_validate({
    "files": {},
    "task_name": "traffic",
    "annotation": [
        {"class_name": "car",
         "boundingbox": CAR,
         "metadata": {"track": 7,
                      "speed_kph": 41.2}},
        {"class_name": "person",
         "boundingbox": PERSON,
         "metadata": {"track": 12,
                      "speed_kph": 4.6}},
        {"class_name": "sunny"},
    ],
})

visualize_record(
    record,
    traffic(*SCENE),
    options=RenderOptions(
        hover_metadata=True),
)
""",
    ),
    Slide(
        title="Boxes",
        body=(
            "`BBox` covers four cases with the *same four numbers*: plain, "
            "**oriented** through `angle`, carrying a **payload** of free text "
            "the way OCR does, and **nested**, where a detection holds its own "
            "parts."
        ),
        source="""
box = {"x": .16, "y": .2,
       "w": .58, "h": .56}

flat = BBox(**box, label="person")
turned = BBox(**box, angle=24,
              label="ship")
ocr = BBox(**box, label="word",
           payload="OPEN")
car = BBox(**box, label="car")
car.add(BBox(x=.30, y=.44,
             w=.30, h=.28,
             label="driver"))

grid([cell(.58).add(flat),
      cell(.62).add(turned),
      cell(.12).add(ocr),
      cell(.55).add(car)],
     ncols=2,
     titles=["plain", "angle=",
             "payload=", "nested"])
""",
    ),
    Slide(
        title="Points and polygons",
        body=(
            "`Keypoints` takes normalized points and the `edges` that join "
            "them into a skeleton; the third number in each point is COCO "
            "visibility, and an occluded joint is drawn hollow. `Mask` takes "
            "either a polygon or a binary array."
        ),
        source="""
w, h = STRIP

pose = cell(.68, w, h).add(Keypoints(
    keypoints=[
        (.44, .18, 2), (.56, .18, 2),
        (.36, .44, 2), (.64, .44, 1),
        (.45, .60, 2), (.55, .60, 2),
        (.42, .90, 2), (.58, .90, 1)],
    edges=[(0, 1), (0, 2), (1, 3),
           (0, 4), (1, 5), (4, 5),
           (4, 6), (5, 7)],
))
crack = cell(.44, w, h).add(Mask(
    points=[(.24, .28), (.46, .20),
            (.68, .44), (.60, .78),
            (.36, .84), (.22, .58)],
    width=w, height=h,
    label="pothole",
))

vstack([pose, crack],
       titles=["Keypoints",
               "Mask(points=)"])
""",
    ),
    Slide(
        title="Whole-frame labels",
        body=(
            "`SemanticMask` takes a label map and the names that decode it, "
            "and colours every class from the same palette the boxes use. "
            "`ignore_index` drops a class out — here the unlabelled verge, "
            "which stays the frame underneath."
        ),
        source="""
(
    street()
    .add(SemanticMask(
        labels=label_map(*SCENE),
        names=CLASSES,
        ignore_index=0))
    .add(Legend(
        entries=[n for i, n in
                 CLASSES.items() if i],
        title="classes",
        corner=Corner.TOP_RIGHT))
)
""",
    ),
    Slide(
        title="Overlays",
        body=(
            "Things drawn *over* the frame rather than at a location: a "
            "`Classification` tag stack, an `InfoCard`, a `Caption`, and a "
            "`Legend` keyed to the same palette the boxes use. Each claims a "
            "corner of its own."
        ),
        source="""
(
    street()
    .add(BBox(**CAR, label="car",
              score=.96))
    .add(BBox(**PERSON, label="person",
              score=.88))
    .add(Classification(
        tags=[("outdoor", .98),
              ("sunny", .7)],
        corner=Corner.TOP_RIGHT))
    .add(InfoCard(
        rows=["frame 421", "12:04:31"],
        title="clip_07",
        corner=Corner.TOP_LEFT))
    .add(Caption(
        text="frame_0421.jpg",
        corner=Corner.BOTTOM_LEFT))
    .add(Legend(
        entries=["car", "person"],
        title="classes",
        corner=Corner.BOTTOM_RIGHT))
)
""",
    ),
    Slide(
        title="What the cursor finds",
        body=(
            "You have been hovering these since the first slide. A `Tooltip` "
            "is plain data — a title, rows, a JSON-ish blob — that an "
            "annotation carries; the renderer records where it landed and the "
            "viewer draws it. Nothing here knows about windows."
        ),
        source="""
def seen(box, name, score, track, speed):
    return BBox(
        **box, label=name, score=score,
        tooltip=Tooltip(
            title=name,
            rows=[("track", track),
                  ("speed", f"{speed} km/h")],
            data={"frame": "0421.jpg"},
        ),
    )

(
    street()
    .add(seen(CAR, "car", .96, "7", "41.2"))
    .add(seen(PERSON, "person", .88, "12", "4.6"))
    .add(seen(SIGN, "sign", .71, "-", "0"))
    .add(InfoCard(
        rows=["track   7", "speed   41.2 km/h"],
        title="what the cursor draws",
        corner=Corner.TOP_RIGHT))
)
""",
    ),
    Slide(
        title="A prediction is a distribution",
        body=(
            "`ClassDistribution` shows the whole vector, not just the winner, "
            "and `mode` picks how to say it — all six here, from the same four "
            "numbers. Naming the `ground_truth` marks whether the model got it "
            "right; here it did not."
        ),
        source="""
scores = {"husky": .58,
          "malamute": .24,
          "wolf": .09,
          "samoyed": .09}

MODES = ("bars", "chips", "gauge",
         "stacked", "pie", "donut")

grid([
    Image(plate(*SIXTH)).add(
        ClassDistribution(
            probabilities=scores,
            ground_truth="malamute",
            mode=mode))
    for mode in MODES
], ncols=3, titles=list(MODES))
""",
    ),
    Slide(
        title="Dense fields",
        body=(
            "`Heatmap` lays a scalar field over the frame under a named "
            "gradient, and `ColorBar` says which value a colour stands for — "
            "the continuous counterpart of the class legend. Pinning `vmin` "
            "and `vmax` keeps two frames comparable."
        ),
        source="""
field = Heatmap(
    values=blobs(*SCENE, [
        (.67, .57, .13),
        (.20, .60, .08)]),
    gradient="magma",
    vmin=0.0, vmax=1.0, alpha=.62,
)

street().add(field).add(
    ColorBar.for_heatmap(
        field,
        title="confidence",
        corner=Corner.TOP_LEFT))
""",
    ),
    Slide(
        title="Arrays are pictures too",
        body=(
            "An LDF `array` label is read by what it means: `ScalarField` for "
            "a depth or disparity map, `FlowField` for two-channel motion. "
            "Each brings its own key, so a reader can tell a value from a "
            "colour."
        ),
        source="""
depth = ScalarField(
    values=disparity(*STRIP),
    gradient="viridis")
motion = FlowField(values=flow(*STRIP))

vstack([
    cell(.5, *STRIP).add(depth).add(
        ColorBar.for_heatmap(
            depth,
            title="disparity (px)",
            corner=Corner.TOP_LEFT)),
    cell(.62, *STRIP).add(motion).add(
        FlowWheel.for_field(
            motion,
            title="flow",
            corner=Corner.TOP_RIGHT)),
], titles=["ScalarField", "FlowField"])
""",
    ),
    Slide(
        title="Polyline",
        body=(
            "An open or closed run of points — a lane edge, a trajectory — "
            "the shape a mask models badly. Give it `values` and a `gradient` "
            "to colour it along its length, and `arrows` to say which way it "
            "runs."
        ),
        source="""
street().add(Polyline(
    points=[(.30, .96), (.36, .89),
            (.43, .83), (.52, .77),
            (.60, .73), (.67, .71)],
    values=[.30, .44, .58,
            .72, .86, 1.0],
    gradient="plasma",
    widths=[5.0] * 6,
    arrows=2,
    label="track 7",
))
""",
    ),
    Slide(
        title="Arrow",
        body=(
            "An `Arrow` relates two things in the scene. Its ends may be "
            "*other annotations* rather than coordinates, so they resolve to "
            "the edges of a box that moves, and `curvature` bows it clear of "
            "whatever is in between — the sign, here."
        ),
        source="""
car = BBox(**CAR, label="car")
walker = BBox(**PERSON, label="person")

(
    street()
    .add(car)
    .add(walker)
    .add(BBox(**SIGN, label="sign"))
    .add(Arrow(start=walker, end=car,
               curvature=.34,
               label="approaching"))
)
""",
    ),
    Slide(
        title="Scale",
        body=(
            "`ScaleBar` picks a round length that fits, and `Ruler` measures a "
            "span — here the gap the pedestrian has to cross. Both take a "
            "caller-supplied `pixels_per_unit`; give them `reference_width` "
            "too, so the bar stays honest at another render size."
        ),
        source="""
(
    street()
    .add(BBox(**CAR, label="car"))
    .add(BBox(**PERSON, label="person"))
    .add(ScaleBar(
        pixels_per_unit=46,
        unit="m",
        reference_width=SCENE_W))
    .add(Ruler(
        start=(.25, .74),
        end=(.55, .71),
        pixels_per_unit=46,
        unit="m",
        reference_width=SCENE_W))
)
""",
    ),
    Slide(
        title="Themes",
        body=(
            "A `Theme` bundles the default style, the palette, and the "
            "background compositing uses. `RenderOptions` carries it, and "
            "every annotation inherits from it unless it says otherwise — so "
            "one scene, two themes, changes the colours and the ground both."
        ),
        source="""
def board(theme, backdrop):
    options = RenderOptions(theme=theme)
    return grid([
        Image(backdrop(hue),
              options=options)
        .add(BBox(x=.12, y=.22,
                  w=.6, h=.56,
                  label=name, score=.93))
        for hue, name in ((.58, "car"),
                          (.08, "bus"))
    ], ncols=2, bg=theme.background)

vstack([board(DARK_THEME, dusk),
        board(LIGHT_THEME, pale)],
       titles=["dark theme",
               "light theme"])
""",
    ),
    Slide(
        title="Palettes that everyone can read",
        body=(
            "A class keeps its colour from its name. The default generator "
            "spaces hues evenly, which crowds them for a colourblind viewer "
            "— the right column simulates one. A colourblind-safe palette is "
            "picked so the closest pair stays further apart, and "
            "`min_separation` measures it rather than asserting it."
        ),
        source="""
NAMES = ["car", "person", "bus",
         "bike", "sign", "truck"]

def key(palette, vision):
    scene = cell(.5, *PANE)
    for i, name in enumerate(NAMES):
        scene.add(BBox(
            x=.03 + .16 * i, y=.14, w=.14, h=.66,
            label=name,
            color=simulate(
                palette.color_for(name), vision),
            style_overrides={"fill_alpha": 1.0}))
    return scene

def gap(palette):
    seen = [palette.color_for(n) for n in NAMES]
    worst = min_separation(seen, ("deuteranopia",))
    return f"deuteranope ΔE {worst:.0f}"

flat = Palette()
safe = Palette(generator="okabe-ito")

grid([key(flat, None), key(flat, "deuteranopia"),
      key(safe, None), key(safe, "deuteranopia")],
     ncols=2,
     titles=["default", gap(flat),
             "okabe-ito", gap(safe)])
""",
    ),
    Slide(
        title="Every string is markup",
        body=(
            "Labels, captions, titles, panel rows and tooltips all take the "
            "same Pango-style tags — `<b>`, `<i>`, `<code>`, and `<span "
            "color=… size=…>`. Pass anything you did not author through "
            "`escape` first."
        ),
        source="""
rows = [
    "<b>bold</b>   <i>italic</i>",
    "<code>mono</code>",
    "<span color='#7e7'>green</span>",
    "<span size='140%'>bigger</span>",
]

(
    street()
    .add(BBox(
        **CAR,
        label="<b>car</b> <i>94%</i>"))
    .add(InfoCard(rows=rows,
                  title="inline markup"))
)
""",
    ),
    Slide(
        title="Composition",
        body=(
            "`grid`, `hstack`, `blend` and `with_panel` return a new scene and "
            "leave their inputs alone. Nothing is flattened, so a composed "
            "scene still renders to vector SVG and still knows where its "
            "annotations are."
        ),
        source="""
left = Image(traffic(*COL)).add(
    BBox(**CAR, label="car"))
right = Image(traffic(*COL)).add(
    BBox(x=.54, y=.46, w=.29, h=.24,
         label="car", score=.9))

with_panel(
    hstack([left, right],
           titles=["ground truth",
                   "prediction"]),
    {"dataset": "traffic",
     "split": "val",
     "iou": 0.71},
    title="sample",
)
""",
    ),
    Slide(
        title="Truth against prediction",
        body=(
            "`compare` matches predictions to ground truth and colours the "
            "result by verdict — hit, miss, false alarm, wrong class — so each "
            "object carries its own answer rather than leaving you to diff two "
            "sets of boxes. All four verdicts are on this frame; `panel=True` "
            "adds the counts, as on the composition slide."
        ),
        source="""
truth = [BBox(**CAR, label="car"),
         BBox(**PERSON, label="person"),
         BBox(**SIGN, label="sign")]
predicted = [
    BBox(x=.53, y=.45, w=.29, h=.25,
         label="car", score=.94),
    BBox(**PERSON, label="bus"),
    BBox(x=.80, y=.74, w=.14, h=.18,
         label="car", score=.41),
]

graded = compare(traffic(*SCENE),
                 gt=truth,
                 pred=predicted,
                 panel=False)
graded.add(Legend(
    entries=VERDICTS,
    title="verdict",
    corner=Corner.TOP_RIGHT))
""",
    ),
    Slide(
        title="This slide is a vizlab render",
        body=(
            "Every screen in this demo — the prose, the syntax-coloured "
            "snippet, the frame around them, the street itself — was drawn "
            "with the same `Canvas`, fonts and markup the examples use, then "
            "shown through the same `Viewer` that `luxonis_ml data inspect` "
            "opens."
        ),
        source="""
rows = [
    "<code>prose, code</code> → markup",
    "<code>the street </code> → polygon",
    "<code>this card  </code> → InfoCard",
    "<code>the window </code> → Viewer",
]

(
    street()
    .add(InfoCard(
        rows=rows,
        title="what drew this demo",
        corner=Corner.TOP_LEFT))
    .add(Caption(
        text="luxonis_ml data inspect",
        corner=Corner.BOTTOM_RIGHT))
)
""",
    ),
]
