"""Slide definitions and shared setup for the vizlab demo.

Each snippet runs in the same namespace. Its final expression becomes the
illustration beside it.
"""

from dataclasses import dataclass

from luxonis_ml.vizlab import Tooltip

# Shared imports, fixtures, and dimensions keep the visible snippets focused.
SETUP = '''
import numpy as np

from luxonis_ml.ldf import DatasetRecord
from luxonis_ml.utils.color.cvd import min_separation, simulate
from luxonis_ml.vizlab import (
    DARK_THEME, LIGHT_THEME, Arrow, BBox, Caption, ClassDistribution,
    Classification, ColorBar, Corner, FlowField, FlowWheel, Heatmap,
    InfoCard, Image, Keypoints, Legend, Mask, Palette, Polyline,
    RenderOptions, Ruler, ScalarField, ScaleBar, SemanticMask, Tooltip,
    compare, escape, grid, hstack, visualize_record, vstack, with_panel,
)
from luxonis_ml.vizlab.comparison.match import (
    CLASS_ERROR_COLOR, FN_COLOR, FP_COLOR, TP_COLOR,
)
from vizlab_examples.demo.scene import (
    CAR, CLASSES, PERSON, POSE, POSE_EDGES, SIGN, label_map, traffic,
)
from vizlab_examples.gallery import gradient

# The verdict colours used by the comparison slide's legend.
VERDICTS = [("hit", TP_COLOR), ("miss", FN_COLOR),
            ("false alarm", FP_COLOR), ("wrong class", CLASS_ERROR_COLOR)]

# Author scenes at their display size.
SCENE_W = min(FRAME_W, FRAME_H * 4 // 3)
SCENE_H = SCENE_W * 3 // 4
# Common layouts used by the examples.
SCENE = (SCENE_W, SCENE_H)                      # fills the frame
STRIP = (FRAME_W, FRAME_H // 2 - 52)            # one of two, stacked
SIXTH = (FRAME_W // 4, FRAME_H // 3 - 16)       # one of six, 3 x 2
PANE = (FRAME_W // 2 - 30, FRAME_H // 2 - 46)   # one of a 2x2
COL = (FRAME_W // 3, FRAME_H * 3 // 4)          # a column in a composite


def street(width=None, height=None):
    return Image(traffic(width or SCENE_W, height or SCENE_H))


def cell(hue=0.58, width=None, height=None):
    return Image(
        gradient(width or PANE[0], height or PANE[1], hue=hue)
    )


def dusk(hue):
    return gradient(*PANE, hue=hue)


def pale(hue):
    return np.clip(dusk(hue).astype(np.int16) + 168, 0, 255).astype(np.uint8)


def plate(width, height):
    return np.full((height, width, 3), 20, np.uint8)


def blobs(width, height, spots):
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    field = np.zeros((height, width))
    for cx, cy, radius in spots:
        field += np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / radius**2))
    return np.clip(field, 0.0, 1.0)


def disparity(width, height):
    """Build a synthetic disparity map with nearer objects at larger values."""
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
    """Build radial forward flow with the car moving across it."""
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
    lead = (("signature", signature),) if signature else ()
    return Tooltip(title=summary, rows=lead + rows, data=data)


# API and setup names shown as hoverable terms in prose and code.
GLOSSARY: dict[str, Tooltip] = {
    "highlighted text": Tooltip(
        title="Glossary terms provide more information"
    ),
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
            "hit": "#35d6a6",
            "miss": "#ff6b6b",
            "false alarm": "#ffc24b",
            "wrong class": "#ff9142",
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
    title: str
    body: str
    source: str


SLIDES: list[Slide] = [
    Slide(
        title="Render an LDF record",
        body=(
            "`visualize_record` reads boxes, keypoints, masks, and tags from an "
            "LDF record. `hover_metadata` exposes detection metadata. Hover "
            "over a box or the `highlighted text` to inspect them."
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
        title="Bounding boxes",
        body=(
            "`BBox` handles plain and oriented boxes, OCR payloads, and nested "
            "parts. All four examples use the same normalized coordinates."
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
        title="Keypoints and masks",
        body=(
            "`Keypoints` joins normalized points with `edges`; the third value "
            "is COCO visibility, with occluded joints drawn hollow. `Mask` "
            "accepts polygons or binary arrays."
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
        title="Missing keypoints",
        body=(
            "`(0, 0, 0)` represents an unlabelled or unpredicted joint. Vizlab "
            "infers its position from the skeleton, marks it with a cross, and "
            "uses dashed edges to retain the limb's structure."
        ),
        source="""
missing = {3, 6}  # an elbow, a wrist
gaps = [(0., 0., 0) if i in missing
        else p
        for i, p in enumerate(POSE)]

def figure(points):
    return cell(.68, *COL).add(
        Keypoints(keypoints=points,
                  edges=POSE_EDGES))

hstack([figure(POSE), figure(gaps)],
       titles=["labelled",
               "predicted"])
""",
    ),
    Slide(
        title="Semantic masks",
        body=(
            "`SemanticMask` colours a label map using its class names. "
            "`ignore_index=0` leaves the unlabelled verge transparent over the "
            "source frame."
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
        title="Frame overlays",
        body=(
            "`Classification`, `InfoCard`, `Caption`, and `Legend` add "
            "frame-level information in separate corners."
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
        title="Tooltips",
        body=(
            "`Tooltip` stores a title, rows, and structured data on an "
            "annotation. The renderer records its hit region and the viewer "
            "draws the tooltip."
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
        title="Class distributions",
        body=(
            "`ClassDistribution` renders the full probability vector in six "
            'modes. `ground_truth="malamute"` marks this incorrect prediction.'
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
        title="Heatmaps",
        body=(
            "`Heatmap` overlays a scalar field; `ColorBar` maps colours back "
            "to values. Fixed `vmin` and `vmax` make frames comparable."
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
        title="Array fields",
        body=(
            "`ScalarField` displays depth or disparity arrays, while "
            "`FlowField` displays two-channel motion. Each example includes "
            "the corresponding colour key."
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
        title="Polylines",
        body=(
            "`Polyline` draws open or closed paths such as lane edges and "
            "trajectories. `values` and `gradient` colour the path; `arrows` "
            "show direction."
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
        title="Arrows",
        body=(
            "`Arrow` can connect coordinates or annotations. Here its endpoints "
            "follow the pedestrian and car boxes, while `curvature` routes it "
            "around the sign."
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
        title="Measurement",
        body=(
            "`ScaleBar` chooses a round display length and `Ruler` measures a "
            "span. `pixels_per_unit` defines the scale; `reference_width` "
            "preserves it across render sizes."
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
            "`RenderOptions` applies a `Theme` to every annotation unless "
            "locally overridden. The same scene uses different palettes and "
            "backgrounds under the dark and light themes."
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
        title="Colourblind-safe palettes",
        body=(
            "`Palette` assigns stable colours by class name. The right column "
            "simulates deuteranopia; `min_separation` reports the closest pair "
            "for the default and Okabe-Ito palettes."
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
        title="Markup",
        body=(
            "Labels, captions, titles, panel rows, and tooltips accept "
            "Pango-style markup. Use `escape` for text that is not authored "
            "markup."
        ),
        source="""
rows = [
    "<b>bold</b>   <i>italic</i>",
    "<code>mono</code>",
    "<span color='#7e7'>green</span>",
    "<span size='140%'>bigger</span>",
    escape("<untrusted>"),
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
            "`hstack` combines the two scenes and `with_panel` adds metadata. "
            "The result retains its annotations and hover regions."
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
        title="Compare predictions",
        body=(
            "`compare` matches predictions with ground truth and colours hits, "
            "misses, false alarms, and class errors. This frame includes all "
            "four verdicts; `panel=True` can add summary counts."
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
        title="Built with vizlab",
        body=(
            "The prose, highlighted code, page frame, and illustration are all "
            "drawn with vizlab. The deck uses the same `Viewer` as "
            "`luxonis_ml data inspect`."
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
