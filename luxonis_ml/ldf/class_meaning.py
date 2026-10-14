r"""What a class name says about the side or the direction of a thing.

Some class names name a side or a direction: ``left_hand``, ``turn_left``,
``arrow_up``. A flip or a turn of the image changes what the picture
shows, so such a name can become wrong. A record says what a class name
means, and `AlbumentationsEngine` gives each label the class that the
meaning asks for, after the whole augmentation pipeline. The augmentation
configuration stays plain: ``HorizontalFlip``, ``RandomRotate90``, ``D4``.

The turns and mirrors of an image are the 8 symmetries of a square, see
`SquareSymmetry`. A class says what each of them makes of it, in one of
four forms. In a record, the class becomes a mapping:

    - ``mirror`` names the class on the other side of the object. Every
      mirror of the image swaps the two, and a rotation keeps them:

      .. python::

          "class": {
              "name": "left_hand",
              "mirror": "right_hand",
          }

    - ``upright`` says that only the image as it is can show the class: a
      digit 6 turned by 180° is a 9, and mirrored text is no text. The
      engine turns or flips such a sample back. With ``mirror``, an upright
      class can still flip left and right, into its partner:

      .. python::

          "class": {
              "name": "keep_left",
              "mirror": "keep_right",
              "upright": True,
          }

    - ``direction`` names a set of classes that point in different
      directions. Every turn and mirror moves each class to the class that
      points where it points now:

      .. python::

          "class": {
              "name": "arrow_up",
              "direction": {
                  "up": "arrow_up",
                  "right": "arrow_right",
                  "down": "arrow_down",
                  "left": "arrow_left",
              },
          }

    - ``symmetry`` gives the class of each symmetry. A symmetry that the
      mapping leaves out makes no valid sample, so the engine undoes it:

      .. python::

          "class": {
              "name": "b",
              "symmetry": {
                  "hflip": "d",
                  "vflip": "p",
                  "rotate_180": "q",
              },
          }

A class without a meaning never changes. Each form declares the classes it
names too: ``right_hand`` becomes a class of the task when ``left_hand``
names it, also before any record has it. One declaration is enough for the
whole task, and a record without the mapping keeps a plain class string.

The engine reads the symmetry of the whole image: three probe points move
with the image through every spatial transform. When a class of the sample
cannot show that symmetry, the engine adds one more flip or turn that
brings the sample to a symmetry that every class can show. It then gives
each box, each image-level class and each segmentation channel its new
class.

Limits:

    - A meaning covers the turns and mirrors of the image, not colour,
      scale or cropping (``red_light``, ``truncated``).
    - ``mirror`` means a side of the object, such as a hand. A name relative
      to the camera, such as a lane left of the car, keeps its side under a
      vertical flip, and no form describes it yet.
    - Each sample takes one symmetry for all of its labels. A transform that
      cuts the image into tiles and shuffles them, such as
      ``RandomGridShuffle``, has no such symmetry, and the engine then
      keeps every class.

Example:
    >>> metadata = ClassMetadata(mirror_pairs=[("left_hand", "right_hand")])
    >>> metadata.table()["left_hand"]["hflip"]
    'right_hand'
    >>> metadata.table()["left_hand"]["rotate_90"]
    'left_hand'

"""

from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypeAlias, get_args

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from luxonis_ml.typing import BaseModelExtraForbid

SquareSymmetry: TypeAlias = Literal[
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "hflip",
    "vflip",
    "transpose",
    "anti_transpose",
]
r"""A turn or a mirror of the image.

    - ``"identity"``: The image as it is.
    - ``"rotate_90"``, ``"rotate_180"``, ``"rotate_270"``: A turn by that
      many degrees, counterclockwise on the screen.
    - ``"hflip"``: A mirror that swaps left and right.
    - ``"vflip"``: A mirror that swaps top and bottom.
    - ``"transpose"``: A mirror across the diagonal from the top-left
      corner to the bottom-right corner.
    - ``"anti_transpose"``: A mirror across the other diagonal.
"""

Direction: TypeAlias = Literal[
    "up",
    "up_right",
    "right",
    "down_right",
    "down",
    "down_left",
    "left",
    "up_left",
]
"""A direction in the image, as seen on the screen."""

#: Each symmetry as a matrix on ``(x, y)``, with ``y`` pointing down.
SYMMETRIES: Final[dict[SquareSymmetry, np.ndarray]] = {
    "identity": np.array([[1, 0], [0, 1]]),
    "rotate_90": np.array([[0, 1], [-1, 0]]),
    "rotate_180": np.array([[-1, 0], [0, -1]]),
    "rotate_270": np.array([[0, -1], [1, 0]]),
    "hflip": np.array([[-1, 0], [0, 1]]),
    "vflip": np.array([[1, 0], [0, -1]]),
    "transpose": np.array([[0, 1], [1, 0]]),
    "anti_transpose": np.array([[0, -1], [-1, 0]]),
}

_DIRECTIONS: Final[dict[Direction, tuple[int, int]]] = {
    "up": (0, -1),
    "up_right": (1, -1),
    "right": (1, 0),
    "down_right": (1, 1),
    "down": (0, 1),
    "down_left": (-1, 1),
    "left": (-1, 0),
    "up_left": (-1, -1),
}
_CARDINAL: Final = frozenset({"up", "right", "down", "left"})
_DIAGONAL: Final = frozenset(
    {"up_right", "down_right", "down_left", "up_left"}
)
_ROTATIONS: Final = frozenset(
    {"identity", "rotate_90", "rotate_180", "rotate_270"}
)

#: A measured map whose two axes differ in length by more than this factor
#: is no turn or mirror of the image, for example after a tile shuffle.
_MAX_STRETCH: Final = 3.0


def _symmetry_of(matrix: np.ndarray) -> SquareSymmetry:
    for name, candidate in SYMMETRIES.items():
        if np.array_equal(candidate, matrix):
            return name
    raise ValueError(f"No symmetry of a square has the matrix {matrix}.")


def compose(outer: SquareSymmetry, inner: SquareSymmetry) -> SquareSymmetry:
    """Return the symmetry that applies ``inner`` and then ``outer``.

    Example:
        >>> compose("hflip", "hflip")
        'identity'
        >>> compose("vflip", "hflip")
        'rotate_180'

    """
    return _symmetry_of(SYMMETRIES[outer] @ SYMMETRIES[inner])


def nearest_symmetry(matrix: np.ndarray) -> SquareSymmetry | None:
    r"""Round a measured :math:`2 \times 2` map to the nearest symmetry.

    The map loses its scale and shear first, so a resize or a small
    rotation keeps the symmetry.

    Returns:
        The nearest symmetry, or ``None`` when the map is no turn or mirror,
        such as a map that collapses or stretches one axis.

    Example:
        >>> nearest_symmetry(np.array([[-2.0, 0.1], [0.0, 1.9]]))
        'hflip'
        >>> nearest_symmetry(np.zeros((2, 2))) is None
        True

    """
    if not np.all(np.isfinite(matrix)):
        return None
    u, scales, vt = np.linalg.svd(matrix)
    if scales[-1] <= 1e-9 or scales[0] / scales[-1] > _MAX_STRETCH:
        return None
    orthogonal = u @ vt
    return max(
        SYMMETRIES,
        key=lambda name: float(np.trace(SYMMETRIES[name].T @ orthogonal)),
    )


def turn(direction: Direction, symmetry: SquareSymmetry) -> Direction:
    """Return where ``direction`` points after ``symmetry``.

    Example:
        >>> turn("up", "rotate_90")
        'left'
        >>> turn("up_left", "hflip")
        'up_right'

    """
    moved = SYMMETRIES[symmetry] @ np.array(_DIRECTIONS[direction])
    return next(
        name
        for name, vector in _DIRECTIONS.items()
        if vector == (int(moved[0]), int(moved[1]))
    )


class ClassMeaning(BaseModelExtraForbid):
    """What one class name says about its side or its direction.

    A record gives it as the mapping of ``class``, beside ``name``.
    `LuxonisDataset.set_class_metadata` takes the same mapping without
    ``name``. See the module for the four forms.

    Example:
        >>> ClassMeaning(mirror="right_hand").mirror
        'right_hand'

    Attributes:
        mirror: The class on the other side of the object.
        upright: Whether only the image as it is can show the class. With
            `mirror`, the class can also flip left and right.
        direction: The classes of a direction set, keyed by the direction
            each one points in. It holds the class itself too.
        symmetry: The class of each symmetry. A symmetry that it leaves
            out, or that maps to ``None``, makes no valid sample.

    """

    mirror: str | None = None
    upright: bool = False
    direction: dict[Direction, str] | None = None
    symmetry: dict[SquareSymmetry, str | None] | None = None

    @model_validator(mode="after")
    def _check_form(self) -> Self:
        """Check that the meaning uses one form."""
        forms = [
            field
            for field in ("mirror", "direction", "symmetry")
            if getattr(self, field) is not None
        ]
        if len(forms) > 1:
            raise ValueError(
                f"A class meaning takes one of `mirror`, `direction` and "
                f"`symmetry`, not {' and '.join(forms)}."
            )
        if self.upright and forms and forms != ["mirror"]:
            raise ValueError(
                "`upright` goes alone or with `mirror`. A direction set or "
                "a symmetry table says which turns are valid already."
            )
        if not forms and not self.upright:
            raise ValueError(
                "A class meaning needs `mirror`, `upright`, `direction` or "
                "`symmetry`."
            )
        if self.direction is not None:
            _check_direction_set(self.direction)
        return self

    def check_name(self, name: str) -> None:
        """Check that the meaning fits the class that declares it.

        Args:
            name: The name of the class.

        Raises:
            ValueError: If the class is its own mirror partner, or if its
                direction set does not hold it.

        """
        if self.mirror == name:
            raise ValueError(
                f"Class '{name}' names itself as its mirror partner. A class "
                "that a mirror keeps needs no meaning."
            )
        if self.direction is not None and name not in self.direction.values():
            raise ValueError(
                f"The direction set of class '{name}' does not hold it: "
                f"{self.direction}."
            )


def _check_direction_set(directions: Mapping[Direction, str]) -> None:
    """Check that a direction set is closed under every turn and mirror."""
    names = list(directions.values())
    if len(set(names)) != len(names):
        raise ValueError(
            f"A direction set names one class twice: {dict(directions)}."
        )
    keys = set(directions)
    for group in (_CARDINAL, _DIAGONAL):
        if keys & group and not group <= keys:
            missing = sorted(group - keys)
            raise ValueError(
                "A turn moves each direction to another one of its group, "
                "so a direction set holds all four cardinal directions or "
                f"none, and all four diagonals or none. Missing: "
                f"{', '.join(missing)}."
            )


class ClassMetadata(BaseModelExtraForbid):
    """The class meanings of one task, as the dataset stores them.

    `LuxonisDataset.add` collects them from the records, and
    `LuxonisDataset.set_class_metadata` changes them. Each relation is
    stored once, so it cannot become one-sided. A class is in at most one
    relation.

    Example:
        >>> metadata = ClassMetadata().with_meanings(
        ...     {
        ...         "digit_6": ClassMeaning(upright=True),
        ...     }
        ... )
        >>> metadata.table()["digit_6"]["rotate_180"] is None
        True

    Attributes:
        mirror_pairs: Pairs of classes on the two sides of an object.
        upright: Classes that only the image as it is can show. A member
            of a mirror pair can also flip left and right.
        direction_sets: Sets of classes keyed by the direction each one
            points in.
        symmetry: Explicit symmetry tables, keyed by one class of each
            table.

    """

    mirror_pairs: list[tuple[str, str]] = []
    upright: list[str] = []
    direction_sets: list[dict[Direction, str]] = []
    symmetry: dict[str, dict[SquareSymmetry, str | None]] = {}

    @model_validator(mode="after")
    def _check_relations(self) -> Self:
        """Check that the relations are disjoint and consistent."""
        seen: dict[str, str] = {}

        def claim(names: Iterable[str], relation: str) -> None:
            for name in names:
                if name in seen and seen[name] != relation:
                    raise ValueError(
                        f"Class '{name}' is in two class meanings: "
                        f"{seen[name]} and {relation}."
                    )
                seen[name] = relation

        for a, b in self.mirror_pairs:
            if a == b:
                raise ValueError(
                    f"Class '{a}' names itself as its mirror partner."
                )
            claim((a, b), f"the mirror pair ({a}, {b})")
        upright = set(self.upright)
        for a, b in self.mirror_pairs:
            if (a in upright) != (b in upright):
                upright |= {a, b}
        self.upright = sorted(upright)
        for directions in self.direction_sets:
            _check_direction_set(directions)
            claim(directions.values(), f"the direction set {directions}")
        for name, table in self.symmetry.items():
            claim(
                {name, *(t for t in table.values() if t is not None)},
                f"the symmetry table of '{name}'",
            )
        paired = {name for pair in self.mirror_pairs for name in pair}
        if overlap := sorted(upright & (set(seen) - paired)):
            raise ValueError(
                f"Classes {', '.join(overlap)} are upright and also in a "
                "direction set or a symmetry table."
            )
        self.table()
        return self

    def classes(self) -> set[str]:
        """Return every class that the meanings name."""
        return set(self.table())

    def table(self) -> dict[str, dict[SquareSymmetry, str | None]]:
        """Return the class of each class under each symmetry.

        ``None`` means that the symmetry makes no valid sample of the
        class.

        Raises:
            ValueError: If a symmetry table contradicts itself.

        """
        result: dict[str, dict[SquareSymmetry, str | None]] = {}
        for representative, row in self._representatives():
            for name, derived in _orbit(representative, row).items():
                if name in result and result[name] != derived:
                    raise ValueError(
                        f"The symmetry table of '{representative}' gives "
                        f"class '{name}' two different tables."
                    )
                result[name] = derived
        return result

    def meanings(self) -> dict[str, ClassMeaning]:
        """Return one meaning for each relation, keyed by one of its classes.

        `LuxonisDataset.set_class_metadata` takes this mapping, so it
        copies the meanings of one dataset into another.
        """
        upright = set(self.upright)
        meanings = {
            a: ClassMeaning(mirror=b, upright=a in upright)
            for a, b in self.mirror_pairs
        }
        paired = {name for pair in self.mirror_pairs for name in pair}
        meanings |= {
            name: ClassMeaning(upright=True)
            for name in self.upright
            if name not in paired
        }
        meanings |= {
            next(iter(directions.values())): ClassMeaning(
                direction=dict(directions)
            )
            for directions in self.direction_sets
        }
        meanings |= {
            name: ClassMeaning(symmetry=dict(table))
            for name, table in self.symmetry.items()
        }
        return meanings

    def with_meanings(
        self, meanings: Mapping[str, ClassMeaning | None]
    ) -> "ClassMetadata":
        """Return the metadata with new meanings for some classes.

        A new meaning replaces each relation that holds a class it names.
        ``None`` removes the relation of that class. A removed pair thus
        removes the meaning of the partner too.

        Args:
            meanings: The new meaning of each class, or ``None``.

        Returns:
            The new metadata.

        Raises:
            ValueError: If a meaning does not fit its class, or if the
                result is not consistent.

        """
        touched: set[str] = set()
        for name, meaning in meanings.items():
            touched.add(name)
            if meaning is not None:
                meaning.check_name(name)
                touched |= _named_classes(name, meaning)

        def keep(names: Iterable[str]) -> bool:
            return not touched & set(names)

        pairs = [pair for pair in self.mirror_pairs if keep(pair)]
        upright = [name for name in self.upright if keep([name])]
        direction_sets = [
            dict(directions)
            for directions in self.direction_sets
            if keep(directions.values())
        ]
        symmetry = {
            name: dict(table)
            for name, table in self.symmetry.items()
            if keep({name, *(t for t in table.values() if t is not None)})
        }
        for name, meaning in meanings.items():
            if meaning is None:
                continue
            if meaning.mirror is not None:
                pairs.append((name, meaning.mirror))
            if meaning.upright:
                upright.append(name)
            if meaning.direction is not None:
                direction_sets.append(dict(meaning.direction))
            if meaning.symmetry is not None:
                symmetry[name] = dict(meaning.symmetry)
        return ClassMetadata(
            mirror_pairs=pairs,
            upright=upright,
            direction_sets=direction_sets,
            symmetry=symmetry,
        )

    def _representatives(
        self,
    ) -> list[tuple[str, dict[SquareSymmetry, str | None]]]:
        """Return one class of each relation, with its full table."""
        names: tuple[SquareSymmetry, ...] = get_args(SquareSymmetry)
        upright = set(self.upright)
        rows: list[tuple[str, dict[SquareSymmetry, str | None]]] = []
        for a, b in self.mirror_pairs:
            if a in upright:
                rows.append((a, _only(names, {"identity": a, "hflip": b})))
            else:
                rows.append(
                    (a, {g: a if g in _ROTATIONS else b for g in names})
                )
        paired = {name for pair in self.mirror_pairs for name in pair}
        rows += [
            (name, _only(names, {"identity": name}))
            for name in self.upright
            if name not in paired
        ]
        for directions in self.direction_sets:
            direction, name = next(iter(directions.items()))
            rows.append(
                (name, {g: directions[turn(direction, g)] for g in names})
            )
        for name, table in self.symmetry.items():
            rows.append((name, _only(names, {**table, "identity": name})))
        return rows


def _only(
    names: tuple[SquareSymmetry, ...],
    given: Mapping[SquareSymmetry, str | None],
) -> dict[SquareSymmetry, str | None]:
    """Return a full table that makes no valid sample outside ``given``."""
    return {g: given.get(g) for g in names}


def _named_classes(name: str, meaning: ClassMeaning) -> set[str]:
    """Return the classes that a meaning of ``name`` names."""
    named = {name}
    if meaning.mirror is not None:
        named.add(meaning.mirror)
    if meaning.direction is not None:
        named |= set(meaning.direction.values())
    if meaning.symmetry is not None:
        named |= {t for t in meaning.symmetry.values() if t is not None}
    return named


def _orbit(
    representative: str, row: Mapping[SquareSymmetry, str | None]
) -> dict[str, dict[SquareSymmetry, str | None]]:
    """Derive the table of every class that ``row`` reaches.

    The class that symmetry ``h`` makes of the representative turns, under
    ``g``, into the class that ``g`` after ``h`` makes of it.
    """
    names = get_args(SquareSymmetry)
    tables: dict[str, dict[SquareSymmetry, str | None]] = {}
    for h in names:
        reached = row[h]
        if reached is None:
            continue
        derived = {g: row[compose(g, h)] for g in names}
        if reached in tables and tables[reached] != derived:
            raise ValueError(
                f"The symmetry table of '{representative}' is not "
                f"consistent: two symmetries that make '{reached}' disagree "
                "about what the other symmetries make of it."
            )
        tables[reached] = derived
    return tables


__all__ = [
    "SYMMETRIES",
    "ClassMeaning",
    "ClassMetadata",
    "Direction",
    "SquareSymmetry",
    "compose",
    "nearest_symmetry",
    "turn",
]
