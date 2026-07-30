import numpy as np
import pytest

from luxonis_ml.vizlab import Image, combine
from luxonis_ml.vizlab.layout.compose import _smart_cols


def _cell(w: int = 10, h: int = 10) -> Image:
    return Image(np.zeros((h, w, 3), np.uint8))


def test_combine_single_image_passthrough() -> None:
    img = _cell()
    out = combine(img)
    # combine's contract: a brand-new Image, never the caller's own object, so
    # mutating the result can't reach back into the input's scene graph.
    assert out is not img
    assert out.annotations is not img.annotations
    assert out.render().shape == (10, 10, 4)


def test_combine_multiple_images() -> None:
    out = combine(_cell(), _cell(), _cell())
    rendered = out.render()
    assert rendered.shape[2] == 4
    # Wider than a single cell: several tiles were laid out together.
    assert rendered.shape[1] > 10


def test_combine_mixed_groups() -> None:
    gt, pred = _cell(), _cell()
    heat = [_cell(8, 8) for _ in range(3)]
    rendered = combine({"GT": gt, "Pred": pred}, heat).render()
    assert rendered.shape[2] == 4
    # Both the titled pair and the heatmap sub-grid were laid out, so the
    # composite is larger than a single cell in both dimensions; a regression
    # that dropped a group or collapsed the layout would fail this.
    assert rendered.shape[0] > 10
    assert rendered.shape[1] > 10


def test_combine_mapping_titles_add_height() -> None:
    plain = combine([_cell(), _cell()]).render()
    titled = combine({"GT": _cell(), "Pred": _cell()}).render()
    # Titles reserve a heading band, so the titled figure is taller.
    assert titled.shape[0] > plain.shape[0]


def test_combine_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty set of groups"):
        combine()


def test_combine_empty_group_raises() -> None:
    with pytest.raises(ValueError, match="empty group"):
        combine([])


def test_combine_empty_mapping_raises() -> None:
    with pytest.raises(ValueError, match="empty group"):
        combine({})


def test_combine_bad_type_raises() -> None:
    with pytest.raises(TypeError, match="unsupported group type"):
        combine(42)  # type: ignore[arg-type]


def test_combine_bad_sequence_member_raises() -> None:
    # A sequence of non-Image values (e.g. raw arrays) is a documented TypeError,
    # not a cryptic AttributeError from calling .render() on the member.
    with pytest.raises(TypeError, match="unsupported group member type"):
        combine([np.zeros((4, 4, 3), np.uint8)])  # type: ignore[list-item]


def test_combine_bad_mapping_member_raises() -> None:
    with pytest.raises(TypeError, match="unsupported group member type"):
        combine({"bad": 42})  # type: ignore[dict-item]


def test_combine_unwraps_single_image_mapping_members() -> None:
    rendered = combine({"only": [_cell()]}).render()
    assert rendered.shape[2] == 4


def test_combine_accepts_composites_as_groups() -> None:
    # `CombineGroup` is typed as any Renderable, and nesting an already-composed
    # scene is the point of `combine` — a grid, a panelled image, or either of
    # them inside a sequence or a titled mapping.
    from luxonis_ml.vizlab import grid

    sub = grid([_cell(), _cell()])
    assert combine(sub, _cell()).render().shape[2] == 4
    assert combine({"group": sub}).render().shape[2] == 4
    assert combine([sub, _cell()]).render().shape[2] == 4
    panelled = _cell(40, 40).with_panel({"k": "v"})
    assert combine([panelled, _cell()]).render().shape[2] == 4


def test_smart_cols_single() -> None:
    assert _smart_cols([_cell()]) == 1


def test_smart_cols_small_wide_single_row() -> None:
    # Three wide tiles -> one row.
    assert _smart_cols([_cell(20, 10) for _ in range(3)]) == 3


def test_smart_cols_small_tall_single_column() -> None:
    # Tall tiles (ratio < 0.6) -> one column.
    assert _smart_cols([_cell(10, 40) for _ in range(3)]) == 1


def test_smart_cols_square_near_sqrt() -> None:
    # Nine square tiles -> ~3 columns (near-square composite).
    assert _smart_cols([_cell(10, 10) for _ in range(9)]) == 3
