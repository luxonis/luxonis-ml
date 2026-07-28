"""Compatibility coverage for the reorganized Vizlab implementation."""


def test_legacy_module_paths_reexport_reorganized_components() -> None:
    from luxonis_ml.vizlab.adapters.ldf import (
        visualize_record as adapter_visualize,
    )
    from luxonis_ml.vizlab.canvas import Canvas
    from luxonis_ml.vizlab.compare import compare
    from luxonis_ml.vizlab.comparison import compare as package_compare
    from luxonis_ml.vizlab.compose import grid
    from luxonis_ml.vizlab.convert import visualize_record
    from luxonis_ml.vizlab.frame import Frame
    from luxonis_ml.vizlab.image import Image
    from luxonis_ml.vizlab.interaction.frame import Frame as InteractionFrame
    from luxonis_ml.vizlab.layout.compose import grid as layout_grid
    from luxonis_ml.vizlab.render.canvas import Canvas as RenderCanvas
    from luxonis_ml.vizlab.scene import Image as SceneImage

    assert Canvas is RenderCanvas
    assert Frame is InteractionFrame
    assert Image is SceneImage
    assert compare is package_compare
    assert grid is layout_grid
    assert visualize_record is adapter_visualize
