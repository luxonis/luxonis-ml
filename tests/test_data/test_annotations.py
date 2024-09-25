from luxonis_ml.data.datasets.utils import add_generator_wrapper, rescale_values


def test_rescale_values_keypoints():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5}
    keypoints = [(0.2, 0.4, 2), (0.5, 0.8, 2)]
    expected = [
        (0.2 * 0.5 + 0.1, 0.4 * 0.5 + 0.2, 2),
        (0.5 * 0.5 + 0.1, 0.8 * 0.5 + 0.2, 2),
    ]
    assert rescale_values(bbox, keypoints, "keypoints") == expected


def test_rescale_values_segmentation_polyline():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5}
    segmentation = {"polylines": [[0.2, 0.4], [0.5, 0.8]]}
    expected = [(0.2 * 0.5 + 0.1, 0.4 * 0.5 + 0.2), (0.5 * 0.5 + 0.1, 0.8 * 0.5 + 0.2)]
    assert rescale_values(bbox, segmentation, "segmentation") == expected


def test_rescale_values_segmentation_rle():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5}
    segmentation = {"rle": {"size": [100, 100], "counts": [40, 0, 60, 0]}}
    # Dummy check: actual implementation requires detailed RLE handling
    result = rescale_values(bbox, segmentation, "segmentation")
    assert isinstance(result, dict)
    assert "height" in result
    assert "width" in result
    assert "counts" in result
    assert result["height"] == 50
    assert result["width"] == 50


def test_rescale_values_invalid_key():
    bbox = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5}
    ann = [0.2, 0.4, 2]
    assert rescale_values(bbox, ann, "invalid_key") is None  # type: ignore


def test_add_generator_wrapper_non_detection():
    def dummy_generator():
        yield {
            "file": "dummy.jpg",
            "annotation": {
                "instance_id": 0,
                "type": "segmentation",
                "class": "person",
                "segmentation": {
                    "polylines": [(0.2, 0.4), (0.5, 0.8), (0.7, 0.9)],
                },
            },
        }

    wrapped_gen = add_generator_wrapper(dummy_generator())
    result = next(wrapped_gen)
    assert isinstance(result, dict)
    assert "file" in result
    assert "annotation" in result
    assert result["annotation"]["type"] == "segmentation"


def test_add_generator_wrapper_missing_scaled_to_boxes():
    def dummy_generator():
        yield {
            "file": "dummy.jpg",
            "annotation": {
                "instance_id": 0,
                "type": "detection",
                "task": "people",
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.5},
                "keypoints": [(0.2, 0.4, 2), (0.5, 0.8, 2)],
            },
        }

    wrapped_gen = add_generator_wrapper(dummy_generator())

    tasks = []
    for record in wrapped_gen:
        assert isinstance(record, dict)
        assert "file" in record
        assert "annotation" in record
        tasks.append(record["annotation"]["task"])

    assert sorted(tasks) == ["people-boundingbox", "people-keypoints"]
