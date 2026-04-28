from luxonis_ml.data.utils.augmentations_collector import (
    AugmentationsCollector,
)


def test_collect_applied_transform_paths_collects_nested_paths():
    rotate = type("Rotate", (), {"params": {"shape": (32, 32, 3)}})()
    blur = type("Blur", (), {"params": {}})()
    inner_one_of = type(
        "OneOf",
        (),
        {"transforms": [blur, rotate]},
    )()
    horizontal_flip = type(
        "HorizontalFlip", (), {"params": {"shape": (32, 32, 3)}}
    )()
    root_compose = type(
        "Compose",
        (),
        {"transforms": [horizontal_flip, inner_one_of]},
    )()

    assert AugmentationsCollector.collect_applied_transform_paths(
        root_compose
    ) == [
        "HorizontalFlip",
        "OneOf/Rotate",
    ]


def test_flatten_config_augmentation_paths_handles_nested_transforms():
    config = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "Blur", "params": {"p": 1.0}},
                    {"name": "Sharpen", "params": {"p": 1.0}},
                ]
            },
        },
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
        "OneOf/Sharpen",
    ]


def test_flatten_config_augmentation_paths_handles_deeply_nested_transforms():
    config = [
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {
                        "name": "OneOf",
                        "params": {
                            "transforms": [
                                {"name": "Rotate", "params": {"p": 1.0}}
                            ]
                        },
                    }
                ]
            },
        }
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "OneOf",
        "OneOf/OneOf",
        "OneOf/OneOf/Rotate",
    ]


def test_flatten_config_augmentation_paths_adds_resize_oneof_alias():
    config = [
        {
            "name": "AtLeastOneBBoxRandomCrop",
            "params": {
                "height": 32,
                "width": 32,
                "erosion_factor": 0.0,
                "p": 0.3,
            },
            "use_for_resizing": True,
        }
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "AtLeastOneBBoxRandomCrop",
        "OneOf/AtLeastOneBBoxRandomCrop",
    ]


def test_load_augmentation_paths_accepts_in_memory_config():
    config = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "Blur", "params": {"p": 1.0}},
                ]
            },
        },
    ]

    assert AugmentationsCollector.load_augmentation_paths(config) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
    ]
