from luxonis_ml.nn_archive.config_building_blocks import (
    HeadClassification,
    HeadObjectDetectionSSD,
    HeadYOLO,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_outputs import (
    OutputsClassification,
    OutputsSSD,
    OutputsYOLO,
)
from luxonis_ml.nn_archive.config_building_blocks.enums import (
    ObjectDetectionSubtypeYOLO,
)

output_classification = OutputsClassification(predictions="191")

output_ssd = OutputsSSD(
    boxes="boxes",
    scores="scores",
)

output_detection = OutputsYOLO(
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos=None,
    keypoints=None,
    angles=None,
)

output_instance_segmentation = OutputsYOLO(
    yolo_outputs=["feats"],
    mask_outputs=["mask"],
    protos="protos",
    keypoints=None,
    angles=None,
)

output_keypoint_detection = OutputsYOLO(
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos=None,
    keypoints="keypoints",
    angles=None,
)

output_obb = OutputsYOLO(
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos=None,
    keypoints=None,
    angles="angles",
)

classification_head = dict(
    HeadClassification(
        family="Classification",
        outputs=output_classification,
        classes=[
            "tench, Tinca tinca",
            "goldfish, Carassius auratus",
        ],
        n_classes=2,
        is_softmax=True,
    )
)

ssd_object_detection_head = dict(
    HeadObjectDetectionSSD(
        family="ObjectDetectionSSD",
        outputs=output_ssd,
        classes=[
            "tench, Tinca tinca",
            "goldfish, Carassius auratus",
        ],
        n_classes=2,
        iou_threshold=0.5,
        conf_threshold=0.5,
        max_det=1000,
        anchors=None,
    )
)

yolo_object_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        outputs=output_detection,
        subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
        classes=["person", "car"],
        n_classes=2,
        iou_threshold=0.5,
        conf_threshold=0.5,
        max_det=1000,
        is_softmax=None,
        anchors=None,
        postprocessor_path=None,
        n_prototypes=None,
        n_keypoints=None,
    )
)

yolo_instance_segmentation_head = dict(
    HeadYOLO(
        family="YOLO",
        outputs=output_instance_segmentation,
        subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
        classes=["person", "car"],
        n_classes=2,
        iou_threshold=0.5,
        conf_threshold=0.5,
        max_det=1000,
        is_softmax=True,
        anchors=None,
        postprocessor_path="postprocessor.onnx",
        n_prototypes=10,
        n_keypoints=None,
    )
)

yolo_keypoint_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        outputs=output_keypoint_detection,
        subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
        classes=["person", "car"],
        n_classes=2,
        iou_threshold=0.5,
        conf_threshold=0.5,
        max_det=1000,
        is_softmax=None,
        anchors=None,
        postprocessor_path=None,
        n_prototypes=None,
        n_keypoints=21,
    )
)

yolo_obb_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        outputs=output_obb,
        subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
        classes=["person", "car"],
        n_classes=2,
        iou_threshold=0.5,
        conf_threshold=0.5,
        max_det=1000,
        is_softmax=None,
        anchors=None,
        postprocessor_path=None,
        n_prototypes=None,
        n_keypoints=None,
    )
)
