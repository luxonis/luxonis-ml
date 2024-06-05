from luxonis_ml.nn_archive.config_building_blocks import (
    HeadClassification,
    HeadObjectDetectionSSD,
    HeadObjectDetectionYOLO,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_outputs import (
    OutputsClassification,
    OutputsInstanceSegmentationYOLO,
    OutputsKeypointDetectionYOLO,
    OutputsOBBDetectionYOLO,
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

output = OutputsYOLO(yolo_outputs=["feats"])

output_instance_segmentation = OutputsInstanceSegmentationYOLO(
    yolo_outputs=["feats"], mask_outputs=["mask"], protos="protos"
)

output_keypoint_detection = OutputsKeypointDetectionYOLO(
    yolo_outputs=["feats"], keypoints="keypoints"
)

output_obb = OutputsOBBDetectionYOLO(yolo_outputs=["feats"], angles="angles")

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
    HeadObjectDetectionYOLO(
        family="ObjectDetectionYOLO",
        outputs=output,
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
    HeadObjectDetectionYOLO(
        family="InstanceSegmentationYOLO",
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
    HeadObjectDetectionYOLO(
        family="KeypointDetectionYOLO",
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
    HeadObjectDetectionYOLO(
        family="OBBDetectionYOLO",
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
