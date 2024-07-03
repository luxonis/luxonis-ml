from luxonis_ml.nn_archive.config_building_blocks import (
    HeadClassification,
    HeadObjectDetectionSSD,
    HeadSegmentation,
    HeadYOLO,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_metadata import (
    HeadClassificationMetadata,
    HeadMetadata,
    HeadObjectDetectionMetadata,
    HeadSegmentationMetadata,
    HeadYOLOMetadata,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_outputs import (
    OutputsClassification,
    OutputsSegmentation,
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

outputs_instance_seg_kpts = OutputsYOLO(
    yolo_outputs=["feats"],
    mask_outputs=["mask"],
    protos="protos",
    keypoints="keypoints",
    angles=None,
)

segmentation_output = OutputsSegmentation(
    predictions="output",
)

head_metadata = HeadMetadata(
    classes=["person", "car"],
    n_classes=2,
)

head_object_detection_metadata = HeadObjectDetectionMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
)

head_classification_metadata = HeadClassificationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=True,
)

head_segmentation_metadata = HeadSegmentationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=True,
)

head_yolo_obj_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path=None,
    n_prototypes=None,
    n_keypoints=None,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
)

head_yolo_instance_seg_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path="postprocessor.onnx",
    n_prototypes=10,
    n_keypoints=None,
    is_softmax=True,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
)

head_yolo_keypoint_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path=None,
    n_prototypes=None,
    n_keypoints=21,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
)

head_yolo_obb_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path=None,
    n_prototypes=None,
    n_keypoints=None,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
)

head_yolo_instance_seg_kpts_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path="postprocessor.onnx",
    n_prototypes=10,
    n_keypoints=21,
    is_softmax=False,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
)

head_segmentation_metadata = HeadSegmentationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=False,
    custom_field="custom_field",  # type: ignore
)

classification_head = dict(
    HeadClassification(
        family="Classification",
        parser_name="Classification",
        outputs=output_classification,
        metadata=head_classification_metadata,
    )
)

ssd_object_detection_head = dict(
    HeadObjectDetectionSSD(
        family="ObjectDetectionSSD",
        parser_name="ObjectDetectionSSD",
        outputs=output_ssd,
        metadata=head_object_detection_metadata,
    )
)

yolo_object_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        parser_name="YOLO",
        outputs=output_detection,
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_segmentation_head = dict(
    HeadYOLO(
        family="YOLO",
        outputs=output_instance_segmentation,
        parser_name="YOLO",
        metadata=head_yolo_instance_seg_metadata,
    )
)

yolo_keypoint_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        parser_name="YOLO",
        outputs=output_keypoint_detection,
        metadata=head_yolo_keypoint_det_metadata,
    )
)

yolo_obb_detection_head = dict(
    HeadYOLO(
        family="YOLO",
        parser_name="YOLO",
        outputs=output_obb,
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_seg_kpts_head = dict(
    HeadYOLO(
        family="YOLO",
        parser_name="YOLO",
        outputs=outputs_instance_seg_kpts,
        metadata=head_yolo_instance_seg_kpts_metadata,
    )
)

custom_segmentation_head = dict(
    HeadSegmentation(
        family="Segmentation",
        outputs=segmentation_output,
        metadata=head_segmentation_metadata,
        parser_name="PP-LiteSeg",
    )
)
