from luxonis_ml.nn_archive.config_building_blocks import Head
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_metadata import (
    HeadClassificationMetadata,
    HeadMetadata,
    HeadObjectDetectionMetadata,
    HeadObjectDetectionSSDMetadata,
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
    postprocessor_path="postprocessor.onnx",
)

head_object_detection_metadata = HeadObjectDetectionMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path=None,
)

head_classification_metadata = HeadClassificationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=True,
    postprocessor_path="postprocessor.onnx",
)

head_segmentation_metadata = HeadSegmentationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=True,
    postprocessor_path=None,
)

head_object_detection_ssd_metadata = HeadObjectDetectionSSDMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    postprocessor_path=None,
    outputs=output_ssd,
)

head_yolo_obj_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    n_prototypes=None,
    n_keypoints=None,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
    postprocessor_path=None,
    outputs=output_detection,
)

head_yolo_instance_seg_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    n_prototypes=10,
    n_keypoints=None,
    is_softmax=True,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
    postprocessor_path="postprocessor.onnx",
    outputs=output_instance_segmentation,
)

head_yolo_keypoint_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    n_prototypes=None,
    n_keypoints=21,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
    postprocessor_path=None,
    outputs=output_keypoint_detection,
)

head_yolo_obb_det_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    n_prototypes=None,
    n_keypoints=None,
    is_softmax=None,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
    postprocessor_path="postprocessor.onnx",
    outputs=output_obb,
)

head_yolo_instance_seg_kpts_metadata = HeadYOLOMetadata(
    classes=["person", "car"],
    n_classes=2,
    iou_threshold=0.5,
    conf_threshold=0.5,
    max_det=1000,
    anchors=None,
    n_prototypes=10,
    n_keypoints=21,
    is_softmax=False,
    subtype=ObjectDetectionSubtypeYOLO.YOLOv6,
    postprocessor_path="postprocessor.onnx",
    outputs=outputs_instance_seg_kpts,
)

head_segmentation_metadata = HeadSegmentationMetadata(
    classes=["person", "car"],
    n_classes=2,
    is_softmax=False,
    custom_field="custom_field",  # type: ignore
    postprocessor_path="postprocessor.onnx",
)

classification_head = dict(
    Head(
        parser="Classification",
        outputs=["output"],
        metadata=head_classification_metadata,
    )
)

ssd_object_detection_head = dict(
    Head(
        parser="ObjectDetectionSSD",
        outputs=["boxes"],
        metadata=head_object_detection_ssd_metadata,
    )
)

yolo_object_detection_head = dict(
    Head(
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_segmentation_head = dict(
    Head(
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_instance_seg_metadata,
    )
)

yolo_keypoint_detection_head = dict(
    Head(
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_keypoint_det_metadata,
    )
)

yolo_obb_detection_head = dict(
    Head(
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_seg_kpts_head = dict(
    Head(
        parser="YOLO",
        outputs=["outputs"],
        metadata=head_yolo_instance_seg_kpts_metadata,
    )
)

custom_segmentation_head = dict(
    Head(
        parser="Segmentation",
        outputs=["output"],
        metadata=head_segmentation_metadata,
    )
)
