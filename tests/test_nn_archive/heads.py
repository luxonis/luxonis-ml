from luxonis_ml.nn_archive.config_building_blocks import Head
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_metadata import (
    HeadClassificationMetadata,
    HeadMetadata,
    HeadObjectDetectionMetadata,
    HeadObjectDetectionSSDMetadata,
    HeadSegmentationMetadata,
    HeadYOLOMetadata,
)

head_metadata = HeadMetadata(postprocessor_path="postprocessor.onnx")

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
    boxes_outputs="boxes",
    scores_outputs="scores",
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
    subtype="yolov6",
    postprocessor_path=None,
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos_outputs=None,
    keypoints_outputs=None,
    angles_outputs=None,
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
    subtype="yolov6",
    postprocessor_path="postprocessor.onnx",
    yolo_outputs=["feats"],
    mask_outputs=["mask"],
    protos_outputs="protos",
    keypoints_outputs=None,
    angles_outputs=None,
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
    subtype="yolov6",
    postprocessor_path=None,
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos_outputs=None,
    keypoints_outputs=["keypoints"],
    angles_outputs=None,
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
    subtype="yolov6",
    postprocessor_path="postprocessor.onnx",
    yolo_outputs=["feats"],
    mask_outputs=None,
    protos_outputs=None,
    keypoints_outputs=None,
    angles_outputs=["angles"],
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
    subtype="yolov6",
    postprocessor_path="postprocessor.onnx",
    yolo_outputs=["feats"],
    mask_outputs=["mask"],
    protos_outputs="protos",
    keypoints_outputs=["keypoints"],
    angles_outputs=None,
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
        name="ClassificationHead",
        parser="Classification",
        outputs=["output"],
        metadata=head_classification_metadata,
    )
)

ssd_object_detection_head = dict(
    Head(
        name="ObjectDetectionSSDHead",
        parser="ObjectDetectionSSD",
        outputs=["boxes"],
        metadata=head_object_detection_ssd_metadata,
    )
)

yolo_object_detection_head = dict(
    Head(
        name="YoloDetectionHead",
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_segmentation_head = dict(
    Head(
        name="YoloInstanceSegHead",
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_instance_seg_metadata,
    )
)

yolo_keypoint_detection_head = dict(
    Head(
        name="YoloKeypointDetectionHead",
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_keypoint_det_metadata,
    )
)

yolo_obb_detection_head = dict(
    Head(
        name="YoloOBBHead",
        parser="YOLO",
        outputs=["output"],
        metadata=head_yolo_obb_det_metadata,
    )
)

yolo_instance_seg_kpts_head = dict(
    Head(
        name="YoloInstaceSegKptHead",
        parser="YOLO",
        outputs=["outputs"],
        metadata=head_yolo_instance_seg_kpts_metadata,
    )
)

custom_segmentation_head = dict(
    Head(
        name="SegmentationHead",
        parser="Segmentation",
        outputs=["output"],
        metadata=head_segmentation_metadata,
    )
)
