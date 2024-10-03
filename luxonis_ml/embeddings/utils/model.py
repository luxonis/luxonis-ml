"""Model Utility Functions.

This script provides utility functions for handling ONNX models.
It allows manipulating ONNX models in order to extract intermediate outputs aka embeddings.

Functions:
    - load_model_onnx(model_path: str = "resnet50.onnx") -> onnx.ModelProto:
        Loads an ONNX model from the specified file path.

    - save_model_onnx(model: onnx.ModelProto, model_path_out: str = "resnet50.onnx"):
        Saves an ONNX model to the specified file path.

    - extend_output_onnx(onnx_model: onnx.ModelProto, intermediate_tensor_name: str = "/Flatten_output_0") -> onnx.ModelProto:
        Sets an intermediate output layer as output of the provided ONNX model.
        If C{overwrite} is set to True, the second to last layer output will be set as output layer and renamed.

Dependencies:
    - onnx
"""

import onnx


def load_model_onnx(model_path: str = "resnet50.onnx") -> onnx.ModelProto:
    """Load an ONNX model from the provided path."""
    return onnx.load(model_path)


def save_model_onnx(
    model: onnx.ModelProto, model_path_out: str = "resnet50.onnx"
) -> None:
    """Save an ONNX model to the specified file path."""
    onnx.save(model, model_path_out)


def extend_output_onnx(
    onnx_model: onnx.ModelProto,
    intermediate_tensor_name: str = "/Flatten_output_0",
    overwrite: bool = False,
) -> onnx.ModelProto:
    """Set an intermediate output layer as output of the provided ONNX
    model.

    If C{overwrite} is set to True, the second to last layer output will
    be set as output layer and renamed.

    (You need to know the name of the intermediate layer, which you can
    find by inspecting the ONNX model with Netron.app)
    """
    if overwrite:
        onnx.checker.check_model(onnx_model)
        second_to_last_node = onnx_model.graph.node[-2]
        old_name = second_to_last_node.output[0]
        second_to_last_node.output[0] = intermediate_tensor_name

        for node in onnx_model.graph.node:
            for idx, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[idx] = intermediate_tensor_name

    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.extend([intermediate_layer_value_info])
    return onnx_model
