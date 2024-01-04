"""Model Utility Functions.

This script provides utility functions for handling pytorch models.
It allows loading the model, exporting it to the ONNX format,
and manipulating ONNX models in order to extract intermediate outputs aka embeddings.

Functions:
    - load_model_resnet50_minus_last_layer() -> nn.Module:
        Returns a pre-trained ResNet-50 model with its last fully connected layer removed.

    - load_model() -> nn.Module:
        Returns a fully pre-trained ResNet-50 model.

    - export_model_onnx(model: nn.Module, model_path_out: str = "resnet50.onnx"):
        Exports the provided PyTorch model into the ONNX format.

    - load_model_onnx(model_path: str = "resnet50.onnx") -> onnx.ModelProto:
        Loads an ONNX model from the specified file path.

    - extend_output_onnx(onnx_model: onnx.ModelProto, intermediate_tensor_name: str = "/Flatten_output_0") -> onnx.ModelProto:
        Sets an intermediate output layer as output of the provided ONNX model.

    - extend_output_onnx_overwrite(onnx_model: onnx.ModelProto, intermediate_tensor_name: str = "/Flatten_output_0") -> onnx.ModelProto:
        Sets the second to last layer ouput as output layer of the provided ONNX model, and renames it.

Usage:
This utility can be imported as a module and provides capabilities to manage and manipulate ResNet-50 and its ONNX counterpart.

Example:
    >>> from luxonis_ml.embeddings.model_utils import load_model, export_model_onnx, load_model_onnx, extend_output_onnx_overwrite
    >>> model = load_model()
    >>> export_model_onnx(model, "path_to_save.onnx")
    >>> onnx_model = load_model_onnx("path_to_save.onnx")
    >>> onnx_model_new = extend_output_onnx_overwrite(onnx_model, "/Flatten_output_0")

Dependencies:
    - torch
    - torch.nn
    - torch.onnx
    - onnx
    - torchvision.models
    - torchvision.models.resnet
"""

import onnx
import torch
import torch.nn as nn
import torch.onnx
import torchvision.models as models
import torchvision.models.resnet as resnet


def load_model_resnet50_minuslastlayer() -> nn.Module:
    """Load a pre-trained ResNet-50 model with the last fully connected layer
    removed."""
    # model = models.resnet50(pretrained=True) # depricated
    model = models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(
        *list(model.children())[:-1]
    )  # Remove the last fully connected layer
    model.eval()
    return model


def load_model() -> nn.Module:
    """Load a pre-trained ResNet-50 model."""
    model = models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def export_model_onnx(model: nn.Module, model_path_out: str = "resnet50.onnx"):
    """Export the provided model to the ONNX format."""
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        model_path_out,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def load_model_onnx(model_path: str = "resnet50.onnx") -> onnx.ModelProto:
    """Load an ONNX model from the provided path."""
    return onnx.load(model_path)


def extend_output_onnx(
    onnx_model: onnx.ModelProto, intermediate_tensor_name: str
) -> onnx.ModelProto:
    """Set an intermediate output layer as output of the provided ONNX model.

    (You need to know the name of the intermediate layer, which you can find by
    inspecting the ONNX model with Netron.app)
    """
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.extend([intermediate_layer_value_info])
    return onnx_model


def extend_output_onnx_overwrite(
    onnx_model: onnx.ModelProto, intermediate_tensor_name: str = "/Flatten_output_0"
) -> onnx.ModelProto:
    """Set the second to last layer ouput as output layer of the provided ONNX model,
    and rename it."""
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
