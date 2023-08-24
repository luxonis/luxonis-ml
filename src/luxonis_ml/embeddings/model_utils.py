import torch
import torch.nn as nn
import torch.onnx
import onnx
import torchvision

def load_model_resnet50_minuslastlayer():
    # Load the pre-trained ResNet-50 model
    # model = torchvision.models.resnet50(pretrained=True)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    # Remove the last fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])

    # Set the model to evaluation mode
    model.eval()

    return model

def load_model():
    # Load the pre-trained ResNet-50 model
    # model = torchvision.models.resnet50(pretrained=True)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    # Set the model to evaluation mode
    model.eval()

    return model

def export_model_onnx(model, model_path_out="resnet50.onnx"):
    # Create a dummy input tensor
    img1 = torch.randn(1, 3, 224, 224)

    # Invoke export
    torch.onnx.export(model, 
                        img1, 
                        model_path_out,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=False,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

def load_model_onnx(model_path="resnet50.onnx"):
    return onnx.load(model_path)

def extend_output_onnx(onnx_model, intermediate_tensor_name="/Flatten_output_0"):
    # Add an intermediate output layer
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.extend([intermediate_layer_value_info])
    #onnx.save(onnx_model, model_path)
    return onnx_model

def extend_output_onnx_overwrite(onnx_model, intermediate_tensor_name="/Flatten_output_0"):
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)
    # Identify the second to last node
    second_to_last_node = onnx_model.graph.node[-2]
    # Store the old name of the second to last node's output
    old_name = second_to_last_node.output[0]
    # Rename the output of the second to last node
    second_to_last_node.output[0] = intermediate_tensor_name

    # Update subsequent nodes that use the old name as an input
    for node in onnx_model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[idx] = intermediate_tensor_name

    # Update the model's outputs to include the new intermediate tensor name
    intermediate_tensor_name = intermediate_tensor_name
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.extend([intermediate_layer_value_info])

    return onnx_model
