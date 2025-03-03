from typing import List, Tuple

import cv2
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import resnet


# PyTorch and ONNX model loading and exporting functions
def load_model_resnet50(discard_last_layer: bool) -> nn.Module:
    """Load a pre-trained ResNet-50 model with the last fully connected
    layer removed."""
    model = models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    if discard_last_layer:
        model = nn.Sequential(
            *list(model.children())[:-1]
        )  # Remove the last fully connected layer
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


# Embedding extraction and saving functions
def extract_embeddings_torch(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings from the given PyTorch model."""
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            outputs = model(images)
            embeddings.extend(outputs.squeeze())
            labels.extend(batch_labels)

    return torch.stack(embeddings), torch.tensor(labels)


def extract_embeddings_onnx(
    ort_session: ort.InferenceSession,
    data_loader: torch.utils.data.DataLoader,
    output_layer_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings from the given ONNX model."""
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            outputs = ort_session.run([output_layer_name], ort_inputs)[0]
            embeddings.extend(torch.from_numpy(outputs).squeeze())
            labels.extend(batch_labels)

    return torch.stack(embeddings), torch.tensor(labels)


def save_embeddings(
    embeddings: torch.Tensor, labels: torch.Tensor, save_path: str = "./"
):
    """Save embeddings and labels tensors to the specified path."""
    torch.save(embeddings, save_path + "embeddings.pth")  # nosemgrep
    torch.save(labels, save_path + "labels.pth")  # nosemgrep


def load_embeddings(
    save_path: str = "./",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings and labels tensors from the specified path."""
    embeddings = torch.load(save_path + "embeddings.pth")  # nosemgrep
    labels = torch.load(save_path + "labels.pth")  # nosemgrep

    return embeddings, labels


# Image loading and preprocessing functions
def generate_new_embeddings(
    img_paths: List[str],
    ort_session: ort.InferenceSession,
    output_layer_name: str = "/Flatten_output_0",
    emb_batch_size: int = 64,
    transform: transforms.Compose = None,
):
    """Generate embeddings for new images using a given ONNX runtime
    session.

    @type img_paths: List[str]
    @param img_paths: List of image paths for new images.
    @type ort_session: L{InferenceSession}
    @param ort_session: ONNX runtime session.
    @type output_layer_name: str
    @param output_layer_name: Name of the output layer in the ONNX
        model.
    @type emb_batch_size: int
    @param emb_batch_size: Batch size for generating embeddings.
    @type transform: torchvision.transforms
    @param transform: Optional torchvision transform for preprocessing
        images.
    @rtype: List[List[float]]
    @return: List of embeddings for the new images.
    """
    # Generate embeddings for the new images using batching
    new_embeddings = []

    if transform is None:
        # Define a transformation for resizing and normalizing the images
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (224, 224)
                ),  # Resize images to (224, 224) or any desired size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize images
            ]
        )

    for i in range(0, len(img_paths), emb_batch_size):
        batch_img_paths = img_paths[i : i + emb_batch_size]

        # Load, preprocess, and resize a batch of images
        batch_images = [cv2.imread(img_path) for img_path in batch_img_paths]
        batch_tensors = [transform(img) for img in batch_images]
        batch_tensor = torch.stack(batch_tensors).cuda()

        # Run the ONNX model on the batch
        ort_inputs = {
            ort_session.get_inputs()[0].name: batch_tensor.cpu().numpy()
        }
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)

        # Append the embeddings from the batch to the new_embeddings list
        batch_embeddings = ort_outputs[0].squeeze()
        new_embeddings.extend(batch_embeddings.tolist())

    return new_embeddings
