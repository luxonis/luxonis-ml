"""Embeddings Extractor and Storage.

This module provides utility functions for extracting embeddings from both PyTorch and ONNX models,
and subsequently storing and retrieving these embeddings from disk.

Functions:
    - extract_embeddings(model, data_loader):
      Extracts embeddings from a given PyTorch model using data from a specified DataLoader.

    - extract_embeddings_onnx(ort_session, data_loader, output_layer_name):
      Extracts embeddings from a specified ONNX model (provided as an ONNX Runtime session)
      using data from a specified DataLoader. Allows targeting a specific output layer for extraction.

    - save_embeddings(embeddings, labels, save_path):
      Saves both embeddings and their associated labels to the disk at a given path.

    - load_embeddings(save_path):
      Loads embeddings and their associated labels from the disk at a given path.

Usage Examples:
    1. Extract embeddings from a PyTorch model:
        embeddings, labels = extract_embeddings(pytorch_model, data_loader)

    2. Extract embeddings from an ONNX model:
        ort_session = ort.InferenceSession('model.onnx')
        embeddings, labels = extract_embeddings_onnx(ort_session, data_loader, "/Flatten_output_0")

    3. Save embeddings to disk:
        save_embeddings(embeddings, labels, "./embeddings/")

    4. Load embeddings from disk:
        loaded_embeddings, loaded_labels = load_embeddings("./embeddings/")

Note:
Ensure the DataLoader provided to the extraction functions outputs batches
in the form (data, labels). Make sure to match the output_layer_name in the ONNX extraction
with the appropriate output layer's name from the ONNX model.

Dependencies:
    - torch
    - torchvision
    - onnxruntime
    - onnx
"""
from PIL import Image
from io import BytesIO
from typing import Tuple, List

import onnxruntime as ort
import torch
import torchvision.transforms as transforms

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import LuxonisFileSystem

def extract_embeddings(
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

def get_image_tensors_from_LFS(
    image_paths: List[str],
    transform: transforms.Compose,
    lfs: LuxonisFileSystem,
) -> Tuple[torch.Tensor, List[int]]:
    tensors = []
    successful_ixs = []

    for i, path in enumerate(image_paths):
        try:
            buffer = lfs.read_to_byte_buffer(remote_path=path).getvalue()
            image = Image.open(BytesIO(buffer)).convert('RGB')
            tensor = transform(image)
            tensors.append(tensor)
            successful_ixs.append(i)
        except:
            print("Error occured while processing image: ", path)
            continue

    return torch.stack(tensors), successful_ixs

def extract_embeddings_onnx_LFS(
    image_paths: List[str],
    ort_session: ort.InferenceSession,
    transform: transforms.Compose,
    lfs: LuxonisFileSystem,
    output_layer_name: str = "/Flatten_output_0",
    batch_size: int = 64,
) -> Tuple[torch.Tensor, List[int]]:
    embeddings = []
    successful_ixs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors, batch_ixs = get_image_tensors_from_LFS(batch_paths, transform, lfs)

        # Extract embeddings using ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: batch_tensors.numpy()}
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)[0]

        embeddings.extend(torch.from_numpy(ort_outputs).squeeze())

        batch_ixs = [i + ix for ix in batch_ixs]
        successful_ixs.extend(batch_ixs)

    return torch.stack(embeddings), successful_ixs

def save_embeddings(
    embeddings: torch.Tensor, labels: torch.Tensor, save_path: str = "./"
):
    """Save embeddings and labels tensors to the specified path."""
    torch.save(embeddings, save_path + "embeddings.pth")
    torch.save(labels, save_path + "labels.pth")


def load_embeddings(save_path: str = "./") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load embeddings and labels tensors from the specified path."""
    embeddings = torch.load(save_path + "embeddings.pth")
    labels = torch.load(save_path + "labels.pth")

    return embeddings, labels
