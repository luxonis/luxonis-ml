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

from typing import Tuple

import onnxruntime as ort
import torch


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
