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
import cv2
import numpy as np
from typing import List, Tuple, Callable
import onnxruntime as ort
from io import BytesIO

from luxonis_ml.utils import LuxonisFileSystem


def preprocess_image_cv2(img: np.ndarray) -> np.ndarray:
    """Custom preprocessing function provided by the user."""
    # Resize and convert BGR to RGB
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img - mean) / std
    
    # Change to channel-first format
    img_transposed = img_normalized.transpose(2, 0, 1)
    
    return img_transposed

def get_image_tensors_from_LFS(
    image_paths: List[str],
    preprocess_function: Callable[[np.ndarray], np.ndarray],
    lfs: LuxonisFileSystem,
) -> Tuple[np.ndarray, List[int]]:
    tensors = []
    successful_ixs = []

    for i, path in enumerate(image_paths):
        try:
            # Read image data into a byte buffer
            buffer = lfs.read_to_byte_buffer(remote_path=path).getvalue()
            image = np.frombuffer(buffer, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            processed_image = preprocess_function(image)
            tensors.append(processed_image)
            successful_ixs.append(i)
        except Exception as e:
            print(f"Error occurred while processing image: {path}, Error: {e}")
            continue
    
    return np.stack(tensors), successful_ixs

def extract_embeddings(
    image_paths: List[str],
    ort_session: ort.InferenceSession,
    lfs: LuxonisFileSystem, 
    preprocess_function: Callable[[np.ndarray], np.ndarray],
    output_layer_name: str = "/Flatten_output_0",
    batch_size: int = 64,
) -> Tuple[List[List[float]], List[int]]:
    embeddings = []
    successful_ixs = []

    if preprocess_function is None:
        preprocess_function = preprocess_image_cv2

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors, batch_ixs = get_image_tensors_from_LFS(batch_paths, preprocess_function, lfs)

        # Prepare inputs for ONNX runtime
        ort_inputs = {ort_session.get_inputs()[0].name: batch_tensors.astype(np.float32)}
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)[0]

        embeddings.extend(ort_outputs.tolist())
        successful_ixs.extend([i + ix for ix in batch_ixs])

    return embeddings, successful_ixs
