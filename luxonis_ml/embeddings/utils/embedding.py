"""Embeddings Extractor and Storage for ONNX Models.

This module provides utility functions specifically for extracting embeddings from ONNX models,
reading images from Luxonis Filesystem (LFS), and storing/retrieving embeddings to/from disk.

Key Functions:
   - extract_embeddings(image_paths, ort_session, lfs, ...)
     Extracts embeddings from an ONNX model, reading images from LFS and handling potential errors.

   - get_image_tensors_from_LFS(image_paths, preprocess_function, lfs)
     Reads images from LFS, applies preprocessing, and returns a list of tensors.

   - preprocess_image_cv2(img)
     Example preprocessing function for resizing, normalization, and channel arrangement.

Workflow:
    1. Load the ONNX model into an ONNX Runtime session.
    2. Provide a list of image paths (stored on LFS) to the C{extract_embeddings} function.
    3. Optionally, specify a custom preprocessing function for image preparation.
    4. The function extracts embeddings in batches, handles errors, and returns the extracted embeddings.

Additional Features:
    - Saving and loading embeddings to/from disk (functions not yet implemented in this version).

Dependencies:
   - onnxruntime
   - cv2
   - numpy

Note:
    - Ensure the output_layer_name in C{extract_embeddings} matches the appropriate output layer in the ONNX model.
    - This module specifically focuses on ONNX models and reading images from Luxonis Filesystem.
"""

from typing import Callable, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

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


def get_image_tensors_from_remote(
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
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors, batch_ixs = get_image_tensors_from_remote(
            batch_paths, preprocess_function, lfs
        )

        # Prepare inputs for ONNX runtime
        ort_inputs = {
            ort_session.get_inputs()[0].name: batch_tensors.astype(np.float32)
        }
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)[0]

        embeddings.extend(ort_outputs.tolist())
        successful_ixs.extend([i + ix for ix in batch_ixs])

    return embeddings, successful_ixs
