import torch
import torchvision

import onnxruntime as ort
import onnx

def extract_embeddings(model, data_loader):
    # Initialize lists to store the embeddings and corresponding labels
    embeddings = []
    labels = []

    # Extract embeddings from the dataset
    with torch.no_grad():
        for images, batch_labels in data_loader:
            outputs = model(images)
            embeddings.extend(outputs.squeeze())
            labels.extend(batch_labels)

    # Convert embeddings and labels to tensors
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)

    return embeddings, labels


def extract_embeddings_onnx(ort_session, data_loader, output_layer_name="/Flatten_output_0"):
    embeddings = []
    labels = []

    # Extract embeddings from the dataset
    with torch.no_grad():
        for images, batch_labels in data_loader:
            # Preprocess images and convert to ONNX Runtime compatible format
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}

            # Run inference with the ONNX Runtime session
            ort_outputs = ort_session.run([output_layer_name], ort_inputs)

            # Get the embeddings from the second-to-last layer
            embeddings.extend(torch.from_numpy(ort_outputs[0]).squeeze())
            labels.extend(batch_labels)
    
    # Convert embeddings and labels to tensors
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)

    return embeddings, labels


def save_embeddings(embeddings, labels, save_path="./"):
    # Save the embeddings tensor to the file
    torch.save(embeddings, save_path + 'embeddings.pth')

    # save the labels tensor to the file
    torch.save(labels, save_path + 'labels.pth')


def load_embeddings(save_path="./"):
    # Load the embeddings tensor from the file
    embeddings = torch.load(save_path + 'embeddings.pth')

    # Load the labels tensor from the file
    labels = torch.load(save_path + 'labels.pth')

    return embeddings, labels
