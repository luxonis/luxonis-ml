import torch
import torch.nn as nn

import torch.onnx
import onnx
import onnxruntime

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale, Lambda

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http import models


def load_data(save_path='./data', num_samples=640, batch_size=64):
    # Define the transformations for preprocessing the image data
    transform = transforms.Compose([
        Grayscale(num_output_channels=3),  # Convert images to grayscale
        Lambda(lambda x: x.convert("RGB")),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize images to (224, 224)
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the MNIST dataset
    dataset = torchvision.datasets.MNIST(root=save_path, train=True, transform=transform, download=True)

    # Define the indices to include in the subset (e.g., first 1000 samples)
    subset_indices = torch.arange(num_samples)

    # Create a subset of the dataset using Subset class
    subset = torch.utils.data.Subset(dataset, subset_indices)

    # Create a data loader to load the dataset in batches
    data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader

def load_model_resnet50_minuslastlayer():
    # Load the pre-trained ResNet-50 model
    model = torchvision.models.resnet50(pretrained=True)

    # Remove the last fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])

    # Set the model to evaluation mode
    model.eval()

    return model


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

def load_model():
    # Load the pre-trained ResNet-50 model
    model = torchvision.models.resnet50(pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    return model

def export_model_onnx(model, model_path_out="resnet18.onnx"):
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

def load_model_onnx(model_path="resnet18.onnx"):
    return onnx.load(model_path)

def extend_output_onnx(onnx_model, intermediate_tensor_name="/Flatten_output_0"):
    # Add an intermediate output layer
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    onnx_model.graph.output.extend([intermediate_layer_value_info])
    #onnx.save(onnx_model, model_path)
    return onnx_model

def extract_embeddings_onnx(ort_session, data_loader):
    embeddings = []
    labels = []

    # Extract embeddings from the dataset
    with torch.no_grad():
        for images, batch_labels in data_loader:
            # Preprocess images and convert to ONNX Runtime compatible format
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}

            # Run inference with the ONNX Runtime session
            ort_outputs = ort_session.run(["/Flatten_output_0"], ort_inputs)

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

def connect_to_qdrant(host="localhost", port=6333):
    client = QdrantClient(host=host, port=port)
    return client

def create_collection(client, collection_name="mnist", vector_size=512, distance=Distance.COSINE):
    # Check if the collection already exists
    try:
        client.get_collection(collection_name=collection_name)
    except:
        # Create a collection with the given name and vector configuration
        client.recreate_collection(collection_name=collection_name,
                                    vectors_config=VectorParams(size=vector_size, 
                                                                distance=distance))

def insert_embeddings(client, embeddings, labels, collection_name="mnist"):
    # Insert the embeddings into the collection    
    client.upsert(collection_name=collection_name, 
                    points= [PointStruct(
                            id=i,
                            vector=embeddings[i].tolist(),
                            payload={"label": labels[i].item()}
                        ) for i in range(len(embeddings))])


def batch_insert_embeddings(client, embeddings, labels, img_paths, batch_size = 50, collection_name="mnist"):
    total_len = len(embeddings)

    for i in range(0, total_len, batch_size):
        start = i
        end = min(i + batch_size, total_len)

        # Ensure IDs start from 1, not 0
        batch_ids = list(range(start + 1, end + 1))
        batch_vectors = embeddings[start:end].tolist()
        batch_payloads = [{'label': labels[j].item(), 'image_path': img_paths[j]} for j in range(start, end)]

        batch = models.Batch(
            ids=batch_ids, 
            vectors=batch_vectors, 
            payloads=batch_payloads
        )

        # Upsert the batch of points to the Qdrant collection
        client.upsert(collection_name=collection_name, points=batch)


def search_embeddings(client, embedding, collection_name="mnist", top=5):
    # Search for the nearest neighbors
    search_results = client.search(collection_name=collection_name, 
                                    query_vector=embedding.tolist(), 
                                    limit=top)
    return search_results

def search_embeddings_by_imagepath(client, embedding, image_path_part, collection_name="mnist", top=5):
    hits = client.search(
        collection_name=collection_name,
        query_vector=embedding.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="image_path",
                    match=models.MatchText(text=image_path_part)
                )
            ] 
        ),
        limit= top
    )
    return hits


def main_pytorch():
    # Load the data
    data_loader = load_data()

    # Load the model
    model = load_model_resnet50_minuslastlayer()

    # Extract embeddings from the dataset
    embeddings, labels = extract_embeddings(model, data_loader)

    save_embeddings(embeddings, labels)


def main_onnx():
    # Load the data
    data_loader = load_data()

    # Load the model
    model = load_model()

    # Export the model to ONNX
    export_model_onnx(model, model_path_out="resnet50.onnx")

    # Load the ONNX model
    onnx_model = load_model_onnx(model_path="resnet50.onnx")

    # Extend the ONNX model with an intermediate output layer
    onnx_model = extend_output_onnx(onnx_model, intermediate_tensor_name="/Flatten_output_0")

    # Save the ONNX model
    onnx.save(onnx_model, "resnet50-1.onnx")

    # Create an ONNX Runtime session
    ort_session = onnxruntime.InferenceSession("resnet50-1.onnx")
    
    # Extract embeddings from the dataset
    embeddings, labels = extract_embeddings_onnx(ort_session, data_loader)

    # Save the embeddings and labels to a file
    save_embeddings(embeddings, labels)


    # Connect to Qdrant
    client = connect_to_qdrant()

    # Create a collection
    vector_size = embeddings.shape[1]
    create_collection(client, collection_name="mnist", vector_size=vector_size, distance=Distance.COSINE)

    # Insert the embeddings into the collection
    insert_embeddings(client, embeddings, labels, collection_name="mnist")

    # Search for the nearest neighbors
    search_results = search_embeddings(client, embeddings[0], collection_name="mnist", top=5)

    # Print the search results
    print(search_results)


if __name__ == '__main__':
    # main_pytorch()
    main_onnx()