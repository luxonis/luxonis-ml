import cv2
import torch
import torch.nn as nn

import torch.onnx
import onnx
import onnxruntime

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale, Lambda

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from qdrant_client.http import models

from luxonis_ml.data.dataset import LuxonisDataset

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

def insert_embeddings_nooverwrite(client, embeddings, labels, collection_name="mnist"):
    # Create a list of search requests
    search_queries = [SearchRequest(
        vector=embeddings[i].tolist(),
        limit=1,
        score_threshold=0.9999) for i in range(len(embeddings))]
    
    # Search for the nearest neighbors
    batch_search_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries
    )
    
    # Get the indices of the embeddings that are not in the collection
    new_ix = [i for i, res in enumerate(batch_search_results) if len(res) == 0]
    if len(new_ix) == 0:
        print("No new embeddings to insert")
        return
    new_embeddings = embeddings[new_ix]
    new_labels = labels[new_ix]

    # Get the collection size - the number of points in the collection - start from this index
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Insert the embeddings into the collection
    client.upsert(collection_name=collection_name,
                    points= [PointStruct(
                            id=collection_size + i,
                            vector=new_embeddings[i].tolist(),
                            payload={"label": new_labels[i].item()}
                        ) for i in range(len(new_embeddings))])
    
    print("Inserted {} new embeddings".format(len(new_embeddings)))

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

def _get_sample_payloads(luxonis_dataset):
    # Iterate over the samples in the LuxonisDataset to get all img_paths and payloads
    all_img_paths = []
    all_payloads = []

    for sample in luxonis_dataset.fo_dataset:
        sample_id = sample['id']
        instance_id = sample['instance_id']
        filepath = sample['filepath']
        class_name = sample['class']
        split = sample['split']

        img_path = luxonis_dataset.path + sample.filepath.split(luxonis_dataset.path.split('/')[-1])[-1]
        all_img_paths.append(img_path)

        all_payloads.append({
            "sample_id": sample_id,
            "instance_id": instance_id,
            "filepath": filepath,
            "class": class_name,
            "split": split
        })
    
    return all_img_paths, all_payloads

def _filter_new_samples(qdrant_client, qdrant_collection_name="mnist", vector_size=2048, all_img_paths=[], all_payloads=[]):
    # Filter out samples that are already in the Qdrant database
    search_queries = [SearchRequest(
        vector=[0] * vector_size,  # Dummy vector
        limit=1,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="sample_id",
                    match=models.MatchText(text=payload["sample_id"])
                )
            ]
        )
    ) for payload in all_payloads]

    search_results = qdrant_client.search_batch(
        collection_name=qdrant_collection_name,
        requests=search_queries
    )

    new_img_paths = [all_img_paths[i] for i, res in enumerate(search_results) if not res]
    new_payloads = [all_payloads[i] for i, res in enumerate(search_results) if not res]

    return new_img_paths, new_payloads

def _generate_new_embeddings(ort_session, output_layer_name="/Flatten_output_0", emb_batch_size=64, new_img_paths=[]):
    # Generate embeddings for the new images using batching
    new_embeddings = []

    for i in range(0, len(new_img_paths), emb_batch_size):
        batch_img_paths = new_img_paths[i:i+emb_batch_size]
        
        # Load and preprocess a batch of images
        batch_images = [cv2.imread(img_path) for img_path in batch_img_paths]
        batch_tensors = [torch.from_numpy(img).permute(2, 0, 1).float() for img in batch_images]
        batch_tensor = torch.stack(batch_tensors).cuda()

        # Run the ONNX model on the batch
        ort_inputs = {ort_session.get_inputs()[0].name: batch_tensor.cpu().numpy()}
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)
        
        # Append the embeddings from the batch to the new_embeddings list
        batch_embeddings = ort_outputs[0].squeeze()
        new_embeddings.extend(batch_embeddings.tolist())
    
    return new_embeddings

def _batch_upsert(qdrant_client, new_embeddings, new_payloads, qdrant_batch_size = 64, qdrant_collection_name="mnist"):
    # Perform batch upserts to Qdrant
    collection_info = qdrant_client.get_collection(collection_name=qdrant_collection_name)
    collection_size = collection_info.vectors_count
    current_id = collection_size

    for i in range(0, len(new_embeddings), qdrant_batch_size):
        batch_embeddings = new_embeddings[i:i+qdrant_batch_size]
        batch_payloads = new_payloads[i:i+qdrant_batch_size]

        points = [models.PointStruct(
            id=current_id + j,
            vector=embedding,
            payload=payload
        ) for j, (embedding, payload) in enumerate(zip(batch_embeddings, batch_payloads))]

        qdrant_client.upsert(collection_name=qdrant_collection_name, points=points)
        print(f"Upserted batch {i // qdrant_batch_size + 1} to Qdrant.")

def generate_embeddings(luxonis_dataset, 
                         ort_session, 
                         qdrant_client, 
                         qdrant_collection_name="mnist", 
                         output_layer_name="/Flatten_output_0", 
                         vector_size=2048,
                         emb_batch_size=64, 
                         qdrant_batch_size=64):
    
    all_img_paths, all_payloads = _get_sample_payloads(luxonis_dataset)

    new_img_paths, new_payloads = _filter_new_samples(qdrant_client, 
                                                      qdrant_collection_name, 
                                                      vector_size, 
                                                      all_img_paths,
                                                      all_payloads)
    
    new_embeddings = _generate_new_embeddings(ort_session,
                                                output_layer_name,
                                                emb_batch_size,
                                                new_img_paths)
    
    _batch_upsert(qdrant_client,
                    new_embeddings,
                    new_payloads,
                    qdrant_batch_size,
                    qdrant_collection_name)
    
    print("Embeddings generation and insertion completed!")



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

    embeddings, labels = load_embeddings()

    # Connect to Qdrant
    client = connect_to_qdrant()

    # Create a collection
    vector_size = embeddings.shape[1]
    create_collection(client, collection_name="mnist", vector_size=vector_size, distance=Distance.COSINE)

    # Insert the embeddings into the collection
    # insert_embeddings(client, embeddings, labels, collection_name="mnist")
    insert_embeddings_nooverwrite(client, embeddings, labels, collection_name="mnist")

    # Search for the nearest neighbors
    search_results = search_embeddings(client, embeddings[0], collection_name="mnist", top=5)

    # Print the search results
    print(search_results)


def main_Luxonis():
    # Initialize the ONNX Runtime session for the model
    onnx_model_path = "path_to_your_onnx_model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_session.set_providers(['CUDAExecutionProvider'])  # Run ONNX on CUDA

    # Initialize the Qdrant client
    client = QdrantClient(host="localhost", port=6333)

    # Initialize the LuxonisDataset
    team_id = "your_team_id"
    team_name = "your_team_name"
    dataset_name = "your_dataset_name"
    dataset_id = LuxonisDataset.create(team_id, team_name, dataset_name)

    # TODO: Upload your dataset to Luxonis

    # Load the LuxonisDataset
    with LuxonisDataset(team_id=team_id, dataset_id=dataset_id) as dataset:
        # Call the _generate_embeddings method
        generate_embeddings(ort_session, client, "your_qdrant_collection_name", dataset, "your_exposed_layer_name")


if __name__ == '__main__':
    # main_pytorch()
    main_onnx()