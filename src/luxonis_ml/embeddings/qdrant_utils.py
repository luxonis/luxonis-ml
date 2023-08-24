import subprocess
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from qdrant_client.http import models

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return output
    except subprocess.CalledProcessError as e:
        error_message = e.output.decode('utf-8').strip()
        print(f"Error executing command: {error_message}")
        return error_message

def docker_installed():
    output = run_command("docker --version")
    return "Docker version" in output

def docker_running():
    output = run_command("sudo docker info")
    return "Containers:" in output  # This is a typical line in the `docker info` output

def start_docker_daemon():
    print("Starting Docker daemon...")
    run_command("sudo systemctl start docker")

def image_exists(image_name):
    images = run_command("sudo docker images -q {}".format(image_name))
    return bool(images)

def container_exists(container_name):
    containers = run_command("sudo docker ps -a -q -f name={}".format(container_name))
    return bool(containers)

def container_running(container_name):
    running_containers = run_command("sudo docker ps -q -f name={}".format(container_name))
    return bool(running_containers)

def start_docker_qdrant():
    image_name = "qdrant/qdrant"
    container_name = "qdrant_container"

    if not docker_installed():
        print("Docker is not installed. Please install Docker to proceed.")
        # You can provide instructions or a script to install Docker here if desired.
        return

    if not docker_running():
        print("Docker daemon is not running. Attempting to start Docker...")
        start_docker_daemon()

        if not docker_running():
            print("Failed to start Docker. Please start Docker manually and try again.")
            return

    if not image_exists(image_name):
        print("Image does not exist. Pulling image...")
        run_command("sudo docker pull {}".format(image_name))

    if not container_exists(container_name):
        print("Container does not exist. Creating container...")
        run_command("sudo docker run -d --name {} -p 6333:6333 -v ~/.cache/qdrant_data:/app/data {}".format(container_name, image_name))
    elif not container_running(container_name):
        print("Container is not running. Starting container...")
        run_command("sudo docker start {}".format(container_name))
    else:
        print("Container is already running.")



def connect_to_qdrant(host="localhost", port=6333):
    client = QdrantClient(host=host, port=port)
    return client

def create_collection(client, collection_name="mnist", vector_size=512, distance=Distance.COSINE):
    # Check if the collection already exists
    try:
        client.get_collection(collection_name=collection_name)
        print("Collection already exists")
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

def batch_insert_embeddings_nooverwrite(client, embeddings, labels, batch_size=50, collection_name="mnist"):
    total_len = len(embeddings)

    for i in range(0, total_len, batch_size):
        start = i
        end = min(i + batch_size, total_len)

        batch_embeddings = embeddings[start:end]
        batch_labels = labels[start:end]
        # batch_img_paths = img_paths[start:end]

        # Create a list of search requests for the current batch
        search_queries = [SearchRequest(
            vector=vec.tolist(),
            limit=1,
            score_threshold=0.9999) for vec in batch_embeddings]

        # Search for the nearest neighbors
        batch_search_results = client.search_batch(
            collection_name=collection_name,
            requests=search_queries
        )

        # Get the indices of the embeddings that are not in the collection
        new_ix = [j for j, res in enumerate(batch_search_results) if len(res) == 0]
        if len(new_ix) == 0:
            print(f"No new embeddings to insert for batch {start}-{end}")
            continue
        new_embeddings = batch_embeddings[new_ix]
        new_labels = batch_labels[new_ix]
        # new_img_paths = [batch_img_paths[j] for j in new_ix]

        # Get the collection size to determine the starting index for new embeddings
        collection_info = client.get_collection(collection_name=collection_name)
        collection_size = collection_info.vectors_count

        # Insert the new embeddings into the collection
        client.upsert(collection_name=collection_name,
                      points=[PointStruct(
                          id=collection_size + k + 1,  # Ensure IDs start from 1, not 0
                          vector=new_embeddings[k].tolist(),
                          payload={"label": new_labels[k].item()}#, "image_path": new_img_paths[k]}
                      ) for k in range(len(new_embeddings))])
        
        # client.upload_collection(collection_name=collection_name,
        #                          vectors=new_embeddings.tolist(),
        #                          payload=[{"label": new_labels[k].item()} for k in range(len(new_embeddings))],
        #                          ids=None,
        #                          parallel=2)

        print(f"Inserted {len(new_embeddings)} new embeddings for batch {start}-{end}")


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
