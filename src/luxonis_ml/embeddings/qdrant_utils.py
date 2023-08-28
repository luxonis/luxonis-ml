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

def get_similarities(qdrant_client, reference_id, other_ids, qdrant_collection_name="mnist"):
    """
    Get similarities of a reference embedding with a list of other embeddings using Qdrant.

    Parameters
    ----------
    qdrant_client : QdrantClient
        The Qdrant client instance to use for searches.
    reference_id : int
        The instance_id of the reference embedding.
    other_ids : list[int]
        The list of instance_ids of other embeddings to compare with the reference.
    qdrant_collection_name : str
        The name of the Qdrant collection. Default is "mnist".

    Returns
    -------
    list
        The list of embeddings similar to the reference embedding, filtered by other_ids.
    """
    # Retrieve the embedding vector for the reference_id
    reference_embedding = get_embeddings_from_ids(qdrant_client, [reference_id], qdrant_collection_name)[0]

    # Search for similar embeddings using the reference embedding
    hits = qdrant_client.search(
        collection_name=qdrant_collection_name,
        query_vector=reference_embedding,
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="instance_id",
                    match=models.MatchText(text=str(id))
                ) for id in other_ids
            ] 
        )
    )

    ids = [hit.id for hit in hits]
    scores = [hit.score for hit in hits]

    return ids, scores

def get_full_similarity_matrix(client, collection_name="mnist", batch_size=100):
    # NOTE: This method is not recommended for large collections.
    #       It is better to use the get_all_embeddings() method and compute the similarity matrix yourself.
    # Get all embeddings
    ids, embeddings = get_all_embeddings(client, collection_name)
    print("Retrieved {} embeddings".format(len(embeddings)))

    # Create a list of search requests
    search_queries = [SearchRequest(
        vector=emb,
        with_payload=False,
        with_vector=False,
        limit=len(embeddings)
        ) for emb in embeddings]
    print("Created {} search queries".format(len(search_queries)))
    
    # Search for the nearest neighbors
    batch_search_results = []
    for patch in range(0, len(search_queries), batch_size):
        batch_search_results_i = client.search_batch(
            collection_name=collection_name,
            requests=search_queries[patch:patch+batch_size]
        )
        batch_search_results.extend(batch_search_results_i)
        print("Completed search for batch {}-{}".format(patch, patch+batch_size))
    
    # Create a dictionary for O(1) lookup of ids
    id_to_index = {id: index for index, id in enumerate(ids)}

    # Get the similarity matrix
    sim_matrix = [[0 for _ in range(len(embeddings))] for _ in range(len(embeddings))]
    for i, res in enumerate(batch_search_results):
        for hit in res:
            j = id_to_index[hit.id]
            sim_matrix[i][j] = hit.score
    
    print("Created similarity matrix")
    return ids, sim_matrix

def get_payloads_from_ids(client, ids, collection_name="mnist"):
    # Retrieve the payloads for the given ids
    hits = client.retrieve(collection_name=collection_name, 
                            ids=ids, 
                            with_payload=True, 
                            with_vectors=False)
    
    # Convert the payloads to a list and order them by the original instance_id
    id_to_index = {id: index for index, id in enumerate(ids)}
    payloads = [None for _ in range(len(ids))]
    for hit in hits:
        i = id_to_index[hit.id]
        payloads[i] = hit.payload

    return payloads

def get_embeddings_from_ids(client, ids, collection_name="mnist"):
    # Retrieve the embeddings for the given ids
    hits = client.retrieve(collection_name=collection_name, 
                            ids=ids, 
                            with_payload=False, 
                            with_vectors=True)
    # embeddings = [hit.vector for hit in hits]
    id_to_index = {id: index for index, id in enumerate(ids)}
    embeddings = [None for _ in range(len(ids))]
    for hit in hits:
        i = id_to_index[hit.id]
        embeddings[i] = hit.vector
    return embeddings

def get_all_ids(client, collection_name="mnist"):
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size)
    ids = [hit.id for hit in hits]
    return ids

def get_all_instance_and_sample_ids(client, collection_name="mnist"):
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size)
    
    instance_ids = [hit.payload['instance_id'] for hit in hits]
    sample_ids = [hit.payload['sample_id'] for hit in hits]
    return instance_ids, sample_ids

def get_all_embeddings(client, collection_name="mnist"):
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size,
                                  with_vectors=True)
    
    ids = [hit.id for hit in hits]
    embeddings = [hit.vector for hit in hits]
    return ids, embeddings