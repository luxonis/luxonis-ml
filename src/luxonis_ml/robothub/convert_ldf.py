import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

from luxonis_ml.data.dataset import HType, IType, LDFComponent, LuxonisDataset, BucketStorage

def train_test_val_split(NUM_SAMPLES, train=0.8, val=0.1, test=0.1, seed=42):
    if train + val + test != 1.0:
        raise ValueError("TRAIN + VAL + TEST must equal 1.0")
    
    np.random.seed(seed)
    # generate random indices for train, val, test splits
    indices = np.random.permutation(NUM_SAMPLES)
    train_indices, val_indices, test_indices = indices[:int(train * NUM_SAMPLES)], indices[int(train * NUM_SAMPLES):int((train + val) * NUM_SAMPLES)], indices[int((train + val) * NUM_SAMPLES):]
    train_test_val = np.array(["train"]*NUM_SAMPLES)
    train_test_val[train_indices] = "train"
    train_test_val[val_indices] = "val"
    train_test_val[test_indices] = "test"

    return train_test_val

def convert_to_additions(detections, train_test_val=None):
    additions = []
    labels = []

    for i, detection in enumerate(detections):
        image_path = detection["framePath"]
        # Assuming the class label is in the 'name' key of the 'data' dictionary
        label = detection["data"]["name"]
        
        # If train_test_val is not None, then use it to split the data
        if train_test_val is not None:
            split = train_test_val[i]
        else:
            split = None
        
        additions.append({
            'image': {
                'filepath': image_path,
                'split': split,
                'class': label
            }
        })
        labels.append(label)
    
    return additions, labels

def create_luxonis_dataset(team_id, team_name, dataset_name, additions, classes, BUCKET_STORAGE=BucketStorage.LOCAL):
    # Create a new dataset in LDF
    dataset_id = LuxonisDataset.create(
        team_id=team_id,
        team_name=team_name,
        dataset_name=dataset_name
    )

    print(f"Dataset ID: {dataset_id}")

    # Add the MNIST data to the dataset
    with LuxonisDataset(
        team_id=team_id,
        dataset_id=dataset_id,
        team_name=team_name,
        dataset_name=dataset_name,
        bucket_storage=BUCKET_STORAGE
    ) as dataset:
        dataset.create_source(
            'robothub',
            custom_components=[
                LDFComponent('image', HType.IMAGE, IType.BGR)
            ]
        )
        dataset.set_classes(np.unique(classes).tolist())
        dataset.add(additions)
        dataset.create_version(note="Adding RobotHub data")
    
    return dataset_id

def plot_first_image(dataset):
    for sample in dataset.fo_dataset:
        # img_path = f"{str(Path.home())}/.cache/luxonis_ml/data/{sample.filepath}"
        print(sample)
        id_sub = dataset.path.split('/')[-1]
        img_rel_path = sample.filepath.split(id_sub)[-1]
        img_path = dataset.path + img_rel_path

        img = cv2.imread(img_path)
        plt.imshow(img)
        plt.show()
        break

def detections_to_ldf():
    BUCKET_STORAGE = BucketStorage.LOCAL
    TRAIN,VAL,TEST = 0.8,0.1,0.1

    detections_path = './tmp/2023-09-01_images/detections.json'

    with open(detections_path, 'r') as f:
        detections = json.load(f)

    NUM_SAMPLES = len(detections)

    train_test_val = train_test_val_split(NUM_SAMPLES, train=TRAIN, val=VAL, test=TEST)

    additions, labels = convert_to_additions(detections, train_test_val)

    # Load the Data into LDF
    team_id = "luxonis_id"
    team_name = "luxonis_team"
    dataset_name = "robothub_dataset"

    dataset_id = create_luxonis_dataset(team_id, team_name, dataset_name, additions, labels, BUCKET_STORAGE=BUCKET_STORAGE)

    with LuxonisDataset(
        team_id=team_id,
        dataset_id=dataset_id
    ) as dataset:
        # Plot the first image
        plot_first_image(dataset)

def main():
    detections_to_ldf()

if __name__ == "__main__":
    main()