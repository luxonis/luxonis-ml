import datetime
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

from luxonis_ml.data.dataset import HType, IType, LDFComponent, LuxonisDataset, BucketStorage
from luxonis_ml.robothub.config_rh import RHConfig

class LDF_Converter:

    def __init__(self, rh_config, tmp_path):
        self.cfg = rh_config.get_config()
        self.tmp_path = tmp_path
        self.TRAIN, self.VAL, self.TEST = 0.8,0.1,0.1
    
    def set_train_test_val(self, train, val, test):
        self.TRAIN, self.VAL, self.TEST = train, val, test

    def train_test_val_split(self, NUM_SAMPLES, seed=None):
        if self.TRAIN + self.VAL + self.TEST != 1.0:
            raise ValueError("TRAIN + VAL + TEST must equal 1.0")
        
        np.random.seed(seed)
        # generate random indices for train, val, test splits
        indices = np.random.permutation(NUM_SAMPLES)

        train_indices = indices[:int(self.TRAIN * NUM_SAMPLES)]
        val_indices = indices[int(self.TRAIN * NUM_SAMPLES):int((self.TRAIN + self.VAL) * NUM_SAMPLES)]
        test_indices = indices[int((self.TRAIN + self.VAL) * NUM_SAMPLES):]

        train_test_val = np.array(["train"]*NUM_SAMPLES)
        train_test_val[train_indices] = "train"
        train_test_val[val_indices] = "val"
        train_test_val[test_indices] = "test"

        return train_test_val

    def convert_to_additions(self, detections, train_test_val=None):
        additions = []
        labels = []

        for i, detection in enumerate(detections):
            id_ = detection["id"] # This is the detection ID
            tags = detection["tags"]
            image_path = detection["framePath"]
            # Assuming the class label is in the 'name' key of the 'data' dictionary
            if self.cfg.get("include_preds", True):
                label = detection["data"]["name"]
            else:
                label = None
            
            # If train_test_val is not None, then use it to split the data
            if train_test_val is not None:
                split = train_test_val[i]
            else:
                split = None
            
            additions.append({
                'image': {
                    'filepath': image_path,
                    'split': split,
                    'class': label,
                    'robothub_id': id_,
                    'robothub_tags': tags
                }
            })
            labels.append(label)
        
        return additions, labels
    
    def decode_bucket(self, bucket):
        if bucket == "s3":
            return BucketStorage.S3
        elif bucket == "gcs":
            return BucketStorage.GCS
        elif bucket == "azure":
            return BucketStorage.AZURE_BLOB
        else:
            return BucketStorage.LOCAL

    def create_luxonis_dataset(self, additions, classes):
        # Get the team ID and name and bucket storage
        target_dataset = self.cfg.get("target_dataset")
        team_id = target_dataset.get("team_id")
        team_name = target_dataset.get("team_name")
        dataset_name = target_dataset.get("dataset_name")
        bucket_storage = target_dataset.get("bucket_storage")
        
        bucket_storage = self.decode_bucket(bucket_storage)

        # Create a new dataset in LDF
        dataset_id = LuxonisDataset.create(
            team_id=team_id,
            team_name=team_name,
            dataset_name=dataset_name
        )

        print(f"Created new dataset! \nDataset ID: {dataset_id}")

        # Add the data to the dataset
        with LuxonisDataset(
            team_id=team_id,
            dataset_id=dataset_id,
            team_name=team_name,
            dataset_name=dataset_name,
            bucket_storage=bucket_storage
        ) as dataset:
            dataset.create_source(
                'robothub',
                custom_components=[
                    LDFComponent('image', HType.IMAGE, IType.BGR)
                ]
            )
            dataset.set_classes(np.unique(classes).tolist())
            dataset.add(additions)
            dataset.create_version(note="Adding initial RobotHub data")
        
        return dataset_id
    
    def fill_luxonis_dataset(self, additions, classes):
        target_dataset = self.cfg.get("target_dataset")
        team_id = target_dataset.get("team_id")
        dataset_id = target_dataset.get("dataset_id")
        bucket_storage = target_dataset.get("bucket_storage")

        bucket_storage = self.decode_bucket(bucket_storage)

        print(f"Adding to existing dataset! \nDataset ID: {dataset_id}")
        
        # Add the data to the dataset
        with LuxonisDataset(
            team_id=team_id,
            dataset_id=dataset_id,
            bucket_storage=bucket_storage
        ) as dataset:
            classes_prev = dataset.get_classes()[0]
            classes = np.unique(classes + classes_prev).tolist()
            dataset.set_classes(classes)
            
            dataset.add(additions)
            dataset.create_version(note="Adding RobotHub data " + str(datetime.datetime.now()))

    def plot_first_image(self, dataset):
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

    def detections_to_ldf(self):
        detections_path = self.tmp_path + '/detections.json'

        with open(detections_path, 'r') as f:
            detections = json.load(f)
        
        NUM_SAMPLES = len(detections)

        train_test_val = self.train_test_val_split(NUM_SAMPLES)

        additions, labels = self.convert_to_additions(detections, train_test_val)

        if self.cfg.get("target_dataset").get("dataset_id") is not None:
            self.fill_luxonis_dataset(additions, labels)
        else:
            dataset_id = self.create_luxonis_dataset(additions, labels)

        # with LuxonisDataset(
        #     team_id=team_id,
        #     dataset_id=dataset_id
        # ) as dataset:
        #     # Plot the first image
        #     self.plot_first_image(dataset)
