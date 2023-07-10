import unittest
import subprocess, shutil, os, glob, json, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
import fiftyone.core.odm as foo
from luxonis_ml.data import *
from copy import deepcopy
from fiftyone import ViewField as F
from bson.objectid import ObjectId

unittest.TestLoader.sortTestMethodsUsing = None

"""
NOTE: Depsite the name, this is integration testing not unit testing.
The test cases are meant to happen in a certain order to check how
the state of the LDF changes with each step.
"""

class LuxonisDatasetTester(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.team_id = "d7625eef-ad99-4019-af95-ffa5ebd48e3c"
        self.team_name = "unittest"
        self.dataset_name = "coco"

        self.coco_images_path = "../data/person_val2017_subset"
        self.coco_annotation_path = "../data/person_keypoints_val2017.json"
        if os.path.exists(self.coco_images_path):
            shutil.rmtree(self.coco_images_path)
        if os.path.exists(self.coco_annotation_path):
            os.remove(self.coco_annotation_path)

        print("Downloading data from GDrive...")
        cmd = "gdown 1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT -O ../data/COCO_people_subset.zip"
        subprocess.check_output(cmd, shell=True)

        print("Extracting data...")
        cmd = "unzip ../data/COCO_people_subset.zip -d ../data/"
        subprocess.check_output(cmd, shell=True)

        self.conn = foo.get_db_conn()
        # if dataset already exists, delete it
        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"dataset_name": self.dataset_name}] }
        ))
        if len(res):
            with LuxonisDataset(self.team_id, res[0]['_id']) as dataset:
                dataset.delete_dataset()

        # get COCO data for testing
        img_dir = '../data/person_val2017_subset'
        annot_file = '../data/person_keypoints_val2017.json'

        # get paths to images sorted by number
        im_paths = glob.glob(img_dir+'/*.jpg')
        nums = np.array([int(path.split('/')[-1].split('.')[0]) for path in im_paths])
        idxs = np.argsort(nums)
        im_paths = list(np.array(im_paths)[idxs])

        # load
        with open(annot_file) as file:
            data = json.load(file)
        imgs = data['images']
        anns = data['annotations']
        self.categories = data['categories']

        # create some artificial splits
        splits = ['train' for _ in range(20)] + ['val' for _ in range(10)]
        self.additions = [] # list of additional training examples

        for i, path in enumerate(im_paths):
            # find annotations matching the COCO image
            gran = path.split('/')[-1]
            img = [img for img in imgs if img['file_name']==gran][0]
            img_id = img['id']
            img_anns = [ann for ann in anns if ann['image_id'] == img_id]

            # load the image
            im = cv2.imread(path)
            height, width, _ = im.shape

            # initialize annotations for LDF
            mask = np.zeros((height, width)) # segmentation mask is always a HxW numpy array
            boxes = [] # bounding boxes are a list of [class, x, y, width, height] of the box
            keypoints = [] # keypoints are a list of classes and (x,y) points

            # create the additions
            for ann in img_anns:
                # COCO-specific conversion for segmentation
                seg = ann['segmentation']
                if isinstance(seg, list):
                    for s in seg:
                        poly = np.array(s).reshape(-1,2)
                        poly = [(poly[i,0],poly[i,1]) for i in range(len(poly))]
                        m = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(m).polygon(poly, outline=1, fill=1)
                        m = np.array(m)
                        mask[m==1] = 1
                # COCO-specific conversion for bounding boxes
                x, y, w, h = ann['bbox']
                boxes.append(['person', x/width, y/height, w/width, h/height])
                # COCO-specific conversion for keypoints
                kps = np.array(ann['keypoints']).reshape(-1, 3)
                keypoint = []
                for kp in kps:
                    if kp[2] == 0:
                        keypoint.append((float('nan'), float('nan')))
                    else:
                        keypoint.append((kp[0]/width, kp[1]/height))
                keypoints.append(['person', keypoint])
            data = {
                'filepath': path,
                'segmentation': mask,
                'boxes': boxes,
                'keypoints': keypoints,
                'split': splits[i]
            }
            if i % 2 == 0:
                A_dict = data
            else:
                B_dict = data
                self.additions.append({
                    'A': A_dict,
                    'B': B_dict
                })

        self.dataset_id = LuxonisDataset.create(
            self.team_id,
            self.team_name,
            self.dataset_name
        )

        print('Testing', self.dataset_id)

    def test_local_init(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:
            pass

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['team_id'], self.team_id, "Wrong team ID")
        self.assertEqual(res['_id'], ObjectId(self.dataset_id), "Wrong dataset ID")
        self.assertEqual(res['team_name'], 'unittest', "Wrong team name")
        self.assertEqual(res['dataset_name'], 'coco', "Wrong dataset name")
        self.assertEqual(res['current_version'], 0., "Version initialize failure")
        self.assertEqual(res['path'], f'{Path.home()}/.cache/luxonis_ml/data/{self.team_id}/datasets/{self.dataset_id}', "Dataset path failure")
        self.assertEqual(res['bucket_type'], 'local', "Default bucket type is not local")

        curr = self.conn.datasets.find(
            { "name": f"{self.team_id}-{self.dataset_id}" }
        )
        res2 = list(curr)
        self.assertGreater(len(res2), 0, "Underlying fo dataset not found")
        self.assertEqual(len(res2), 1, "Duplicate underlying fo datasets")
        res2 = res2[0]
        self.assertEqual(res['_dataset_id'], res2['_id'], "Luxonis dataset does not reference fo dataset")

        with LuxonisDataset(self.team_id, self.dataset_id, bucket_type='aws') as dataset:
            dataset.version = 5

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['current_version'], 5., "Update version field fail")
        self.assertEqual(res['bucket_type'], 'local', "Default override_bucket_type arg fail")

        with LuxonisDataset(self.team_id, self.dataset_id, bucket_type='aws') as dataset:
            dataset.version = 0 # set back to 0

    def test_aws_init(self):

        with LuxonisDataset(self.team_id, self.dataset_id, bucket_type='aws', override_bucket_type=True) as dataset:
            pass

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['bucket_type'], 'aws', "Default override_bucket_type arg fail")

    def test_source(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:
            dataset.create_source(
                'test_source',
                custom_components = [
                    LDFComponent('A', HType.IMAGE, IType.BGR),
                    LDFComponent('B', HType.IMAGE, IType.DEPTH),
                ]
            )

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)[0]
        curr = self.conn.luxonis_source_document.find(
            { "_luxonis_dataset_id": res["_id"] }
        )
        res2 = list(curr)
        self.assertGreater(len(res2), 0, "Source document not created")
        self.assertEqual(len(res2), 1, "Multiple source documents created")
        res2 = res2[0]
        self.assertEqual(res2['name'], 'test_source', "Wrong source name")
        self.assertEqual(res2['source_type'], 'custom', "Wrong source type")
        self.assertEqual(res2['component_names'], ['A','B'], "Wrong component names")
        self.assertEqual(res2['component_htypes'], [1,1], "Wrong HType")
        self.assertEqual(res2['component_itypes'], [1,4], "Wrong IType")

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:
            dataset.create_source(
                'test_source',
                custom_components = [
                    LDFComponent('A', HType.IMAGE, IType.BGR),
                    LDFComponent('B', HType.IMAGE, IType.BGR),
                ]
            )

        curr = self.conn.luxonis_source_document.find(
            { "_luxonis_dataset_id": res["_id"] }
        )
        res2 = list(curr)[0]
        self.assertEqual(res2['component_itypes'], [1,1], "Wrong IType after changing source")

    def test_transactions(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            dataset.set_classes(['person', 'orange', 'pear']) # needed for classification and detection
            dataset.set_mask_targets({1:'person', 2:'orange', 3: 'pear'}) # needed for segmentation
            dataset.set_skeleton({
                'labels': self.categories[0]['keypoints'],
                'edges': (np.array(self.categories[0]['skeleton'])-1).tolist()
            }) # optional for keypoints to define the skeleton visualization

            result = dataset._check_transactions()
            self.assertEqual(result, None, "Empty transactions fail")

            dataset._make_transaction(LDFTransactionType.ADD)
            result = dataset._check_transactions()
            self.assertEqual(result, None, "Missing END transaction fail")
            time.sleep(0.5)
            curr = self.conn.transaction_document.find(
                { "_dataset_id": dataset.dataset_doc.id }
            )
            res = list(curr)
            self.assertEqual(len(res), 0, "Faulty transaction not deleted")

            dataset._make_transaction(LDFTransactionType.UPDATE)
            dataset._make_transaction(LDFTransactionType.DELETE)
            dataset._make_transaction(LDFTransactionType.END)
            time.sleep(0.5)
            result = dataset._check_transactions()
            self.assertEqual(len(result), 3, "With END transaction fail")

            # cleanup
            self.conn.transaction_document.delete_many({ '_dataset_id': dataset.dataset_doc.id })

    def test_add(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            dataset.create_source(
                'test_source',
                custom_components = [
                    LDFComponent('A', HType.IMAGE, IType.BGR),
                    LDFComponent('B', HType.IMAGE, IType.BGR),
                ]
            )

            transaction_to_additions, media_change, field_change = dataset._add_filter(self.additions)

            self.assertEqual(len(transaction_to_additions), 30, "Wrong number of entries in transaction_to_additions")
            self.assertEqual(media_change, True, "media_change failed")
            self.assertEqual(field_change, False, "field_change failed")

            curr = self.conn.transaction_document.find(
                { "_dataset_id": dataset.dataset_doc.id }
            )
            res = list(curr)
            self.assertEqual(len(res), 31, "Wrong number of saved transactions")
            num_adds = np.sum([True if t['action']=='ADD' else False for t in res])
            num_updates = np.sum([True if t['action']=='UPDATE' else False for t in res])
            num_deletes = np.sum([True if t['action']=='DELETE' else False for t in res])
            num_ends = np.sum([True if t['action']=='END' else False for t in res])
            self.assertEqual(num_adds, 30, "Wrong number of ADDs")
            self.assertEqual(num_updates, 0, "Wrong number of UPDATEs")
            self.assertEqual(num_deletes, 0, "Wrong number of DELETEs")
            self.assertEqual(num_ends, 1, "Wrong number of ENDs")

            # Execution
            dataset._add_execute(self.additions, transaction_to_additions)

            self.assertEqual(len(dataset.fo_dataset), 15, "Not added to dataset")
            curr = self.conn.transaction_document.find(
                { "_dataset_id": dataset.dataset_doc.id }
            )
            res = list(curr)
            num_executed = np.sum([True if t['executed']==True else False for t in res])
            self.assertEqual(num_executed, 31, "Wrong number of executed transactions")

    def test_version_1(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            dataset.create_version(note="test version 1")
            self.assertEqual(dataset.version, 1.0, "Wrong data version incr")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'A'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 1.0:
                    num_version += 1
            self.assertEqual(num_latest, 15, "Not all versioned samples are latest for A")
            self.assertEqual(num_version, 15, "Not all versioned samples have the right version for A")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'B'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 1.0:
                    num_version += 1
            self.assertEqual(num_latest, 15, "Not all versioned samples are latest for B")
            self.assertEqual(num_version, 15, "Not all versioned samples have the right version for B")

            curr = self.conn.version_document.find(
                { "dataset_id_str": dataset.dataset_doc.fo_dataset_id }
            )
            res = list(curr)
            self.assertEqual(len(res), 1, "Number of saved versions")

    def test_modify(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            transaction_to_additions = []
            a = deepcopy(self.additions[-1])
            a['A']['class'] = ['person'] # adding a new field (classification)
            a['A']['weather'] = 'sunny' # adding a new field
            del a['A']['split'] # removing a field
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(media_change, False, "media_change failed")
            self.assertEqual(field_change, True, "field_change failed")

            res = dataset._check_transactions()
            self.assertEqual(len(res), 4, "Correct number of new transactions")
            num_adds = np.sum([True if t['action']=='ADD' else False for t in res])
            num_updates = np.sum([True if t['action']=='UPDATE' else False for t in res])
            num_deletes = np.sum([True if t['action']=='DELETE' else False for t in res])
            num_ends = np.sum([True if t['action']=='END' else False for t in res])
            num_executed = np.sum([True if t['executed']==True else False for t in res])
            self.assertEqual(num_adds, 0, "Wrong number of ADDs")
            self.assertEqual(num_updates, 3, "Wrong number of UPDATEs")
            self.assertEqual(num_deletes, 0, "Wrong number of DELETEs")
            self.assertEqual(num_ends, 1, "Wrong number of ENDs")
            self.assertEqual(num_executed, 0, "Some transactions are unexpectedly executed")

            # non-annotation field change
            a = deepcopy(self.additions[-1])
            a['A']['weather'] = 'stormy'
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Non-annotation field update")
            # classification change
            a = deepcopy(self.additions[-1])
            a['A']['class'] = ['orange', 'pear']
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Classification update")
            # boxes change
            a = deepcopy(self.additions[-1])
            a['A']['boxes'][0][0] = 'orange' # test class change with str
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Boxes update (change class str)")
            a = deepcopy(self.additions[-1])
            a['A']['boxes'][0][0] = 1 # test class change with int
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Boxes update (change class int)")
            a = deepcopy(self.additions[-1])
            a['A']['boxes'][0][1] += 0.1 # test coordinate change
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Boxes update (change coordinate)")
            # segmentation change
            a = deepcopy(self.additions[-1])
            a['A']['segmentation'] = np.zeros(a['A']['segmentation'].shape)
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Segmentation update")
            # keypoints change
            a = deepcopy(self.additions[-1])
            a['A']['keypoints'][0][0] = 'orange' # test class change with str
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Keypoints update (change class str)")
            a = deepcopy(self.additions[-1])
            a['A']['keypoints'][0][0] = 1 # test class change with int
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Keypoints update (change class int)")
            a = deepcopy(self.additions[-1])
            a['A']['keypoints'][0][1][0] = (a['A']['keypoints'][0][1][0][0]+0.1, a['A']['keypoints'][0][1][0][1])
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Keypoints update (change coordinate non-NaN->non-NaN)")
            a = deepcopy(self.additions[-1])
            a['B']['keypoints'][0][1][8] = (a['A']['keypoints'][0][1][0][0]+0.1, a['A']['keypoints'][0][1][0][1])
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Keypoints update (change coordinate NaN->non-Nan)")
            a = deepcopy(self.additions[-1])
            a['A']['keypoints'][0][1][0] = (np.nan, np.nan)
            tta, media_change, field_change = dataset._add_filter([a])
            transaction_to_additions += tta
            self.assertEqual(field_change, True, "Keypoints update (change coordinate non-NaN->NaN)")

            dataset._add_execute()
            dataset.fo_dataset.group_slice = 'A'
            query = dataset.fo_dataset.match(
                (F('latest') == False) & (F('version') == -1)
            )
            self.assertEqual(len(query), 1, "Only one change in A when executing modifications")
            for sample in query:
                break
            self.assertEqual(sample['weather'], 'stormy', "Executed change to added field")
            self.assertEqual(sample['split'], None, "Executed change to added field")
            self.assertEqual(sample['class']['classifications'][0]['label'], 'orange', "Executed change to classification label")
            # TODO: better box test
            self.assertEqual(np.sum(sample['segmentation']['mask']), 0., "Executed change to segmentation mask")
            # TODO: better keypoint test

            dataset.fo_dataset.group_slice = 'B'
            query = dataset.fo_dataset.match(
                (F('latest') == False) & (F('version') == -1)
            )
            self.assertEqual(len(query), 1, "Only one change in B when executing modifications")
            for sample in query:
                break
            # TODO: keypoint test

    def test_version_2(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            dataset.create_version(note="test version 2")
            self.assertEqual(dataset.version, 1.1, "Wrong data version incr")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'A'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 1.1:
                    num_version += 1
            self.assertEqual(num_latest, 15, "Not all versioned samples are latest for A")
            self.assertEqual(num_version, 1, "Not all versioned samples have the right version for A")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'B'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 1.1:
                    num_version += 1
            self.assertEqual(num_latest, 15, "Not all versioned samples are latest for B")
            self.assertEqual(num_version, 1, "Not all versioned samples have the right version for B")

            curr = self.conn.version_document.find(
                { "dataset_id_str": dataset.dataset_doc.fo_dataset_id }
            )
            res = list(curr)
            self.assertEqual(len(res), 2, "Number of saved versions")
            set1 = set(res[0]['samples'])
            set2 = set(res[1]['samples'])
            set1_unique = set1.union(set2) - set2
            set2_unique = set1.union(set2) - set1
            self.assertEqual(len(set1_unique), 2, "One deprecated sample")
            self.assertEqual(len(set2_unique), 2, "One added sample")

    def test_delete(self):
        
        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            sample = dataset.fo_dataset[f"/{self.team_id}/datasets/{self.dataset_id}/A/000000000139.jpg"]
            dataset.delete([sample.id])

            res = dataset._check_transactions(for_versioning=True) # will show the latest 3 transactions
            self.assertEqual(len(res), 3, "Wrong number of transactions for delete")
            num_executed = np.sum([True if t['executed']==True else False for t in res])
            num_adds = np.sum([True if t['action']=='ADD' else False for t in res])
            num_updates = np.sum([True if t['action']=='UPDATE' else False for t in res])
            num_deletes = np.sum([True if t['action']=='DELETE' else False for t in res])
            num_ends = np.sum([True if t['action']=='END' else False for t in res])
            self.assertEqual(num_executed, 3, "Wrong number of executed transactions")
            self.assertEqual(num_adds, 0, "Wrong number of ADD transactions")
            self.assertEqual(num_updates, 0, "Wrong number of UPDATE transactions")
            self.assertEqual(num_deletes, 2, "Wrong number of DELETE transactions")
            self.assertEqual(num_ends, 1, "Wrong number of END transactions")

            dres = [t for t in res if t['action']=='DELETE']
            for t in dres:
                sample = dataset.fo_dataset[t['sample_id']]
                self.assertEqual(sample.latest, False, "Delete latest!=False")

    def test_version_3(self):
            
        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:

            dataset.create_version(note="deleted some samples")
            self.assertEqual(dataset.version, 2.1, "Wrong data version incr")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'A'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 2.1:
                    num_version += 1
            self.assertEqual(num_latest, 14, "Not all versioned samples are latest for A")
            self.assertEqual(num_version, 0, "Not all versioned samples have the right version for A")

            num_latest, num_version = 0, 0
            dataset.fo_dataset.group_slice = 'B'
            for sample in dataset.fo_dataset:
                if sample.latest:
                    num_latest += 1
                if sample.version == 2.1:
                    num_version += 1
            self.assertEqual(num_latest, 14, "Not all versioned samples are latest for B")
            self.assertEqual(num_version, 0, "Not all versioned samples have the right version for B")

            curr = self.conn.version_document.find(
                { "dataset_id_str": dataset.dataset_doc.fo_dataset_id }
            )
            res = list(curr)
            self.assertEqual(len(res), 3, "Number of saved versions")

    def test_delete_dataset(self):

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)[0]
        old_id = res["_id"]

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:
            dataset.delete_dataset()

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        )
        res = list(curr)
        self.assertEqual(len(res), 0, "Document not deleted")

        curr = self.conn.luxonis_source_document.find(
            { "_luxonis_dataset_id": old_id }
        )
        res2 = list(curr)
        self.assertEqual(len(res2), 0, "Source document not deleted")

        # TODO: test dataset version documents are deleted
        # TODO: test dataset transaction documents are deleted

    @classmethod
    def tearDownClass(self):

        with LuxonisDataset(self.team_id, self.dataset_id) as dataset:
            dataset.delete_dataset()


if __name__ == "__main__":

    suite = unittest.TestSuite()
    suite.addTest(LuxonisDatasetTester('test_local_init'))
    suite.addTest(LuxonisDatasetTester('test_aws_init'))
    suite.addTest(LuxonisDatasetTester('test_source'))
    suite.addTest(LuxonisDatasetTester('test_transactions'))
    suite.addTest(LuxonisDatasetTester('test_add'))
    suite.addTest(LuxonisDatasetTester('test_version_1'))
    suite.addTest(LuxonisDatasetTester('test_modify'))
    suite.addTest(LuxonisDatasetTester('test_version_2'))
    suite.addTest(LuxonisDatasetTester('test_delete'))
    suite.addTest(LuxonisDatasetTester('test_version_3'))
    suite.addTest(LuxonisDatasetTester('test_delete_dataset'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
