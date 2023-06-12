import unittest
import subprocess, shutil, os
from pathlib import Path
import fiftyone.core.odm as foo
from luxonis_ml.ops import *

unittest.TestLoader.sortTestMethodsUsing = None

class LuxonisDatasetTester(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.team = "unittest"
        self.name = "coco"

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
        # could be problematic if .delete() breaks
        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        ))
        if len(res):
            with LuxonisDataset(self.team, self.name) as dataset:
                dataset.delete()

    def test_local_init(self):

        with LuxonisDataset(self.team, self.name) as dataset:
            pass

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['team_name'], 'unittest', "Wrong team name")
        self.assertEqual(res['dataset_name'], 'coco', "Wrong dataset name")
        self.assertEqual(res['current_version'], 0., "Version initialize failure")
        self.assertEqual(res['path'], f'{Path.home()}/.cache/luxonis_ml/data/unittest/datasets/coco', "Dataset path failure")
        self.assertEqual(res['bucket_type'], 'local', "Default bucket type is not local")

        curr = self.conn.datasets.find(
            { "name": f"{self.team}-{self.name}" }
        )
        res2 = list(curr)
        self.assertGreater(len(res2), 0, "Underlying fo dataset not found")
        self.assertEqual(len(res2), 1, "Duplicate underlying fo datasets")
        res2 = res2[0]
        self.assertEqual(res['_dataset_id'], res2['_id'], "Luxonis dataset does not reference fo dataset")

        with LuxonisDataset(self.team, self.name, bucket_type='aws') as dataset:
            dataset.version = 5

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['current_version'], 5., "Update version field fail")
        self.assertEqual(res['bucket_type'], 'local', "Default override_bucket_type arg fail")

    def test_aws_init(self):

        with LuxonisDataset(self.team, self.name, bucket_type='aws', override_bucket_type=True) as dataset:
            pass

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        )
        res = list(curr)
        self.assertGreater(len(res), 0, "Document not created")
        self.assertEqual(len(res), 1, "Multiple documents created")
        res = res[0]
        self.assertEqual(res['bucket_type'], 'aws', "Default override_bucket_type arg fail")

    def test_source(self):

        with LuxonisDataset(self.team, self.name) as dataset:
            dataset.create_source(
                'test_source',
                custom_components = [
                    LDFComponent('A', HType.IMAGE, IType.BGR),
                    LDFComponent('B', HType.IMAGE, IType.DEPTH),
                ]
            )

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
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

        with LuxonisDataset(self.team, self.name) as dataset:
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

    def test_delete(self):

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        )
        res = list(curr)[0]
        old_id = res["_id"]

        with LuxonisDataset(self.team, self.name) as dataset:
            dataset.delete()

        curr = self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        )
        res = list(curr)
        self.assertEqual(len(res), 0, "Document not deleted")
        # TODO: test if source is deleted
        curr = self.conn.luxonis_source_document.find(
            { "_luxonis_dataset_id": old_id }
        )
        res2 = list(curr)
        self.assertEqual(len(res2), 0, "Source document not deleted")

        # TODO: test dataset version documents are deleted

if __name__ == "__main__":

    # test_suite = unittest.TestLoader().loadTestsFromTestCase(LuxonisDatasetTester)
    suite = unittest.TestSuite()
    suite.addTest(LuxonisDatasetTester('test_local_init'))
    suite.addTest(LuxonisDatasetTester('test_aws_init'))
    suite.addTest(LuxonisDatasetTester('test_source'))
    suite.addTest(LuxonisDatasetTester('test_delete'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
