import numpy as np
import hashlib

def get_granule(filepath, addition, component_name):
    granule = filepath.split('/')[-1]
    if 'new_image_name' in addition[component_name].keys():
        filepath = filepath.replace(granule, addition[component_name]['new_image_name'])
        granule = addition[component_name]['new_image_name']
    return granule

def check_media(dataset, filepath, mount_path, component_name, granule, from_bucket):
    source = dataset.source
    if dataset.bucket_type == 'local':
        local_path = f'{dataset.path}/{mount_path}'
        md5 = hashlib.md5()
        with open(filepath, 'rb') as file:
            for chunk in iter(lambda: file.read(8192), b''):
                md5.update(chunk)
        old_checksum = md5.hexdigest()
    elif dataset.bucket_type == 'aws':
        prefix = f'{dataset.bucket_path}/{component_name}/{granule}'
        resp = dataset.client.list_objects(Bucket=dataset.bucket, Prefix=prefix, Delimiter='/', MaxKeys=1)
        exists = 'Contents' in resp
        if not exists:
            raise Exception("Issue with dataset! Sample file does not exist on S3!")
        old_checksum = resp['Contents'][0]['ETag'].strip('"')

    if from_bucket:
        if dataset.bucket_type == 'aws':
            resp = dataset.client.list_objects(Bucket=dataset.bucket, Prefix=filepath, Delimiter='/', MaxKeys=1)
            exists = 'Contents' in resp
            if not exists:
                raise Exception(f"File {filepath} not found on S3!")
            new_checksum = resp['Contents'][0]['ETag'].strip('"')
    else:
        md5 = hashlib.md5()
        with open(filepath, 'rb') as file:
            for chunk in iter(lambda: file.read(8192), b''):
                md5.update(chunk)
        new_checksum = md5.hexdigest()
    if new_checksum != old_checksum:
        if dataset.bucket_type == 'local':
            new_path = f'{dataset.path}/{dataset.team}/datasets/{dataset.name}/archive/{source.name}/{component_name}/{granule}'
            os.makedirs(new_path, exist_ok=True)
            cmd = f'cp {local_path} {new_path}'
            subprocess.check_output(cmd, shell=True)
        elif dataset.bucket_type == 'aws':
            new_prefix = f'{dataset.bucket_path}/archive/{source.name}/{component_name}/{granule}'
            dataset.client.copy_object(
                Bucket=dataset.bucket,
                Key=new_prefix,
                CopySource={'Bucket': dataset.bucket, 'Key': prefix}
            )
        # update existing datasat sample to point to this new_prefix
        # limitation: all old versions point to the latest archive only (TODO: fix)
        for sample in sample_view:
            sample.filepath = mount_path.replace(
                f'/{dataset.team}/{dataset.name}/',
                f'/{dataset.team}/{dataset.name}/archive/'
            )
            sample.save()

        return False
    else:
        return True

def check_classification(val1, val2):
    val1 = fo.Classification(label=val1).to_dict()
    if len(val1.keys()) == len(val2.keys()):
        for key in val1:
            if not key.startswith('_'):
                if val1[key] != val2[key]:
                    return False
    else:
        return False
    return True

def check_boxes(dataset, val1, val2):
    if len(val1) == len(val2['detections']):
        for val1, val2 in list(zip(val1, val2['detections'])):
            if isinstance(val1[0], str) and val2['label'] != val1[0]:
                return False
            if not isinstance(val1[0], str) and val2['label'] != dataset.fo_dataset.classes['boxes'][val1[0]]:
                return False
            for c1, c2 in list(zip(val1[1:], val2['bounding_box'])):
                if abs(c1-c2) > 1e-8:
                    return False
    else:
        return False
    return True

def check_segmentation(val1, val2):
    if (val1.shape != val2.shape) or (np.linalg.norm(val1-val2) > 1e-8):
        return False
    return True

def check_keypoints(val1, val2):
    if len(val1) == len(val2['keypoints']):
        for val1, val2 in list(zip(val1, val2['keypoints'])):
            if isinstance(val1[0], str) and val2['label'] != val1[0]:
                return False
            if not isinstance(val1[0], str) and val2['label'] != dataset.fo_dataset.classes['keypoints'][val1[0]]:
                return False
            for c1, c2 in list(zip(val1[1], val2['points'])):
                if not (np.isnan(c1[0]) and isinstance(c2[0], dict)):
                    if not np.isnan(c1[0]) and not isinstance(c2[0], dict):
                        if c1[0] != c2[0] or c1[1] != c2[1]:
                            return False
                    else:
                        return False

    else:
        return False
    return True

def check_fields(dataset, latest_sample, addition, component_name):

    ignore_fields_match = set([dataset.source.name, 'version', 'metadata', 'latest', 'tags', 'split'])
    ignore_fields_check = set(['filepath'])

    sample_dict = latest_sample.to_dict()
    f1 = set(addition[component_name].keys())
    f2 = set(sample_dict.keys())
    new = f1 - f1.intersection(f2)
    missing = f2 - f1.intersection(f2) - ignore_fields_match
    if len(new) or len(missing):
        return False

    check_fields = list(f1.intersection(f2) - ignore_fields_check)
    for field in check_fields:
        val1 = addition[component_name][field]
        val2 = sample_dict[field]

        if field in dataset.tasks:
            if field == 'class':
                check = check_classification(val1, val2)
                if not check: return False
            elif field == 'boxes':
                check = check_boxes(dataset, val1, val2)
                if not check: return False
            elif field == 'segmentation':
                val2 = latest_sample.segmentation.mask
                check = check_segmentation(val1, val2)
                if not check: return False
            elif field == 'keypoints':
                check = check_keypoints(val1, val2)
                if not check: return False
            else:
                raise NotImplementedError()
        else:
            if not val1 == val2:
                return False

    return True
