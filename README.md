## Luxonis Dataset Format

**NOTICE:** This software is an early alpha version. The API is prone to significant changes.

The Luxonis Dataset Format (LDF) is intended to standardize the way data is stored for ML training for OAK products. We can use LDF with the `luxonis_ml.ops` package. LDF should provide the foundation for tasks such as training, evaluation, querying, analytics, and visualization.

The currently recommended way to use LDF is by adding, modifying, or removing data locally. Then, pushing this data to the cloud through LakeFS using the `LuxonisDatasetArtifact` versioning system.

Additionally, there is handy CLI tool for initial configuration and to make versioning a bit more git-like.

The work on this project is in an MVP state, so it may be missing some critical features or have some issues - please report any feedback!

### Installation and Setup

Currently, the `luxonis_ml` package is in a private repo. To install,

```bash
git clone https://github.com/luxonis/luxonis-ml.git && cd luxonis-ml
pip install -e .
```

Additionally, we will need to install some other software for LakeFS versioning.

#### lakectl

Download the `lakectl` package from [here](https://github.com/treeverse/lakeFS/releases/tag/v0.89.0). Then, download version 0.89.0 for your OS.

After downloading, extract the binaries and move them to your `PATH`

```bash
tar xvfz lakeFS_0.89.0*.tar.gz
sudo mv lakectl lakefs /usr/local/bin
```

#### rclone

Install `rclone` as described [here](https://rclone.org/install/). So far, version 1.61.1 has been tested.

One easy way for Debian is to download the `.deb` file from [here](https://rclone.org/downloads/) and run

```bash
sudo dpkg -i rclone-v1.61.1*.deb
```

Additionally, we need to setup up rclone with the following commands. Feel free to reference [this](https://docs.lakefs.io/v0.52/integrations/rclone.html) as well.

First, run `rclone config file`. You will likely see something like `/home/[user]/.config/rclone/rclone.conf`. Ensure this path is correct in the below command.

```bash
cat <<EOT >> /home/[user]/.config/rclone/rclone.conf
[lakefs]
type = s3
provider = AWS
env_auth = false
access_key_id = xxxxxxxxxxxxxxxxx
secret_access_key = xxxxxxxxxxxxxxxxx
endpoint = xxxxxxxxxxxxxxxxx
no_check_bucket = true
EOT
```

Here, `access_key_id`, `secret_access_key`, and `endpoint` are the LakeFS Access Key and LakeFS Secret Access Key, and LakeFS Endpoint respectively.

#### b2

The `b2` pip package will be automatically installed. However, there is some additional setup to
1. Use the `b2` command line tool
2. Add a `b2` bucket to `rclone`

To configure the `b2` command line tool, you must run
```bash
b2 authorize-account [AWS_ACCESS_KEY_ID] [AWS_SECRET_ACCESS_KEY]
```

To add a `b2` remote store to `rclone`, run
```bash
cat <<EOT >> /home/[user]/.config/rclone/rclone.conf
[b2]
type = b2
account = xxxxxxxxxxxxxxxxx
key = xxxxxxxxxxxxxxxxx
hard_delete = true
EOT
```
where `account` is your AWS_ACCESS_KEY_ID and `key` is AWS_SECRET_ACCESS_KEY. More information on this process can be found [here](https://rclone.org/b2/).

#### s3cmd

If you wish to stream data from S3 using `LuxonisLoader`, `s3cmd` is needed. It can be installed [here](https://s3tools.org/s3cmd).

To configure `s3cmd`, run `s3cmd --configure` after installation.

#### luxonis_ml config

After installing the `luxonis_ml` package with `pip`, you should be able to run

```bash
luxonis_ml config
```

It will ask you to enter the following variables:

```text
AWS Access Key: xxxxxxxxxxxxxxxxx
AWS Secret Access Key: xxxxxxxxxxxxxxxxx
AWS Endpoint URL: (optional)
LakeFS Access Key: xxxxxxxxxxxxxxxxx
LakeFS Secret Access Key: xxxxxxxxxxxxxxxxx
LakeFS Endpoint URL: xxxxxxxxxxxxxxxxx
```

### LDF Structure

When first creating a dataset, we want to make sure we have an associated LakeFS repo initialized. There is currently no way to do this with `luxonis_ml`. To create the dataset, we want to create a local directory with the same name as the LakeFS repo. Then, we will initialize the dataset by giving it the lakefs repo and an S3 bucket we might want to clone data to in the case of streaming.

```bash
mkdir repo && cd repo
luxonis_ml dataset init --lakefs_repo repo --s3_path s3://bucket/prefix/path
```

Whenever initializing a dataset in Python, we should use `with` statements, since the cache relies on Python context manager. Let's initialize the local LDF repository we just made in Python.

```python
from luxonis_ml.ops import *

with LuxonisDataset("repo") as dataset:
    print(dataset.name)
```

#### Sources

Sources abstract the source of some segment of data.

For example, we may have data from real hardware, which can be a standard product like OAK-D. Then, we might have the COCO dataset. We can differentiate these with two different sources, especially if the kinds of data from each look different.

In another example, we may have phase 1 hardware and then a phase 2 hardware that adds another sensor. By creating a new source for phase 2, we can easily handle the differences between these sources in the dataloader.

```python
with LuxonisDataset("repo") as dataset:

    # this uses a default configuration which can be used for
    # any OAK-D-like hardware
    # In this example, we literally have OAK-D
    dataset.create_source(oak_d_default=True)
```

Here's a source for COCO

```python
with LuxonisDataset("repo") as dataset:

    # custom source for our COCO 2017
    custom_components = [
        # the first component is main component, which is important for hashing
        LDFComponent(name="image", htype=HType.IMAGE, itype=IType.BGR)
    ]
    dataset.create_source(name="coco", custom_components=custom_components)
```

#### Components

In the code above, you can see that since COCO is not from a default configuration of Luxonis hardware, we want to create custom components for it to define the structure of the source. In the COCO example, it's simple. We don't need to define any components for annotations, we just want to define them for any images we might have. Obviously, we don't have multi-cam setups in the COCO dataset, so we just define one component called `image` that represents our input BGR image.

The default components for OAK-D looks like this:

```python
components = [
                LDFComponent('left', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('right', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('color', htype=HType.IMAGE, itype=IType.BGR),
                LDFComponent('disparity', htype=HType.IMAGE, itype=IType.DISPARITY),
                LDFComponent('depth', htype=HType.IMAGE, itype=IType.DEPTH)
            ]
```

You could see that this configuration could also be used for any OAK-D-like product. When adding data, you do not need to provide data for every single component.

Some more definitions:

* `HType`\: The type of the component. Right now, there's really only `IMAGE`. But eventually, we will have video and point cloud.
* `IType`\: For an image component, the type of the image. `BGR` is for any 3-channel image, `MONO` is for single-channel images, and `DEPTH` and `DISPARITY` can be uint16 single-channel images for subpixel disparity. The only difference between `DISPARITY` and `DEPTH` is a semantic difference.

Finally, every source should have a single `JSON` component by default. This can contain annotations, predictions, or metadata.

The default structure of the `JSON` is as follows:

```txt
{
    [source_name]:
        annotations [list]: [{
            class: [int, handled by LDF],
            class_name: [string],
            bbox: [COCO bounding box format],
            segmentation: [COCO polygon or COCO RLE format],    
            keypoints: [COCO keypoint format],
            attributes: {
                [custom attribute key]: [custom attribute value]
            }
        }]
    ...
}
```

When adding your own data, you will want to specify `class_name` at a minimum. The other annotations are optional and depend on the prediction task. You can also specify your own keys in the list of `annotations` - you will just need to define a custom map function for `LuxonisLoader` when loading these custom annotations as tensors. Here's some examples of what annotations would look like or different tasks:

* **Classification**\: a single annotation with `class_name` only
* **Multi-Class Classification**\: multiple annotations with `class_name` only
* **Object Detection**\: one or more annotations with `class_name` and `bbox`
* **Semantic Segmentation**\: one annotation per class with `class_name` and `segmentation`, where `segmentation` is likely in the RLE format
* **Instance Segmentation**\: one annotation per instance with `class_name` and `segmentation`, where `segmentation` is likely in the polygon format
* **Keypoint Detection**\: one annotation per instance with `class_name` and `keypoints`
* **OCR**\: one annotation per instance with `class_name`, `bbox`, and maybe a custom annotation called `text`. Or, you can include `text` in the list of `attributes`, as well as other qualifiers used in training about an instance. You will need to write a custom map function to work with `LuxonisLoader`.

```python
with LuxonisDataset("repo") as dataset:
    print([source for source in dataset.sources])
```

This should yield an `oak-d` source and a `coco` source.

#### Adding Data

For COCO and YOLO, it's quite easy to add data with the `from_coco` and `from_yolo` methods.

```python
from luxonis_ml.ops.parsers import from_coco

with LuxonisDataset("repo") as dataset:

    # add COCO 2017 val split
    from_coco(
        dataset,
        source_name="coco",
        image_dir="demo_source/coco/TODO",
        annotation_path="demo_source/coco/TODO",
        split="val"
    )
```

We can also write custom scripts to add data from any files with the `add_data` method. For example, for OAK-D, we may just have some images saved to disk we want to add, as well as a calibration file.

```python
import depthai as dai
import glob
import cv2

oak_d_calib_file = "demo_source/oak_d_dataset/oak_d_calib.json"
oak_d_calib = dai.CalibrationHandler(oak_d_calib_file)
oak_d_calib = oak_d_calib.eepromToJson()

with LuxonisDataset("repo") as dataset:

    # add in color camera data and subpixel disparity from OAK-D
    main_files = sorted(glob.glob(f"demo_source/oak_d_dataset/*.rgb.png"))
    for color_path in main_files:
        disp_path = color_path.replace('.rgb.png', '.disp.png')
        color = cv2.imread(color_path)
        disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH) # used to read uint16
        dataset.add_data("oak_d", {
            "color": color,
            "disparity": disp
        }, calibration=oak_d_calib)
```

#### Removing Data

You'll notice that after adding data to the LDF, there will be files stored with a hashed basename, such as `b9bbdedc4fdf61878ec2d6493d47dac1f6a505f0fc7b93ea67e31f2155357842`. We can either manually remove some training examples by their basename, or we can remove an entire source.

Example of removing a specific training example:

```python
with LuxonisDataset("repo") as dataset:

    dataset.remove_data(
        ids=[
            'b9bbdedc4fdf61878ec2d6493d47dac1f6a505f0fc7b93ea67e31f2155357842'
        ]
    )
```

Example of removing an entire source:

```python
with LuxonisDataset("repo") as dataset:

    dataset.remove_source(
        source_name="oak_d"
    )
```

### Dataset Versioning Structure

Now, we have some additional structure for how datasets will be versioned in LakeFS. We have abstracted LakeFS to be very git-like, but we are also making a distinction with two types of branches: *boughs* and regular *branches*.

A [bough](https://www.dictionary.com/browse/bough) just means a larger branch. All branches stem from boughs and ultimately the user works with branches. However, you can branch of different boughs based on which bough you want.

Existing `Bough` types:

* `PROCESSED`\: This is the default bough and is equivalent to the `main` branch in git. The data here is either getting ready for training, but you can also add to this branch while awaiting annotations and add the annotations in later.
* `RAW`\: This can be used for raw collected data **IF** the images need some future processing before it is in a format ready for training. The primary example of this would be distorted data before undistortion.
* `WEBDATASET`\: This is truly the data ready for training, as it is in a format ready for the `LuxonisLoader`. The reason we have a separate bough for this is because it might make the most sense to only clone this for training in Paperspace, either locally or in an S3 bucket to stream the data.

In LakeFS, both boughs and branches will appear as branches.

#### luxonis_ml dataset checkout

Now, before you can do the equivalent of a `git add`/`git commit`/`git push`, you need to checkout a branch. There is a Python API for all of these versioning functions, but the recommended way is to use the CLI to be git-like.

This will checkout a branch called `_processed_stage` to stage some data we are hoping to add.

```bash
luxonis_ml dataset checkout -b _processed_stage
```

Note that using `-b processed_stage` and `-b stage` will also checkout `_processed_stage`. Thus, the word between the first 2 `_` characters will *always* specify the corresponding bough.

If you want to switch boughs, you must explicitly specify the other bough, such as

```bash
luxonis_ml dataset checkout -b _webdataset_stage
```

Finally, if you have a specific commit you want to checkout, this can also be done with

```bash
luxonis_ml dataset checkout -c dd294d6473ea54f0bdb273be4eb08c4df5366c95cf2dbe951afa44b3f9be80ef
```

#### luxonis_ml dataset status

This will tell you the branch and commit you are on. Additionally, it will tell you the number of files added or removed.

```bash
luxonis_ml dataset status
```

#### luxonis_ml dataset push

This will push the data you have added or removed. You will likely get an error if there are no changes.

This function also include a `git commit` equivalent, so `git commit` and `git push` are packaged together. To give a commit message, use

```bash
luxonis_ml dataset push -m "my first commit!"
```

#### luxonis_ml dataset pull

If you want to pull data, make sure you checkout a branch or commit first, then run

```bash
luxonis_ml dataset pull
```

If you include the `--pull_to_s3` flag, it will pull the data to the S3 bucket you specified in `luxonis_ml dataset init` instead of locally. This can be used for streaming.

```bash
luxonis_ml dataset pull --pull_to_s3
```

### Dataloading

Finally, now we can take our COCO val data and wrap it in a `LuxonisLoader`. But first, we need to convert it to a webdataset format, which is as simple as this:

```python
with LuxonisDataset(local_path="demo") as dataset:

    dataset.to_webdataset()
```

Now, `LuxonisLoader` can automatically decode some annotations into tensor formats. Even with multiple sources, LDF automatically tracks all of your classes. You can see that like so

```python
with LuxonisDataset(local_path=DIR) as dataset:

    print(dataset.classes)
```

Now, let's create our PyTorch loader, and also give it some user defined torch transforms. In this case, we just want to resize images to 512x320.

```python
from torchvision import transforms as tfs

with LuxonisDataset("repo") as dataset:

    torch_tfs = tfs.Resize((512, 320))

    def resize(data):
        img, classify, bboxes, seg = data
        img = torch_tfs(img)
        seg = torch_tfs(seg)
        return img, classify, bboxes, seg

    loader = LuxonisLoader(dataset, split='val')
    loader.map(loader.auto_preprocess)
    loader.map(resize)

    pytorch_loader = loader.to_pytorch(batch_size=2)

    for data in pytorch_loader:
        print(type(data))
        print([type(d) for d in data])
        print([d.shape for d in data])
        print(data[2])
        break
```

From this, we can see how the loader automatically loads ground truth tensors for classification, object detection, and semantic segmentation.

* `img`/`data[0]` is (batch size x number of channels x height x width)
* `classify`/`data[1]` is (batch size x number of classes)
* `bboxes`/`data[2]` is (number of boxes in batch x 6 [class, x, y, width, height] )
* `seg`/`data[3]` is (number of classes x height x width)

Here's one more example where we can just auto decode to numpy for visualization.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

with LuxonisDataset("repo") as dataset:

    loader = LuxonisLoader(dataset, split='val')
    loader.map(loader.auto_preprocess_numpy)

    # note that without .to_pytorch, this is batch size of 1
    for img, classify, bboxes, seg in loader.webdataset:
        show = np.transpose(img, (1,2,0))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB) # since matplotlib expects RGB
        ih, iw, _ = show.shape

        uniq_cls = []

        for box in bboxes:
            cls, x, y, w, h = box
            cls = int(cls)
            if cls not in uniq_cls:
                uniq_cls.append(cls)
            x, y, w, h = x*iw, y*ih, w*iw, h*ih
            cv2.rectangle(show, (round(x), round(y)), (round(x+w), round(y+h)), (255,0,0), 2)
            cv2.putText(show, dataset.classes[cls], (round(x), round(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        plt.imshow(show)
        plt.show()

        for cls in uniq_cls:
            cls_seg = seg[cls]
            plt.imshow(cls_seg)
            plt.show()

        break
```

### Known Bugs

* There may be some issues with multiple processes working with the same local LDF filesystem simultaneously

### Licenses

* `lakefs`: [license](https://github.com/treeverse/lakeFS/blob/master/LICENSE)
* `rclone`: [license](https://github.com/rclone/rclone/blob/master/COPYING)
* `b2`: [license](https://github.com/Backblaze/B2_Command_Line_Tool/blob/master/LICENSE)
