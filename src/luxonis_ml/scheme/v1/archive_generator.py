from config import Config
import tarfile
import os
import json

class ArchiveGenerator:
    """ class for constructing the archive tar file containing executables, metadata, and all information required for decoding """
 
    def __init__(
        self,
        archive_name,
        archive_path,
        cfg_dict,
        onnx_model_path,
        platform_model_path,
        ):
        
        """
        - archive_name: desired .tar file name
        - archive_path: where we want .tar file to be saved
        - cfg_dict: configuration dict
        - onnx_model_path: path to the onnx model
        - platform_model_path: path to the platform model (e.g. .dlc model for rvc4 platform)
            (for now, we are constructiong platform model outside; TODO: construct it within the class)
        """
        
        self.archive_name = archive_name if archive_name.endswith(".tar") else f"{archive_name}.tar"
        self.archive_path = archive_path
        self.onnx_model_path = onnx_model_path
        self.platform_model_path = platform_model_path

        self.cfg_dict = cfg_dict
        self.cfg = Config( # pydantic config check
            metadata = cfg_dict["metadata"],
            inputs = cfg_dict["inputs"],
            outputs = cfg_dict["outputs"],
            heads = cfg_dict["heads"],
        )
        
    def make_archive(self):
        
        # create config json
        json_path = self.make_json()

        # create archive
        with tarfile.open(os.path.join(self.archive_path, self.archive_name), "w") as tar: #"w:gz") as tar:
            tar.add(self.onnx_model_path, arcname=os.path.basename(self.onnx_model_path))
            tar.add(self.platform_model_path, arcname=os.path.basename(self.platform_model_path))
            tar.add(json_path, arcname=os.path.basename(json_path))

        # delete config json? TODO

    def make_json(self):
        json_path = os.path.join(self.archive_path, f"{self.archive_name}.json")
        with open(json_path, "w") as outfile: 
            json.dump(json.loads(self.cfg.model_dump_json()), outfile)
        return json_path
