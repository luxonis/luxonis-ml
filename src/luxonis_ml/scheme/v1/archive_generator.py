from config import Config
from io import BytesIO
import tarfile
import os
import json
from typing import List

class ArchiveGenerator:
    """ class for constructing the archive .tar file containing executables, metadata, and all information required for decoding """
 
    def __init__(
        self,
        archive_name: str,
        save_path: str,
        cfg_dict: dict,
        executables_paths: List[str],
        compress: bool = False,
        ):
        
        """
        - archive_name: desired name for .tar file
        - save_path: path to where we want the .tar file to be saved
        - cfg_dict: configuration dict
        - executables_paths: paths to executables aka. models (e.g. .dlc model for rvc4 platform)
        - compress: if True, .tar file is compressed with .gz compression
        """
        
        self.archive_name = archive_name if archive_name.endswith(".tar") else f"{archive_name}.tar"
        self.mode = "w"
        if compress:
            self.archive_name += ".gz"
            self.mode = "w:gz"

        self.save_path = save_path
        self.executables_paths = executables_paths

        self.cfg = Config( # pydantic config check
            metadata = cfg_dict["metadata"],
            inputs = cfg_dict["inputs"],
            outputs = cfg_dict["outputs"],
            heads = cfg_dict["heads"],
        )
        
    def make_archive(self):
        
        # create config JSON in-memory file-like object
        json_data, json_buffer = self.make_json()

        # construct .tar archive
        with tarfile.open(os.path.join(self.save_path, self.archive_name), self.mode) as tar:

            # add executables
            for executable_path in self.executables_paths:
                tar.add(executable_path, arcname=os.path.basename(executable_path))

            # add config JSON
            tarinfo = tarfile.TarInfo(name=f"{self.archive_name}.json")
            tarinfo.size = len(json_data)
            json_buffer.seek(0)  # reset the buffer to the beginning
            tar.addfile(tarinfo, json_buffer)


    def make_json(self):
        
        # read-in config data as dict
        data = json.loads(self.cfg.model_dump_json())

        # create an in-memory file-like object
        json_buffer = BytesIO()
        
        # encode the dictionary as bytes and write it to the in-memory file
        json_data = json.dumps(data, indent=4).encode('utf-8')
        json_buffer.write(json_data)

        return json_data, json_buffer