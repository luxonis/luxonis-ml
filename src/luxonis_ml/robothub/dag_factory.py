import os
import sys

# Get the directory of the current file and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from robothub_dataingest import RobotHubIngest

class DAGFactory():
    """The DAG Factory Class"""

    def __init__(self):
        pass

    def create(self, dag_type, config, repeat):
        supported_dags = {
            'RobotHubIngest': RobotHubIngest
        }
        dag = supported_dags[dag_type](
            config_name=config,
            repeat=repeat
        )
        return dag
