from enum import Enum

class Platform(Enum):
    """ 
    Represents existing Luxonis hardware platforms. 
    """
    HAILO = "hailo"
    RVC2 = "rvc2"
    RVC3 = "rvc3"
    RVC4 = "rvc4"