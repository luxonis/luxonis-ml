from enum import Enum

class PreprocessingBlockType(Enum): # required only for RVC4

    """ block used in preprocessing """

    MEAN = "mean" # ["mean_b","mean_g","mean_r"]
    SCALE = "scale" # ["scale_b","scale_g","scale_r"]
    REVERSE_CHANNELS = "reverse_channels" #bgr<->rgb
    INTERLEAVED_TO_PLANAR = "interleaved_to_planar" # TODO: change naming?