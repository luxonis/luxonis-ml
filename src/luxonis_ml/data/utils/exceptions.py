class DataTransactionException(Exception):

    """
    Exception for the _add_filer function, which provides
    a specific reason along with a more general message
    """

    def __init__(self, filepath, exception_type, reason):
        message = f"Creating a transaction for filepath {filepath} failed with {exception_type}: {reason}"
        super().__init__(message)


class AdditionsStructureError(Exception):

    """Excpetion if incorrect format of additions"""

    pass


class AdditionNotFoundError(Exception):

    """Excpetion for not finding filepath"""

    pass


class ClassificationFormatError(Exception):
    """Exception for wrong classification format being passed in additions"""

    pass


class BoundingBoxFormatError(Exception):
    """Exception for wrong bounding box format being passed in additions"""

    pass


class SegmentationFormatError(Exception):
    """Exception for wrong segmentation mask format being passed in additions"""

    pass


class KeypointFormatError(Exception):
    """Exception for wrong segmentation mask format being passed in additions"""

    pass


class ClassNotFoundError(Exception):
    """Exception for classes not found in dataset"""

    pass
