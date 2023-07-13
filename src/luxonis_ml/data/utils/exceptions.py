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


class BoundingBoxFormatError(Exception):
    """Exception for wrong bounding box format being passed in additions"""

    pass
