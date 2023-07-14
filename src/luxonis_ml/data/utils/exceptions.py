class DataTransactionException(Exception):
    """
    Exception for the _add_filer function, which provides
    a specific reason along with a more general message
    """

    def __init__(self, filepath, exception_type, reason):
        message = f"Creating a transaction for filepath {filepath} failed with {exception_type}: {reason}"
        super().__init__(message)


class DataExecutionException(Exception):
    """
    Exception for the _add_execute function, which provides
    a specific reason along with a more general message
    """

    def __init__(self, transaction, exception_type, reason):
        if transaction is None:
            message = f"Executing transaction {transaction} failed with {exception_type}: {reason}"
        else:
            message = f"Executing transaction {transaction['_id']} [{transaction['action']}] failed with {exception_type}: {reason}"
        super().__init__(message)


class AdditionsStructureException(Exception):
    """Excpetion if incorrect format of additions"""

    pass


class AdditionNotFoundException(Exception):
    """Excpetion for not finding filepath"""

    pass


class ClassificationFormatException(Exception):
    """Exception for wrong classification format being passed in additions"""

    pass


class BoundingBoxFormatException(Exception):
    """Exception for wrong bounding box format being passed in additions"""

    pass


class SegmentationFormatException(Exception):
    """Exception for wrong segmentation mask format being passed in additions"""

    pass


class KeypointFormatException(Exception):
    """Exception for wrong segmentation mask format being passed in additions"""

    pass


class ClassUnknownException(Exception):
    """Exception for classes not found in dataset"""

    pass


class TransactionNotFoundException(Exception):
    """Exception not finding a transaction in data execution"""

    pass
