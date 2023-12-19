from ..guard_extras import guard_missing_extra

with guard_missing_extra("embedd"):
    from .methods import *
    from .utils import *
