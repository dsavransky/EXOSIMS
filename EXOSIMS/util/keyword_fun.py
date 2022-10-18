import inspect
from typing import List


def get_all_args(mod: type) -> List[str]:
    """Return list of all arguments to inits of every base of input class

    Args:
        mod (type):
            Class of object of interest

    Returns
        list:
            List of all arguments to mod.__init__()

    """

    kws = []
    bases = inspect.getmro(mod)
    for b in bases:
        if (b != object) and hasattr(b, "__init__"):
            kws += inspect.getfullargspec(b.__init__).args

    return kws
