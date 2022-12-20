"""
Various utility methods
"""

from typing import Dict, Any
import hashlib


def dictToSortedStr(indict: Dict[str, Any]) -> str:
    """Utility method for generating a string representation of a dict
    with keys sorted alphabetically

    Args:
        indict (dict):
            Dictionary to stringify

    Returns:
        str:
            Dictionary contents as string with keys sorted alphabetically
    """

    # first sort
    sortedlist = sorted(indict.items(), key=lambda item: item[0])

    # now assemble output
    outlist = [f"{t[0]}:{str(t[1])}" for t in sortedlist]

    return ",".join(outlist)


def genHexStr(instr: str) -> str:
    """Utility method generating an md5 hash from any input string

    Args:
        instr (str):
            Input string to hashify

    Returns:
        str:
            hash
    """

    return hashlib.md5(instr.encode("utf-8")).hexdigest()
