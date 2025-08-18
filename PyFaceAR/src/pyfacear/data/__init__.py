from importlib.resources import read_text as rt


def read_text(resource: str) -> str:
    """Read the text from a resource found in this directory.

    Parameters
    ----------
    resource: str
        The name of the resource to read.

    Returns
    -------
    str
        The text contents of the resource.
    """
    return rt(__name__, resource)
