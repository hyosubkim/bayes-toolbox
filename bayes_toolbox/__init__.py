# file: __init__.py

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayes_toolbox")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError