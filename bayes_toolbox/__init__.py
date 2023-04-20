from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayes-toolbox")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass