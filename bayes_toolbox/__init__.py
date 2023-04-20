from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bayes_toolbox")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass