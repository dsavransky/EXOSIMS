import logging

name = "EXOSIMS"
__version__ = "3.7.0alpha"

# Set up a default logging handler to avoid "No handler found" warnings.
# Other handlers can add to this one.
logging.getLogger(__name__).addHandler(logging.NullHandler())
