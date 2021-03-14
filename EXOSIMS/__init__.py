name = "EXOSIMS"
__version__ = '2.2.0'
# Set up a default logging handler to avoid "No handler found" warnings.
# Other handlers can add to this one.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
