import logging
import sys


def configure_logger(quiet: bool = False) -> logging.Logger:
    log = logging.getLogger('dpdl')

    # Keep the logger itself at DEBUG so the stream handler can be given any level
    log.setLevel(logging.DEBUG)

    # create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO if quiet else logging.DEBUG)

    # create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the new handler
    log.addHandler(handler)

    # Prevent messages from propagating to the root logger, which causes double logging
    log.propagate = False

    return log
