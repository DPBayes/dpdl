import logging
import sys

def configure_logger() -> logging.Logger:
    log = logging.getLogger('dpdl')
    log.setLevel(logging.INFO)

    # create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the new handler
    log.addHandler(handler)

    # Also configure root/opacus loggers so library INFO logs are visible.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    opacus_log = logging.getLogger('opacus')
    opacus_log.setLevel(logging.INFO)
    opacus_log.propagate = True

    return log
