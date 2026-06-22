import logging
import sys

# Custom level between INFO (20) and WARNING (30): always shown in quiet mode.
METRICS = 25
logging.addLevelName(METRICS, 'METRICS')

def _metrics(self, message, *args, **kwargs):
    if self.isEnabledFor(METRICS):
        self._log(METRICS, message, args, **kwargs)

logging.Logger.metrics = _metrics

LOG_LEVELS: dict[str, int] = {
    'debug':   logging.DEBUG,
    'info':    logging.INFO,
    'quiet':   METRICS,
    'warning': logging.WARNING,
}


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger('dpdl')
    
    # Keep the logger itself at INFO so that the file handler always receives the full INFO stream
    log.setLevel(logging.INFO)

    # create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the new handler
    log.addHandler(handler)

    # Prevent messages from propagating to the root logger, which causes double logging
    log.propagate = False

    return log
