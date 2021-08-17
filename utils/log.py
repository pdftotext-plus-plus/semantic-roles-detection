import logging

# The default logging formatter.
FORMATTER = logging.Formatter("%(asctime)s %(filename)s:%(lineno)d  %(levelname)s : %(message)s")
# The default logging level.
LOGGING_LEVEL = logging.DEBUG

# =================================================================================================


def get_logger(name=None):
    """
    Return a logger with the specified name or, if name is None, return a logger which is the root
    logger of the hierarchy.
    """
    logger = logging.getLogger(name)

    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_LEVEL)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)

    return logger


def to_log_level(str):
    """
    Translates the given log level string to the corresponding logging level. Returns None if the
    given string doesn't denote a valid logging level.

    >>> to_log_level("debug") == logging.DEBUG
    True
    >>> to_log_level("DEBUG") == logging.DEBUG
    True
    >>> to_log_level("XXX") == None
    True
    """
    return LOG_LEVELS[str.lower()][0] if str.lower() in LOG_LEVELS else None


def to_keras_verbose_flag(str):
    """
    Translates the given log level string to the corresponding keras verbose flag. Returns 0 if the
    given string doesn't denote a valid logging level.

    >>> to_keras_verbose_flag("debug")
    1
    >>> to_keras_verbose_flag("DEBUG")
    1
    >>> to_keras_verbose_flag("XXX")
    0
    """
    return LOG_LEVELS[str.lower()][1] if str.lower() in LOG_LEVELS else 0


# A mapping of log level strings to the corresponding logging levels of python and tensorflow, and
# the keras verbose flags.
LOG_LEVELS = {
    "fatal": (logging.CRITICAL, 0),
    "error": (logging.ERROR, 2),
    "warn": (logging.WARNING, 2),
    "info": (logging.INFO, 1),
    "debug": (logging.DEBUG, 1)
}
