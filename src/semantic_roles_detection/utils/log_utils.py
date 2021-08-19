"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains some common
methods related to logging.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import logging

# ==================================================================================================
# Parameters.

# The default logging formatter.
FORMATTER = logging.Formatter("%(asctime)s - %(origin)30.50s - %(levelname)-7s : %(message)s")
# The default logging level.
LOGGING_LEVEL = logging.DEBUG

# A mapping of log level names to the corresponding logging levels of python and the corresponding
# keras verbose flags.
LOG_LEVELS = {
    "fatal": (logging.CRITICAL, 0),
    "error": (logging.ERROR, 2),
    "warn": (logging.WARNING, 2),
    "info": (logging.INFO, 1),
    "debug": (logging.DEBUG, 1)
}

# ==================================================================================================


def get_logger(name: str = None) -> logging.Logger:
    """
    This method returns a logger with the specified name or, if name is None, a logger which is the
    root logger of the hierarchy.

    Args:
        name: str
            The name of the logger.
    Returns:
        Logger
            The logger.
    """
    logging.setLogRecordFactory(OriginLogRecord)
    logger = logging.getLogger(name)

    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_LEVEL)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)

    return logger


def to_log_level(level_name: str) -> int:
    """
    This method translates the given log level name to the corresponding logging level. Returns
    None if the given string doesn't denote a valid logging level.

    Args:
        level_name: str
            The name of the logging level.
    Returns:
        int
            The corresponding logging level.

    >>> to_log_level("debug") == logging.DEBUG
    True
    >>> to_log_level("DEBUG") == logging.DEBUG
    True
    >>> to_log_level("XXX") == None
    True
    """
    return LOG_LEVELS[level_name.lower()][0] if level_name.lower() in LOG_LEVELS else None


def to_keras_verbose_flag(level_name: str) -> int:
    """
    This method translates the given log level name to the corresponding keras verbose flag.
    Returns 0 if the given string doesn't denote a valid logging level.

    Args:
        level_name: str
            The name of the logging level.
    Returns:
        int
            The corresponding Keras verbose flag.

    >>> to_keras_verbose_flag("debug")
    1
    >>> to_keras_verbose_flag("DEBUG")
    1
    >>> to_keras_verbose_flag("XXX")
    0
    """
    return LOG_LEVELS[level_name.lower()][1] if level_name.lower() in LOG_LEVELS else 0

# ==================================================================================================


class OriginLogRecord(logging.LogRecord):
    """
    A custom log record that combines the fields "filename" and "lineno" to the single field
    "origin". This enables the option to define a fixed width for this field. Otherwise, we would
    have to define a fixed width for both fields which breaks both fields apart.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = f"{self.filename}:{self.lineno}"
