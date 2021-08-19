"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains some custom
types that can be used in the "type" attribute of an argparse.add_argument() method.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse
from typing import Union

# ==================================================================================================


def num(string: str) -> Union[int, float]:
    """
    This method translates the given string to an int or a float, depending which type is actually
    represented by the string.

    Args:
        string: str
            The string to translate.
    Returns:
        Union[int, float]
            The string translated to an int or a float, depending which type is actually
            represented by the string.

    >>> num("0")
    0
    >>> num("11")
    11
    >>> num("1.1")
    1.1
    >>> num(".3")
    0.3
    """
    try:
        return int(string)
    except ValueError:
        return float(string)


def boolean(string: str) -> bool:
    """
    This method translates the given string to a boolean value. Raises a ArgumentTypeError if the
    string can't be translated to a boolean value.

    Args:
        string: str
            The string to translate.
    Returns:
        bool
            The string translated to a boolean value.

    >>> boolean("True")
    True
    >>> boolean("true")
    True
    >>> boolean("1")
    True
    >>> boolean("False")
    False
    >>> boolean("false")
    False
    >>> boolean("0")
    False
    >>> boolean("X")
    Traceback (most recent call last):
      ...
    argparse.ArgumentTypeError: Boolean value expected.
    """
    if string.lower() in {"yes", "true", "1"}:
        return True
    elif string.lower() in {"no", "false", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def escaped_char_sequence(string: str) -> str:
    """
    This method unescapes the escaped control characters in the given string. For example,
    translates "\\t" to an actual tab character and "\\n" to an actual newline character.

    Args:
        string: str
            The string to translate.
    Returns:
        str
            The string with all escaped control characters unescaped.

    >>> escaped_char_sequence(None) is None
    True
    >>> escaped_char_sequence("") == ''
    True
    >>> escaped_char_sequence("abc") == 'abc'
    True
    >>> escaped_char_sequence("ab\\tcd")
    'ab\\tcd'
    >>> escaped_char_sequence("ab\\ncd")
    'ab\\ncd'
    """
    if string is None:
        return None

    try:
        return bytes(string, "utf-8").decode("unicode_escape")
    except Exception:
        raise argparse.ArgumentTypeError("(Escaped) char sequence expected.")
