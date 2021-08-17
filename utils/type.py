import argparse


def num(str):
    """
    Translates the given string to int or float, depending which type is represented by the
    string.

    Args:
        str (str):
            The string to translate.

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
        return int(str)
    except ValueError:
        return float(str)


def boolean(str):
    """
    Translates the given string to a boolean value. Raises a ArgumentTypeError if the string
    can't be translated to a boolean value.

    Args:
        str (str):
            The string to translate.

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
    if str.lower() in {"yes", "true", "1"}:
        return True
    elif str.lower() in {"no", "false", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def escaped_char_sequence(str):
    """
    Unescapes the escaped control characters in the given string. For example, translates "\\t" to
    an actual tab characters and translates "\\n" to an actual newline character.

    Args:
        str (str):
            The string to translate.

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
    if str is None:
        return None

    try:
        return bytes(str, "utf-8").decode("unicode_escape")
    except Exception:
        raise argparse.ArgumentTypeError("(Escaped) char sequence expected.")
