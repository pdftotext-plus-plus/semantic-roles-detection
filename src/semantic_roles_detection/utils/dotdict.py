"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains the
implementation of a dictionary that enables dot.notation access to dictionary attributes.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

# ==================================================================================================


class DotDict(dict):
    """
    This class is a dictionary that enables dot.notation access to dictionary attributes. For
    example, if dict = {"foo": 1, "bar": 2}, the "foo" attribute can be accessed via "dict['foo']"
    *and* "dict.foo".
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
