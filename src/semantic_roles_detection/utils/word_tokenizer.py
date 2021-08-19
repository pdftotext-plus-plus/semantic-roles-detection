"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains code to split
text at given delimiters into words.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

# ==================================================================================================
# Parameters.

# The default symbols to be considered as word delimiters in a text.
from typing import List, Union


WORD_DELIMITERS = " "
# The default symbol with which each word delimiter symbol should be replaced.
TARGET_SYMBOL = "✂"

# ==================================================================================================


class WordTokenizer:
    """
    This class is a word tokenizer that can be used to split a given text into words.
    """

    def __init__(self, word_delimiters: Union[str, List[str]] = WORD_DELIMITERS):
        """
        This constructor creates and initializes a new `WordTokenizer`.

        Args:
            word_delimiters Union[str, List[str]]:
                The symbols to be considered as word delimiters.
        """
        self.word_delimiters = word_delimiters
        self.word_delimiters_map = str.maketrans({c: TARGET_SYMBOL for c in self.word_delimiters})

    def tokenize_into_words(self, text: str) -> List[str]:
        """
        This method splits the given text at each occurence of a character in self.word_delimiters
        into words. Returns the list of words.

        Args:
            text: str
                The text to split into words.
        Returns:
            List[str]
                The words into which the text was splitted.

        >>> wt = WordTokenizer()
        >>> wt.tokenize_into_words("foo bar baz.boo")
        ['foo', 'bar', 'baz.boo']
        >>> wt = WordTokenizer(" .")
        >>> wt.tokenize_into_words("foo bar baz.boo")
        ['foo', 'bar', 'baz', 'boo']
        >>> wt = WordTokenizer(".")
        >>> wt.tokenize_into_words("foo bar baz.boo")
        ['foo bar baz', 'boo']
        """
        # Translate each symbol to be considered as a word delimiter to TARGET_SYMBOL.
        text = text.translate(self.word_delimiters_map)
        # Split the text at each occurence of TARGET_SYMBOL.
        return [w for w in text.split(TARGET_SYMBOL) if w]
