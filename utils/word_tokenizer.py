# The symbols to be consider as word delimiters in a text.
WORD_DELIMITERS = " "
# The symbol with which each word delimiter symbol should be replaced.
TARGET_SYMBOL = "✂"

# =================================================================================================


class WordTokenizer:
    """
    A word tokenizer that can be used to split a given text into words.
    """

    def __init__(self, word_delimiters=WORD_DELIMITERS):
        """
        Creates a new word tokenizer.

        Args:
            word_delimiters (str or list):
                The symbols to be considered as word delimiters.
        """
        self.word_delimiters = word_delimiters
        self.word_delimiters_map = str.maketrans({c: TARGET_SYMBOL for c in self.word_delimiters})

    def tokenize_into_words(self, text):
        """
        Tokenizes the given text into words using the given word delimiters. Returns a list of
        words.

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
        # Translate each symbol to be considered as a word delimiter to WORD_DELIM.
        text = text.translate(self.word_delimiters_map)
        return [w for w in text.split(TARGET_SYMBOL) if w]
