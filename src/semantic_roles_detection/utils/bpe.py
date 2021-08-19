"""
A simple implementation of the byte pair encoding algorithm that can be used to tokenize word
sequences. The algorithm is explained in full detail here: https://arxiv.org/abs/1508.07909.
Basically, the algorithm segments words into tokens (we use the term *token* to denote any
substring of a word; for example a character, a subword, or the entire word and assigns a unique
integer id to each token. This method particularly allows to deal with rare or unknown words (words
that don't occur in the training data set) by trying to reconstruct the meaning of a word from its
parts. For example, the suffix "-est" lets you guess that the word is probably a superlative
adjective ("lowest", "highest", "largest", etc.)
There are two parts of the algorithm: (1) the learning part and (2) the encoding part. In the
learning part, the algorithm "learns" the tokenization from a (large) text corpus. For a given
word, the algorithm appends a special word delimiter symbol, starts with its sequence of characters
and iteratively merges the most frequent token pairs into new (longer) tokens.
In the encoding part, the algorithm encodes a (possibly unknown) word by (1) appending the same
word delimiter symbol as above, (2) segmenting the word into as few learned tokens as possible,
and (3) representing the word by the sequence of ids of the tokens.

Example:
Let the word delimiter symbol be "·" and the learned tokens be {"er·": 0, "w": 1, "fl": 2,
"o": 3, "po": 4}. Then, the sequence "flower power" will be encoded as [2, 3, 1, 0, 4, 1, 0].
"""

import logging  # NOQA
import os  # NOQA
import sys
sys.path.append("../..")  # Needed so that utils.* can be found. # NOQA

from semantic_roles_detection.utils import log_utils

# =================================================================================================

# The logger.
LOG = log_utils.get_logger(__name__)
LOG.setLevel(logging.DEBUG)

# =================================================================================================

# The symbol to use as padding.
PADDING_SYMBOL = "■"
# The symbol to use instead of a character unknown to the vocabulary.
UNKNOWN_CHAR_SYMBOL = "�"
# The symbol to use as word delimiter.
WORD_DELIM_SYMBOL = "✂"

# The number of Unicode characters with which the vocabulary should be initialized.
NUM_INITIAL_CHARS = 256
# The number of byte pairs to merge while creating the vocabulary.
NUM_MERGES = 2000

# =================================================================================================


def create_vocabulary(words, num_initial_chars=NUM_INITIAL_CHARS, num_merges=NUM_MERGES):
    """
    Creates a vocabulary from the given list of words as needed for byte pair encoding word
    sequences, that is: a dictionary mapping the first <num_initial_chars>-many *printable*
    Unicode characters and the <num_merges>-many most frequent tokens used in the words to an
    unique integer id.

    Args:
        words (list of str):
            The list of words to create the vocabulary from.
        num_initial_chars (int):
            The number of Unicode characters with which the vocabulary should be initialized.
        num_merges (int):
            The number of byte pairs to merge while creating the vocabulary.

    >>> LOG.setLevel(logging.ERROR)
    >>> vocab = create_vocabulary(["low", "lower", "flow", "flower", "flower"], num_merges=5)
    >>> [vocab.get(k, None) for k in ("f", "l", "o", "w", "e", "r")]
    [69, 75, 78, 86, 68, 81]
    >>> [vocab.get(k, None) for k in ("lo", "low", "lowe", "lower", "lower✂")]
    [256, 257, 258, 259, 260]
    >>> len(vocab)
    261

    >>> vocab = create_vocabulary(["low", "lower", "flower"], num_initial_chars=3, num_merges=5)
    >>> sorted(vocab.items())
    [('!', 0), ('"', 1), ('#', 2), ('lo', 3), ('low', 4), ('lowe', 5), ('lower', 6), ('lower✂', 7)]
    """
    # Initialize the vocabulary with the first <num_initial_chars> *printable* Unicode characters,
    # for example: vocabulary = {'!': 0, '"': 1, '#': 2, '$': 3, ...}
    LOG.debug("Initializing the vocabulary with {} characters ...".format(num_initial_chars))
    vocabulary = {}
    unicode = -1
    while len(vocabulary) < num_initial_chars:
        unicode += 1
        # Ignore all control characters between unicode 0 and 31 and the whitespace (code 32).
        if unicode <= 32:
            continue
        # Ignore all control characters between unicode 127 and 160.
        if unicode >= 127 and unicode <= 160:
            continue
        # Ignore "Soft hyphen" (code 173).
        if unicode == 173:
            continue

        # Add {char: id} to the vocabulary.
        vocabulary[chr(unicode)] = len(vocabulary)
    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[-3:]])
    LOG.debug("vocabulary = {{{}, ..., {}}}".format(head, tail))

    # Also add the <num_merges>-many most frequent token pairs to the vocabulary. To do so, we
    # first compute the token frequencies and then determine the most frequent token pairs.

    LOG.debug("Count the word frequencies ...")
    # First, compute the token freqs: for ["foo", "boo" "foo"], compute [("foo·", 2), ("boo·", 1)].
    token_freqs = {}
    for word in words:
        # Add the word delimiter to each word.
        word += WORD_DELIM_SYMBOL
        if word not in token_freqs:
            token_freqs[word] = 1
        else:
            token_freqs[word] += 1
    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(token_freqs.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(token_freqs.items())[-3:]])
    LOG.debug("token_freqs = {{{}, ..., {}}}".format(head, tail))
    token_freqs = token_freqs.items()

    LOG.debug("Adding the {} most frequent token pairs to the vocabulary ... ".format(num_merges))
    # Merge <num_merges>-many token pairs and add them to the vocabulary, for example:
    # vocabulary = {..., "oo": 256, "o·": 257, "oo·": 258, "fo": 259, "bo": 260}
    for i in range(num_merges):
        if i == 0 or (i + 1) % max(1, int(0.1 * num_merges)) == 0:
            LOG.debug("Merging token pairs, iteration {}/{} ...".format(i + 1, num_merges))

        # Inspect the token pairs in the strings of token_freqs and find the most frequent one.
        # For example, for token_freqs=[("foo·", 2), ("boo·", 1)], compute ("o", "o"), because
        # this pair occurs most frequently (three times; in "foo", which occurs twice and in "boo",
        # which occurs once).
        best_merge_pair = find_best_token_pair(token_freqs)

        # Abort if no token pair was found.
        if best_merge_pair is None:
            break

        # Add the token pair to the vocabulary and update the token frequencies.
        # For example, for pair ("o", "o"), compute:
        # vocabulary = {..., "oo": 256}
        # token_freqs = [(["f", "oo", "·"], 2), (["b", "oo", "·"], 1)]
        token_freqs = apply_merge(best_merge_pair, vocabulary, token_freqs)

    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[-3:]])
    LOG.debug("vocabulary = {{{}, ..., {}}}".format(head, tail))

    return vocabulary


def find_best_token_pair(token_freqs):
    """
    Iterates through the given token frequencies, in the format:
    [("string1", <freq1>), ("string2", <freq2>), ...] or [(["s1", "s2", ...], <freq1>), ...]
    and computes the most frequent token pair.

    Args:
        token_freqs (list of tuples):
            The dictionary with the token frequencies to find the most frequent token pair from.

    >>> LOG.setLevel(logging.ERROR)
    >>> find_best_token_pair([]) is None
    True
    >>> find_best_token_pair([(["foo"], 3)]) is None
    True
    >>> find_best_token_pair([("foo", 3)])
    ('f', 'o')
    >>> find_best_token_pair([("flower✂", 1), ("power✂", 2)])
    ('o', 'w')
    >>> find_best_token_pair([(["f", "o", "o"], 2), (["b", "o", "o"], 3)])
    ('o', 'o')
    >>> find_best_token_pair([(["fo", "o", "l"], 2), (["fo", "o", "x"], 3)])
    ('fo', 'o')
    >>> find_best_token_pair([(["a", "b", "c✂"], 2), (["d", "b", "c✂"], 3)])
    ('b', 'c✂')
    >>> find_best_token_pair([(["a", "✂"], 5)])
    ('a', '✂')
    """
    pairs = {}
    for token, freq in token_freqs:
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            if pair not in pairs:
                pairs[pair] = freq
            else:
                pairs[pair] += freq
    # Note: 'pairs' can be empty if no merge pair could be found anymore.
    return max(pairs, key=pairs.get) if pairs else None


def apply_merge(best_merge_pair, vocabulary, token_freqs):
    """
    Adds the given "best" (= most frequent) token pair to the given vocabulary and creates (and
    returns) a new token frequencies dictionary, based on the given "old" token frequencies.

    NOTE: The given vocabulary will be updated in-place, whereas the returned token frequencies
    dictionary is a newly created object.

    Args:
        best_merge_pair (tuple (str, str)):
            The token pair to add to the vocabulary and the token frequencies dictionary.
        vocabulary (dict str:int):
            The vocabulary to update with the token pair.
        token_freqs (dict str:int or list:int):
            The token frequencies dictionary to update with the token pair.

    >>> LOG.setLevel(logging.ERROR)
    >>> vocab = {'f': 0, 'o': 1, 'x': 2}
    >>> token_freqs = [(['f', 'o', 'o'], 2), (['f', 'o', 'x'], 1)]

    >>> token_freqs = apply_merge(('f', 'o'), vocab, token_freqs)
    >>> vocab
    {'f': 0, 'o': 1, 'x': 2, 'fo': 3}
    >>> token_freqs
    [(['fo', 'o'], 2), (['fo', 'x'], 1)]

    >>> token_freqs = apply_merge(('fo', 'o'), vocab, token_freqs)
    >>> vocab
    {'f': 0, 'o': 1, 'x': 2, 'fo': 3, 'foo': 4}
    >>> token_freqs
    [(['foo'], 2), (['fo', 'x'], 1)]
    """
    # Add the token pair to the vocabulary.
    merged = best_merge_pair[0] + best_merge_pair[1]
    vocabulary[merged] = len(vocabulary)

    # Recompute the token frequencies. For example, if the token frequencies were
    # [(['f', 'o', 'o'], 2), (['f', 'o', 'x'], 1)] and the best merge pair is ('f', 'o'),
    # create new token frequencies [(['fo', 'o'], 2), (['fo', 'x'], 1)]
    new_token_freqs = []
    for token, freq in token_freqs:
        new_token = []
        i = 1
        while i <= len(token):
            if tuple(token[(i - 1):(i + 1)]) == best_merge_pair:
                new_token.append(merged)
                i += 2
            else:
                new_token.append(token[i - 1])
                i += 1
        new_token_freqs.append((new_token, freq))
    return new_token_freqs

# =================================================================================================


class BytePairEncoder:
    """
    An encoder that can be used to encode word sequences using byte pair encoding.
    """

    def __init__(self, vocabulary={}):
        """
        Creates a new instance of this encoder.

        Args:
            vocabulary (dict of str:int)
                The vocabulary to use on encoding the sequences, mapping tokens to unique ids.
            logging_level (str):
                The logging level for this encoder.

        >>> bpe = BytePairEncoder({})
        >>> sorted(bpe.vocabulary.items(), key=lambda x:x[1])
        [('■', 0), ('�', 1), ('✂', 2)]
        >>> sorted(bpe.rev_vocabulary.items(), key=lambda x:x[0])
        [(0, '■'), (1, '�'), (2, '✂')]
        """
        # The vocabulary, mapping tokens to unique ids.
        self.vocabulary = vocabulary
        # Add the padding symbol and the word delimiter symbol to the vocabulary.
        if PADDING_SYMBOL not in self.vocabulary:
            self.vocabulary[PADDING_SYMBOL] = max(vocabulary.values()) + 1 if vocabulary else 0
        if UNKNOWN_CHAR_SYMBOL not in self.vocabulary:
            self.vocabulary[UNKNOWN_CHAR_SYMBOL] = max(vocabulary.values()) + 1
        if WORD_DELIM_SYMBOL not in self.vocabulary:
            self.vocabulary[WORD_DELIM_SYMBOL] = max(vocabulary.values()) + 1

        # The reversed vocabulary, mapping unique ids to tokens.
        self.rev_vocabulary = {v: k for k, v in vocabulary.items()}
        # The cache with encodings already computed (mapping a word to its actual encoding).
        self.encodings_cache = {}

    # =============================================================================================

    def encode(self, words, target_length=-1):
        """
        Encodes the given list of words using byte pair encoding and cuts or pads the sequence to
        the given target length.

        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab_file = os.path.join(base_dir, "../../examples/vocab-words.example.txt")
        >>> vocab = utils.files.read_vocabulary_file(vocab_file)

        >>> bpe = BytePairEncoder(vocab)
        >>> bpe.encode(None) is None
        True
        >>> bpe.encode([])
        []
        >>> bpe.encode([None])
        []
        >>> bpe.encode([""])
        []
        >>> bpe.encode(["computer", "Trash", "killer"])
        [270, 79, 84, 83, 258, 51, 81, 64, 82, 71, 283, 74, 72, 259]
        >>> bpe.encode(["computer", "Trash", "killer"], target_length=17)
        [270, 79, 84, 83, 258, 51, 81, 64, 82, 71, 283, 74, 72, 259, 281, 281, 281]
        >>> bpe.encode(["computer", "Trash", "killer"], target_length=5)
        [270, 79, 84, 83, 258]
        """
        if words is None:
            return

        encoded_words = []
        for word in words:
            if not word:
                continue
            word += WORD_DELIM_SYMBOL
            encoded_words.extend(self.byte_pair_encode(word))

        # Cut or pad the sequence to the given target length.
        if target_length >= 0:
            # Pad the sequence to the target length.
            while len(encoded_words) < target_length:
                encoded_words.append(self.vocabulary[PADDING_SYMBOL])
            # Cut the sequence to the target length.
            encoded_words = encoded_words[:target_length]
        return encoded_words

    def byte_pair_encode(self, word):
        """
        Computes the byte pair encoding of the given word.

        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab_file = os.path.join(base_dir, "../../examples/vocab-words.example.txt")
        >>> vocab = utils.files.read_vocabulary_file(vocab_file)

        >>> bpe = BytePairEncoder(vocab)
        >>> bpe.byte_pair_encode(None) is None
        True
        >>> bpe.byte_pair_encode("")
        []
        >>> bpe.byte_pair_encode("computer")
        [270, 79, 84, 83, 257]
        >>> bpe.byte_pair_encode("computer✂")
        [270, 79, 84, 83, 258]
        >>> bpe.byte_pair_encode("Trash")
        [51, 81, 64, 82, 71]
        >>> bpe.byte_pair_encode("killer")
        [74, 72, 256, 257]
        >>> bpe.byte_pair_encode("September")
        [278, 257]
        >>> bpe.byte_pair_encode("september")
        [82, 68, 79, 83, 68, 76, 65, 257]
        >>> bpe.byte_pair_encode("September✂")
        [279]
        """
        if word is None:
            return

        if len(word) == 0:
            return []

        # Return the cached encoding, if available.
        if word in self.encodings_cache:
            return self.encodings_cache[word]

        # Compute the token pairs of the word, with the respective start positions:
        # "foxifox" -> {"fo": [0, 4], "ox": [1, 5], "xi": [2], "if": [3]}
        token_pair_positions = self.compute_token_pair_positions(word)

        if not token_pair_positions:
            return [self.vocabulary[word]]

        word_tokens = word
        while True:
            # From token_pair_positions, find the pair that is also included in the vocabulary.
            best_pair_positions = None
            for token in token_pair_positions:
                if token in self.vocabulary:
                    best_pair_positions = token_pair_positions[token]
                    break

            if not best_pair_positions:
                break

            # Merge all occurences of the best token pair to a new token.
            i = 0
            new_word_tokens = []
            while i < len(word_tokens):
                if i in best_pair_positions:
                    new_word_tokens.append("".join(word_tokens[i:i+2]))
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1

            word_tokens = new_word_tokens
            if len(word_tokens) == 1:
                break

            token_pair_positions = self.compute_token_pair_positions(word_tokens)

        # Translate the tokens to their ids in the vocabulary.
        encoded = []
        for token in word_tokens:
            if token in self.vocabulary:
                encoded.append(self.vocabulary[token])
            else:
                encoded.append(self.vocabulary[UNKNOWN_CHAR_SYMBOL])

        self.encodings_cache[word] = encoded
        return encoded

    def compute_token_pair_positions(self, word):
        """
        Computes the token pairs of the given word with their start positions. Returns a
        dictionary, where the keys are the pairs represented as a string (the two elements of a
        pair merged), and the values are set of integers, representing the start positions of the
        pair in the word.

        Args:
            word (str or list of str):
                The word to compute the token pairs for.

        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab_file = os.path.join(base_dir, "../../examples/vocab-words.example.txt")
        >>> vocab = utils.files.read_vocabulary_file(vocab_file)

        >>> bpe = BytePairEncoder(vocab)
        >>> bpe.compute_token_pair_positions(None) is None
        True
        >>> bpe.compute_token_pair_positions([])
        {}
        >>> bpe.compute_token_pair_positions([None])
        {}
        >>> bpe.compute_token_pair_positions([""])
        {}
        >>> sorted(bpe.compute_token_pair_positions("foxifox").items())
        [('fo', {0, 4}), ('if', {3}), ('ox', {1, 5}), ('xi', {2})]
        >>> sorted(bpe.compute_token_pair_positions(["fo", "x", "if", "ox", "i", "fox"]).items())
        [('fox', {0}), ('ifox', {2, 4}), ('oxi', {3}), ('xif', {1})]
        """
        if word is None:
            return

        if len(word) == 0:
            return {}

        positions = {}
        tokens = tuple(word)
        prev_token = tokens[0]
        for i, token in enumerate(tokens[1:]):
            merged = prev_token + token
            if merged in positions:
                positions[merged].add(i)
            else:
                positions[merged] = {i}
            prev_token = token
        return positions

    # =============================================================================================

    def decode(self, sequence):
        """
        Decodes the given byte-pair-encoded sequence of words.

        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab_file = os.path.join(base_dir, "../../examples/vocab-words.example.txt")
        >>> vocab = utils.files.read_vocabulary_file(vocab_file)

        >>> bpe = BytePairEncoder(vocab)
        >>> bpe.decode(None) is None
        True
        >>> bpe.decode([])
        []
        >>> bpe.decode([-1])
        []
        >>> bpe.decode(bpe.encode(["computer", "Trash", "killer"]))
        ['computer', 'Trash', 'killer']
        >>> bpe.decode(bpe.encode(["computer", "Trash", "killer"], target_length=20))
        ['computer', 'Trash', 'killer']
        >>> bpe.decode(bpe.encode(["computer", "Trash", "killer"], target_length=4))
        ['comput']
        """
        if sequence is None:
            return

        if len(sequence) == 0:
            return []

        decoded_sequence = []
        for id in sequence:
            # Ignore padding symbols.
            if id == self.vocabulary[PADDING_SYMBOL]:
                continue
            # Ignore unknown symbols.
            if id not in self.rev_vocabulary:
                continue
            decoded_sequence.append(self.rev_vocabulary[id])

        if len(decoded_sequence) == 0:
            return []

        # Translate the ids to tokens and ignore the padding.
        decoded = "".join(decoded_sequence)
        # Remove trailing word delimiters.
        decoded = decoded.strip(WORD_DELIM_SYMBOL)
        # Split the sequence into words.
        return decoded.split(WORD_DELIM_SYMBOL)
