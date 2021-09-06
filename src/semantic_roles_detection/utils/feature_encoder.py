"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains the code to
encode text blocks (extracted from PDF files) into feature vectors, in the form as required by
our models for predicting the semantic roles of the text blocks.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import string
from typing import Dict, List, Tuple

import numpy as np

from semantic_roles_detection.utils import bpe
from semantic_roles_detection.utils import log_utils
from semantic_roles_detection.utils import word_tokenizer
from semantic_roles_detection.utils import zip_utils
from semantic_roles_detection.utils.models import Document, Page, TextBlock

# =================================================================================================
# Parameters.

# The logger.
LOG = log_utils.get_logger(__name__)
# The symbols to be considered as non-characters.
NON_CHAR_SYMBOLS = "'!#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n "
# The symbols to be considered as word-delimiters.
WORD_DELIMITERS = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n "

# =================================================================================================


class FeatureEncoder:
    """
    This class encodes given text blocks into feature vectors, in the form as required by our
    models for predicting the semantic roles of the text blocks.
    """

    def __init__(self, bpe_vocab: Dict[str, int] = {}, roles_vocab: Dict[str, int] = {},
                 word_delimiters: str = WORD_DELIMITERS, is_lowercase_text: bool = False,
                 word_seq_length: int = 100, is_include_positions: bool = True,
                 is_include_font_sizes: bool = True, is_include_font_styles: bool = True,
                 is_include_char_features: bool = True, is_include_semantic_features: bool = True,
                 countries_db_file: str = None, human_names_db_file: str = None,
                 logging_level: str = "info"):
        """
        This constructor creates and initializes a new `FeatureEncoder`.

        Args:
            bpe_vocab: Dict[str, int]:
                The vocabulary to be used on encoding the word sequences with byte pair encoding.
                It is supposed to be a dictionary that maps byte pairs to integer ids.
            roles_vocab: Dict[str, int]:
                The vocabulary to be used on encoding the semantic roles of the text blocks. It is
                supposed to be a dictionary that maps all semantic role names to integer ids.
            word_delimiters: str
                The characters to be considered as word delimiters on splitting the text of a
                text block into words. For example, if set to "-+", the text "foo-bar+baz boo"
                is splitted into the words "foo", "bar" and "baz boo" (note that "baz boo" won't be
                split, because the character " " is not part of "word_delimiters").
            is_lowercase_text: bool
                A boolean flag indicating whether or not the text of the text blocks should be
                lowercased.
            word_seq_length: int
                The target length for the word sequences. If set to n, all returned word sequences
                will be of size n. All word sequences that are actually shorter than n will be
                padded with the padding symbol stored in `bpe_vocab`. All word sequences that are
                actually longer than n will be truncated. If this parameter is set to -1, *all*
                word sequnces will be padded to the length of the longest word sequence.
            is_include_positions: bool
                A boolean flag indicating whether or not to include the positions of the text
                blocks should be included in the layout feature sequences.
            is_include_font_sizes: bool
                A boolean flag indicating whether or not to include the font sizes of the text
                blocks in the layout feature sequences.
            is_include_font_styles: bool
                A boolean flag indicating whether or not to include the font styles (= the
                information whether or not a block is printed in bold and/or italic) of the text
                blocks in the layout feature sequences.
            is_include_char_features: bool
                A boolean flag indicating whether or not to include the features about the
                characters contained in a text block (e.g., whether or not the block contains a
                '@', which could hint at the role 'AUTHOR_MAIL', or the number of uppercased
                characters) into the layout feature sequences.
            is_include_semantic_features: bool
                A boolean flag indicating whether or not to include some (simple) semantic features
                about the words of a text block (e.g., whether a word denotes the name of a
                human or a country) into the layout feature sequences.
            countries_db_file: str
                The path to a file providing a set of country names, in the format: one country per
                line. These names are used to determine if a given word belongs to a country name
                or not (see also the information given for parameter 'include_semantic_features').
            human_names_db_file: str
                The path to a file providing human names, in the format: one name per line.
                These names are used to determine if a given word belongs to a human name or not
                (see also the information given for parameter 'include_semantic_features').
            logging_level: str
                The logging level.
        """
        self.bpe_vocab = bpe_vocab
        self.rev_bpe_vocab = {v: k for k, v in bpe_vocab.items()}
        self.roles_vocab = roles_vocab
        self.rev_roles_vocab = {v: k for k, v in roles_vocab.items()}
        self.word_tokenizer = word_tokenizer.WordTokenizer(word_delimiters)
        self.is_lowercase_text = is_lowercase_text
        self.word_seq_length = word_seq_length
        self.is_include_positions = is_include_positions
        self.is_include_font_sizes = is_include_font_sizes
        self.is_include_font_styles = is_include_font_styles
        self.is_include_char_features = is_include_char_features
        self.is_include_semantic_features = is_include_semantic_features
        self.countries_db_file = countries_db_file
        self.human_names_db_file = human_names_db_file

        # A dict mapping each non-char symbol to a whitespace.
        self.non_chars_trans = str.maketrans({c: " " for c in NON_CHAR_SYMBOLS})

        # Configure the logging level.
        LOG.setLevel(log_utils.to_log_level(logging_level))

    # =============================================================================================

    def encode_documents(self, documents: List[Document]) -> Tuple[np.array, np.array, np.array]:
        """
        This method encodes the text blocks of the given documents and returns:

        (1) A (flat) list containing all text blocks of the documents. This is useful to relate
            a word sequence and/or layout feature sequence in the arrays described below to the
            corresponding text block.
        (2) an array of word sequences, each containing the words of a specific text block from the
            given documents, encoded with byte pair encoding.
        (3) an array of layout feature sequences, each containing some layout features of a text
            block from the given documents, for example: the position, the font information, etc.
        (4) an array of the semantic roles, each containing the role of a text block, encoded
            with one-hot-encoding.

        Args:
            documents: List[Document]
                The documents for which to encode the text blocks.
        Returns:
            Tuple[np.array, np.array, np.array]
                The arrays described above.

        TODO
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> roles_vocab_file = os.path.join(base_dir, "./examples/vocab-roles.example.txt")
        >>> roles_vocab = utils.files.read_vocabulary_file(roles_vocab_file)

        >>> dr = DataReader("examples",
        ...   input_file_name_pattern="*.tsv",
        ...   shuffle_input_files=False,
        ...   num_words_per_seq=2,
        ...   logging_level="fatal",
        ...   include_semantic_features=False,
        ...   roles_vocab=roles_vocab
        ... )
        >>> word_sequences, features, roles = dr.read()
        >>> sorted(dr.roles_vocab.items(), key=lambda x: x[1])
        ... #doctest: +NORMALIZE_WHITESPACE
        [('AUTHOR_NAME', 0), ('PUBLICATION-DATE', 1), ('TITLE', 2), ('HEADING_HEADING', 3), \
         ('PARAGRAPHS', 4), ('AUTHOR_MAIL', 5), ('UNK_ROLE', 6)]
        >>> sorted(dr.rev_roles_vocab.items(), key=lambda x: x[0])
        ... #doctest: +NORMALIZE_WHITESPACE
        [(0, 'AUTHOR_NAME'), (1, 'PUBLICATION-DATE'), (2, 'TITLE'), (3, 'HEADING_HEADING'), \
         (4, 'PARAGRAPHS'), (5, 'AUTHOR_MAIL'), (6, 'UNK_ROLE')]
        >>> dr.text_blocks
        ... #doctest: +NORMALIZE_WHITESPACE
        [BB(AUTHOR_NAME; 1; 200.0;540.0;280.0;550.0; lmroman; 12.0; 0; 0; "Ben Müller"), \
         BB(PUBLICATION-DATE; 1; 0.0;0.0;0.0;0.0; lmroman; 12.0; 0; 0; "September 2017"), \
         BB(TITLE; 1; 141.9;571.4;469.4;627.3; arial; 17.2; 0; 1; "A catchy title!"), \
         BB(HEADING_HEADING; 2; 158.0;332.3;219.7;342.4; lmroman; 14.3; 1; 0; "Abstract"), \
         BB(PARAGRAPHS; 3; 210.0;460.0;230.0;500.0; lmroman; 10.0; 0; 0; "Bullshit-Bingo."), \
         BB(AUTHOR_MAIL; 1; 92.1;702.8;517.6;720.0; nimbussanl; 17.9; 1; 0; "miller@gmail.com."), \
         BB(TITLE; 1; 62.0;682.8;547.7;720.0; nimbussanl; 17.9; 1; 0; "A comprehensive survey."), \
         BB(AUTHOR_NAME; 1; 114.8;642.9;202.4;654.2; arial; 12.0; 0; 0; "Anne Müller"), \
         BB(AUTHOR_COUNTRY; 1; 143.0;613.9;174.2;621.4; nimbussanl; 10.0; 0; 0; "Mexico"), \
         BB(PARAGRAPHS; 2; 62.8;176.1;292.9;184.2; lmroman; 9.0; 0; 0; "1 Introduction.")]
        >>> np.round(features, 1).tolist()
        ... #doctest: +NORMALIZE_WHITESPACE
         [[0.0, 0.5, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.2], \
          [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.1], \
          [0.0, 0.6, 0.6, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.1], \
          [0.5, 0.3, 0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1], \
          [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1], \
          [0.0, 0.5, 0.8, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0], \
          [0.0, 0.5, 0.8, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0], \
          [0.0, 0.3, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.2], \
          [0.0, 0.3, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2], \
          [1.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.5, 0.1]]
        >>> np.round(roles, 1).tolist()
        ... #doctest: +NORMALIZE_WHITESPACE
        [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], \
         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], \
         [0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], \
         [0, 0, 0, 0, 1, 0, 0]]
        """
        text_blocks = []
        word_seqs = []
        layout_feature_seqs = []
        roles = []

        LOG.info(f"Encoding {len(documents)} documents ...")
        for document in documents:
            w, l, r = self.encode_text_blocks(document.blocks, document.pages)

            # TODO: What to do if w and/or l is empty?
            if w.size == 0 or l.size == 0 or r.size == 0:
                continue

            text_blocks.extend(document.blocks)
            word_seqs.append(w)
            layout_feature_seqs.append(l)
            roles.append(r)

        return text_blocks, np.concatenate(word_seqs), np.concatenate(layout_feature_seqs), \
            np.concatenate(roles)

    def encode_text_blocks(self, text_blocks: List[TextBlock], pages: List[Page]) -> \
            Tuple[np.array, np.array, np.array]:
        """
        This method encodes the given text blocks and returns:

        (1) an array of word sequences, each containing the words of a specific text block from the
            given documents, encoded with byte pair encoding.
        (2) an array of layout feature sequences, each containing some layout features of a text
            block from the given documents, for example: the position, the font information, etc.
        (3) an array of the semantic roles, each containing the role of a text block, encoded
            with one-hot-encoding.

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks to encode.
            pages: List[Page]
                The pages of the document from which the given text blocks were extracted. The
                page belonging to the i-th text block is expected to be stored at
                pages[text_blocks[i].page_num - 1].
        Returns:
            Tuple[np.array, np.array, np.array]
                The arrays described above.
        """
        # Split the texts of the text blocks into word sequences (one word sequence per block).
        word_seqs = self.split_into_word_sequences(text_blocks, self.is_lowercase_text)
        # Encode the word sequences.
        word_seqs = self.encode_word_sequences(word_seqs, self.bpe_vocab, self.word_seq_length)

        # Initialize the array of layout feature sequences (one empty sequence per text block).
        features = [[] for _ in text_blocks]

        if self.is_include_positions:
            encoded_positions = self.encode_positions(text_blocks, pages)
            features = zip_utils.zip_lists(features, encoded_positions)

        if self.is_include_font_sizes:
            encoded_font_sizes = self.encode_font_sizes(text_blocks)
            features = zip_utils.zip_lists(features, encoded_font_sizes)

        if self.is_include_font_styles:
            encoded_font_styles = self.encode_font_styles(text_blocks)
            features = zip_utils.zip_lists(features, encoded_font_styles)

        if self.is_include_char_features:
            encoded_char_features = self.encode_character_features(text_blocks)
            features = zip_utils.zip_lists(features, encoded_char_features)

        # if self.is_include_semantic_features:
        #     LOG.info("Encoding the semantic features ...")
        #     encoded_semantic_features = self.encode_semantic_features(
        #         text_blocks, self.countries_db_file, self.human_names_db_file)
        #     features = utils.zip.zip_lists(features, encoded_semantic_features)

        roles = self.encode_semantic_roles(text_blocks, self.roles_vocab)

        # Translate the sequences to numpy arrays.
        word_sequences = np.array(word_seqs)
        assert not np.any(np.isnan(word_sequences))

        features = np.array(features)
        assert not np.any(np.isnan(features))

        roles = np.array(roles)
        assert not np.any(np.isnan(roles))

        return word_sequences, features, roles

    # =============================================================================================

    def split_into_word_sequences(self, text_blocks: List[TextBlock],
                                  is_lowercase_text: bool = False) -> List[List[str]]:
        """
        This method splits the text of each given text block at each character occurring in
        self.word_delimiters into words. Converts all words to lower cases if 'is_lowercase_text'
        is set to True. Returns a list of string lists, where the i-th string list is the sequence
        of words computed from the i-th text block.

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks to split into words.
            is_lowercase_text: bool
                A boolean flag indicating whether or not to convert the words to lower cases.
        Returns:
            List[List[str]]:
                A list of string lists, where the i-th string list is the sequence of words
                computed from the i-th text block.

        TODO
        >>> dr = DataReader(None, logging_level="fatal", word_delimiters=" ")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files(["examples/article.4.tsv"])

        >>> dr.split_into_word_sequences(blocks)
        ... #doctest: +NORMALIZE_WHITESPACE
        [['Ben', 'Müller'], ['September', '2017'], ['A', 'catchy', 'title!'], ['Abstract'], \
         ['Bullshit-Bingo.']]
        >>> dr.split_into_word_sequences(blocks, lowercase=True)
        ... #doctest: +NORMALIZE_WHITESPACE
        [['ben', 'müller'], ['september', '2017'], ['a', 'catchy', 'title!'], ['abstract'], \
         ['bullshit-bingo.']]

        >>> dr = DataReader(None, logging_level="fatal", word_delimiters=" -!")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files( \
                ["examples/article.4.tsv", "examples/vldb.2.tsv"])
        >>> dr.split_into_word_sequences(blocks)
        ... #doctest: +NORMALIZE_WHITESPACE
        [['Ben', 'Müller'], ['September', '2017'], ['A', 'catchy', 'title'], ['Abstract'], \
         ['Bullshit', 'Bingo.'], ['A', 'comprehensive', 'survey.'], ['Anne', 'Müller'], \
         ['Mexico'], ['1', 'Introduction.']]
        """
        word_seqs = []

        for block in text_blocks:
            # Split the text into words.
            words: List[str] = self.word_tokenizer.tokenize_into_words(block.text)

            # Convert the words to lower cases.
            if is_lowercase_text:
                words = [w.lower() for w in words]

            word_seqs.append(words)

        return word_seqs

    # =============================================================================================

    def encode_word_sequences(self, word_seqs: List[List[str]], bpe_vocab: Dict[str, int] = None,
                              word_seq_length: int = 100) -> List[List[int]]:
        """
        This method encodes each given word sequence (denoting the words of a specific text block)
        with byte pair encoding. Returns a list of integer lists, where the i-th integer list
        denotes the encoding of the i-th word sequence. Each integer list contains exactly
        <num_words_per_seq>-many integers.

        Args:
            word_seqs: List[List[str]]:
                The word sequences to encode.
            bpe_vocab: Dict[str, int]:
                The BPE vocabulary, that is: a dictionary that maps byte pairs to integer values.
            num_words_per_seq: int
                The target length for the word sequences. All word sequences that actually contain
                more elements than this value, will be truncated. All word sequences that actually
                contain less elements than this value will be padded with the padding symbol stored
                in bpe_vocab.
        Returns:
            List[List[int]]
                A list of integer lists, where the i-th integer list denotes the encoding of the
                i-th word sequence. Each integer list is of length <num_words_per_seq>.
        """
        encoder = bpe.BytePairEncoder(bpe_vocab)
        return [encoder.encode(seq, word_seq_length) for seq in word_seqs]

    # =============================================================================================

    def encode_positions(self, text_blocks: List[TextBlock], pages: List[Page]) -> \
            List[List[float]]:
        """
        This method encodes the positions of the given text blocks (that is: the page numbers
        and the x/y-coordinates). The page number of a text block is normalized by dividing it
        by the maximum page number among the text blocks. The x/y coordinates are normalized
        by dividing them by the width/height of the belonging page.
        Returns a list of float lists, where the i-th float list denotes the encoded position of
        the i-th text block.

        Args:
            text_blocks: List[TextBlock]:
                The list of text blocks for which to encode the positions.
            pages: List[Page]
                The pages of the document from which the given text blocks were extracted. The
                page belonging to the i-th text block is expected to be stored at
                pages[text_blocks[i].page_num - 1].
        Returns:
            List[List[float]]
                A list of float lists, where the i-th float list denotes the encoded position of
                the i-th text block.

        TODO
        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, pages, _, _ = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"])
        >>> encoded = dr.encode_positions(blocks, pages)
        >>> [["%.2f" % x for x in block] for block in encoded] #doctest: +NORMALIZE_WHITESPACE
        [['0.00', '0.50', '0.55'], ['0.00', '0.00', '0.00'], ['0.00', '0.64', '0.60'], \
         ['0.50', '0.32', '0.40'], ['1.00', '1.00', '1.00'], ['0.00', '0.53', '0.83'], \
         ['0.00', '0.28', '0.76'], ['0.00', '0.28', '0.73'], ['1.00', '0.26', '0.24'], \
         ['0.00', '0.53', '0.84']]
        """
        encoded_positions = []

        # Compute the maximum page number among the pages.
        page_nums = [x.page_num for x in pages]
        max_page_num = max(page_nums) if len(page_nums) > 0 else 0

        for block in text_blocks:
            position = []

            # Encode the page number.
            if max_page_num > 1:
                # Normalize the 1-based page numbers.
                position.append((block.page_num - 1) / (max_page_num - 1))
            else:
                position.append(0.0)

            page_width = pages[block.page_num - 1].width
            page_height = pages[block.page_num - 1].height

            # Encode the x/y-coordinates of the lower left.
            position.append((block.lower_left_x / page_width) if page_width > 0 else 0.0)
            position.append((block.lower_left_y / page_height) if page_height > 0 else 0.0)

            # Encode the x/y-coordinates of the upper right.
            position.append((block.upper_right_x / page_width) if page_width > 0 else 0.0)
            position.append((block.upper_right_y / page_height) if page_height > 0 else 0.0)

            encoded_positions.append(position)

        return encoded_positions

    # =============================================================================================

    def encode_font_sizes(self, text_blocks: List[TextBlock]) -> List[float]:
        """
        This method encodes the font sizes of the given text blocks by translating the font
        sizes to a value between 0 (denoting the minimum font in the document) and 1 (denoting the
        maximum font size in the document).
        Returns a list of floats, where the i-th float is the encoded font size of the i-th
        text block (a value between 0 and 1).

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks for which to encode the font sizes.
        Returns:
            List[float]
                A list of floats, where the i-th float is the encoded font size of the i-th
                text block (a value between 0 and 1).

        TODO
        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, _, min_fs, max_fs = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"])
        >>> font_sizes = dr.encode_font_sizes(blocks, min_fs, max_fs)
        >>> [round(fs, 1) for fs in font_sizes]
        [0.3, 0.3, 1.0, 0.6, 0.0, 1.0, 0.3, 0.1, 0.0, 0.0]
        """
        encoded_font_sizes = []

        # Compute the minimum and maximum font size among the text blocks.
        min_font_size = float("inf")
        max_font_size = 0
        for block in text_blocks:
            min_font_size = min(min_font_size, block.font_size)
            max_font_size = max(max_font_size, block.font_size)

        for block in text_blocks:
            # Use the whole interval [0, 1], that is: translate the min font size to 0 and the max
            # font size to 1. For example, if the min font size in a document is 8 and the max font
            # size is 10, then translate font size 10 to: (10 - 8) / (12 - 8) = 0.5
            if min_font_size == max_font_size:
                encoded_font_sizes.append(0.0)
            else:
                encoded_font_sizes.append(
                    (block.font_size - min_font_size) / (max_font_size - min_font_size)
                )

        return encoded_font_sizes

    # =============================================================================================

    def encode_font_styles(self, text_blocks: List[TextBlock]) -> List[List[int]]:
        """
        This method encodes the font styles (that is: the information whether or not a text
        block is printed in bold and/or italics) of the given text blocks.
        Returns a list of integer lists, where the i-th integer lists denotes the encoded font
        style of the i-th text block.

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks for which to encode the font styles.
        Returns:
            List[List[int]]
                A list of integer lists, where the i-th integer lists denotes the encoded font
                style of the i-th text block.

        # TODO
        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"])
        >>> dr.encode_font_styles(blocks)
        [[0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0]]
        """
        encoded_font_styles = []

        for block in text_blocks:
            # Append the 'isBold' flag and 'isItalic' flag (both 0 or 1).
            encoded_font_styles.append([int(block.is_bold), int(block.is_italic)])

        return encoded_font_styles

    # =============================================================================================

    def encode_character_features(self, text_blocks: List[TextBlock]) -> List[List[float]]:
        """
        This method encodes some features about the characters contained in each given text
        block, for example: whether a text block contains an '@' (which could hint at the
        semantic role AUTHOR_INFO), or whether a text block is entirely written in uppercase
        characters.
        Returns a list of float lists, where the i-th float list includes the encoded character
        features of the i-th text block.

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks for which to encode the character features.
        Returns:
            List[List[float]]
                A list of float lists, where the i-th float list includes the encoded character
                features of the i-th text block.

        TODO
        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"])
        >>> char_features = dr.encode_character_features(blocks)
        >>> [[round(x, 2) for x in cf] for cf in char_features]
        ... #doctest: +NORMALIZE_WHITESPACE
        [[0.0, 0.0,  0.0,  0.1,  0.0,  1.0, 0.22], \
         [0.0, 0.0, 0.29,  0.0,  0.0,  0.5, 0.08], \
         [0.0, 0.0,  0.0,  0.0, 0.07, 0.33, 0.08], \
         [0.0, 0.0,  0.0,  0.0,  0.0,  1.0, 0.12], \
         [0.0, 0.0,  0.0,  0.0, 0.13,  1.0, 0.13], \
         [0.0, 0.0,  0.0,  0.0, 0.04, 0.33, 0.05], \
         [0.0, 0.0,  0.0, 0.09,  0.0,  1.0,  0.2], \
         [0.0, 0.0,  0.0,  0.0,  0.0,  1.0, 0.17], \
         [0.0, 1.0, 0.07,  0.0, 0.07,  0.5, 0.07], \
         [1.0, 0.0,  0.0,  0.0, 0.18,  0.0,  0.0]]
        """
        encoded_char_features = []

        for block in text_blocks:
            features = []
            text = block.text
            words = text.split()
            num_words = len(words)
            num_digits = sum([x.isdigit() for x in text])
            num_uppercased_chars = sum([x.isupper() for x in text])
            num_non_ascii_chars = sum([ord(x) > 127 for x in text])
            num_punctuations = sum([x in string.punctuation for x in text])
            num_non_spaces = sum([x not in {" ", "\t", "\n", "\v", "\f", "\r"} for x in text])
            num_capitalized_words = sum([x[0].isupper() for x in words])

            # The flag whether or not the text of the block contains a '@'.
            features.append(1.0 if '@' in text else 0.0)
            # The flag whether or not the text starts with a digit.
            features.append(1.0 if text and text[0].isdigit() else 0.0)
            # The percentage of digits in the text.
            features.append(num_digits / num_non_spaces if num_non_spaces > 0 else 0.0)
            # The percentage of non-ascii characters.
            features.append(num_non_ascii_chars / num_non_spaces if num_non_spaces > 0 else 0.0)
            # The percentage of punctuation characters.
            features.append(num_punctuations / num_non_spaces if num_non_spaces > 0 else 0.0)
            # The percentage of capitalized words.
            features.append(num_capitalized_words / num_words if num_words > 0 else 0.0)
            # The percentage of uppercased characters (ignoring whitespaces).
            features.append(num_uppercased_chars / num_non_spaces if num_non_spaces > 0 else 0.0)
            # The number of words. TODO: Normalize?
            # features.append(float(num_words))
            # The average word length. TODO: Normalize?
            # features.append(num_words_chars / num_words)

            encoded_char_features.append(features)

        return encoded_char_features

    # =============================================================================================

    def encode_semantic_roles(self, text_blocks: List[TextBlock],
                              roles_vocab: Dict[str, int]) -> List[List[int]]:
        """
        This method encodes the semantic roles of the given text blocks, by (1) translating
        each role to the integer value that is associated with the role in the given vocabulary and
        (2) encoding this integer by using one-hot encoding. It returns a list of integer lists,
        where the i-th integer list denotes the encoding of the semantic role of the i-th text
        block.

        Args:
            text_blocks: List[TextBlock]
                The list of text blocks for which to encode the semantic roles.
            roles_vocab: Dict[str, int]
                A dictionary that maps role names to integer values.
        Returns:
            List[List[int]]
                A list of integer lists where the i-th integer list denotes the encoding of the
                semantic role of the i-th text block.

        TODO
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab_file = os.path.join(base_dir, "./examples/vocab-roles.example.txt")
        >>> vocab = utils.files.read_vocabulary_file(vocab_file)
        >>> dr = DataReader(None, roles_vocab=vocab, logging_level="debug")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
        ...   os.path.join(base_dir, "./examples/article.4.tsv"),
        ...   os.path.join(base_dir, "./examples/vldb.2.tsv"),
        ...   os.path.join(base_dir, "./examples/sig-alternate.1.tsv")
        ... ])
        >>> encoded_roles, _ = dr.encode_semantic_roles(blocks, vocab)
        >>> encoded_roles
        ... #doctest: +NORMALIZE_WHITESPACE
        [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], \
         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0], \
         [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0], \
         [0, 0, 0, 0, 0, 1, 0]]
        >>> sorted(dr.roles_vocab.items(), key=lambda x: x[1])
        ... #doctest: +NORMALIZE_WHITESPACE
        [('AUTHOR_NAME', 0), ('PUBLICATION-DATE', 1), ('TITLE', 2), ('HEADING_HEADING', 3), \
         ('PARAGRAPHS', 4), ('AUTHOR_MAIL', 5), ('UNK_ROLE', 6)]
        >>> sorted(dr.rev_roles_vocab.items(), key=lambda x: x[0])
        ... #doctest: +NORMALIZE_WHITESPACE
        [(0, 'AUTHOR_NAME'), (1, 'PUBLICATION-DATE'), (2, 'TITLE'), (3, 'HEADING_HEADING'), \
         (4, 'PARAGRAPHS'), (5, 'AUTHOR_MAIL'), (6, 'UNK_ROLE')]
        """
        encoded_roles = []
        for block in text_blocks:
            # if block.role not in roles_vocab:
            #     LOG.warn(f"Roles vocabulary does not contain role '{block.role}'.")

            encoded_role = roles_vocab.get(block.role, -1)
            one_hot_encoded_role = [0] * len(roles_vocab)
            if encoded_role >= 0:
                one_hot_encoded_role[encoded_role] = 1
            encoded_roles.append(one_hot_encoded_role)

        return encoded_roles

    # =============================================================================================

    # def encode_semantic_features(self, text_blocks, countries_db_file, human_names_db_file):
    #     """
    #     Encodes some (simple) features about the semantics of words contained by a text block,
    #     for example: whether a word denotes the part of a human name, or a country name.
    #     Returns a list of list of encoded semantic features, where the i-th list includes the
    #     semantic features of the i-th text block.

    #     Args:
    #         text_blocks (list of textBlock):
    #             The list of text blocks for which the semantic features should be encoded.
    #         countries_db_file (str):
    #             The path to a file containing country names, in the format: one country per line.
    #         human_names_db_file (str):
    #             The path to a file containing human names, in the format: one human name per line.

    #     >>> base_dir = os.path.dirname(os.path.realpath(__file__))
    #     >>> countries_db_file = os.path.join(base_dir, "./examples/country-names.example.txt")
    #     >>> human_names_db_file = os.path.join(base_dir, "./examples/human-names.example.txt")

    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
    #     ...   "examples/article.4.tsv",
    #     ...   "examples/vldb.2.tsv",
    #     ...   "examples/sig-alternate.1.tsv"
    #     ... ])
    #     >>> dr.encode_semantic_features(blocks, countries_db_file, human_names_db_file)
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], \
    #      [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    #     """
    #     encoded_semantic_features = []

    #     country_names_index = None
    #     if countries_db_file:
    #         LOG.debug("Reading country names from '{0}' ...".format(countries_db_file))
    #         # Read the country names from file and create a name index from the country names.
    #         country_names = utils.files.read_lines(countries_db_file)
    #         country_names_index = self.create_name_index(country_names)

    #     human_names_index = None
    #     if human_names_db_file:
    #         LOG.debug("Reading human names from '{0}' ...".format(human_names_db_file))
    #         # Read the human names from file and create a name index from the human names.
    #         human_names = utils.files.read_lines(human_names_db_file)
    #         human_names_index = self.create_name_index(human_names)

    #     if country_names_index or human_names_index:
    #         for i, block in enumerate(text_blocks):
    #             # Translate the text of the text block to normalized names.
    #             words = self.name_to_normalized_words(block.text)

    #             features = []
    #             # The percentage of characters relating to a country name.
    #             num_countries_words = self.compute_name_word_coverage(words, country_names_index)
    #             features.append(num_countries_words / len(words) if len(words) > 0 else 0)

    #             # The percentage of characters relating to a human name.
    #             num_human_names_words = self.compute_name_word_coverage(words, human_names_index)
    #             features.append(num_human_names_words / len(words) if len(words) > 0 else 0)

    #             encoded_semantic_features.append(features)

    #     return encoded_semantic_features

    # def create_name_index(self, names):
    #     """
    #     Creates a name index from the given list of names. With name index, we mean a dictionary
    #     that maps each word sequence of a name starting at position 0 to a boolean value. The
    #     boolean value denotes whether the word sequence represents the original name (True) or a
    #     prefix of the name (False). This index can be used to identify names in a given text
    #     efficiently.

    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> sorted(dr.create_name_index([]).items())
    #     []
    #     >>> sorted(dr.create_name_index(["Papua-Neuguinea"]).items())
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [('Papua', False), ('Papua Neuguinea', True)]
    #     >>> sorted(dr.create_name_index(["Germany", "People's Republic of China"]).items())
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #   [('Germany', True), ('People', False), ('People s', False), ('People s Republic', False), \
    #      ('People s Republic of', False), ('People s Republic of China', True)]
    #     """
    #     name_index = {}
    #     for name in names:
    #         # Translate the name to normalized words.
    #         words = self.name_to_normalized_words(name)
    #         for i in range(len(words) - 1):
    #             text = " ".join(words[:i+1])
    #             if text not in name_index:
    #                 name_index[text] = False
    #         name_index[" ".join(words)] = True
    #     return name_index

    # def name_to_normalized_words(self, name):
    #     """
    #     Splits the given name (for example a country name or a human name) at each non-character
    #     smybol to words and translate each word to a title word (if it starts with a uppercased
    #     character).

    #     Args:
    #         name (str):
    #             The name to split into words.

    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> dr.name_to_normalized_words(None) is None
    #     True
    #     >>> dr.name_to_normalized_words("")
    #     []
    #     >>> dr.name_to_normalized_words("-+*")
    #     []
    #     >>> dr.name_to_normalized_words("Germany")
    #     ['Germany']
    #     >>> dr.name_to_normalized_words("GERMANY")
    #     ['Germany']
    #     >>> dr.name_to_normalized_words("germany")
    #     ['germany']
    #     >>> dr.name_to_normalized_words("GERMANY-")
    #     ['Germany']
    #     >>> dr.name_to_normalized_words("Papua-Neiguinea")
    #     ['Papua', 'Neiguinea']
    #     >>> dr.name_to_normalized_words("Papua - Neiguinea")
    #     ['Papua', 'Neiguinea']
    #     >>> dr.name_to_normalized_words("Papua - neiguinea")
    #     ['Papua', 'neiguinea']
    #     >>> dr.name_to_normalized_words("Papwazi-Nouvèl-Gine")
    #     ['Papwazi', 'Nouvèl', 'Gine']
    #     >>> dr.name_to_normalized_words("People's Republic of China")
    #     ['People', 's', 'Republic', 'of', 'China']
    #     >>> dr.name_to_normalized_words("PeOple's REPublic of CHINA")
    #     ['People', 's', 'Republic', 'of', 'China']
    #     """
    #     if name is None:
    #         return
    #     # Replace all non-characters by a whitespace and split the name at each whitespace.
    #     words = name.translate(self.non_chars_trans).split()
    #     # If the first character of a word is capitalized, lowercase all other characters.
    #     words = [w.title() if (w and w[0].isupper()) else w for w in words]
    #     return [w for w in words if w]

    # def compute_name_word_coverage(self, words, index):
    #     """
    #     Computes the number of words that refer to a name in the given name index.

    #     Args:
    #         words (list of str):
    #             The list of words to search for names.
    #         index (dict str:bool)
    #             The name index to use.

    #     >>> index = {"Germany": True, "Papua": False, "Papua Neiguinea": True, "People": False, \
    #                  "People s": False, "People s Republic": False, \
    #                  "People s Republic of": False, "People s Republic of China": True}
    #     >>> dr = DataReader(None, logging_level="fatal")

    #     >>> words = ["Germany"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     1
    #     >>> words = ["I", "live", "in", "Germany"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     1
    #     >>> words = ["I", "live", "in", "Germany", "and", "Papua", "Neiguinea"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     3
    #     >>> words = ["Papua", "Neiguixxx", "is", "great"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     0
    #   >>> words = ["The", "cities", "in", "People", "s", "Republic", "of", "China", "are", "big"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     5
    #     >>> words = ["The", "cities", "in", "People", "Republic", "of", "China", "are", "big"]
    #     >>> dr.compute_name_word_coverage(words, index)
    #     0
    #     """
    #     num_matched_words = 0

    #     if not index:
    #         return 0

    #     i = 0
    #     j = 0
    #     while j < len(words):
    #         text = " ".join(words[i:j+1])
    #         if text not in index:
    #             i = j + 1
    #             j = j + 1
    #             continue

    #         if index[text]:
    #             num_matched_words += (j + 1 - i)
    #             i = j + 1
    #             j = j + 1
    #             continue

    #         j = j + 1
    #     return num_matched_words


# =================================================================================================
