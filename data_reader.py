import os  # NOQA
import string
from typing import List

import numpy as np

import settings

import utils.encoding.bpe
import utils.files
import utils.log
import utils.word_tokenizer
import utils.zip

# =================================================================================================

# The logger.
LOG = utils.log.get_logger(__name__)

# =================================================================================================

# The symbol to use for an unknown role.
UNKNOWN_ROLE_SYMBOL = "UNK_ROLE"

# =================================================================================================


class DataReader:
    def __init__(self, input_dir=None, input_file_name_pattern="*", max_num_input_files=-1,
                 shuffle_input_files=True, encoding="bpe", words_vocab={}, roles_vocab={},
                 word_delimiters="!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ", lowercase_words=False,
                 num_words_per_seq=100, include_positions=True, include_font_sizes=True,
                 include_font_styles=True, include_char_features=True,
                 include_semantic_features=True, countries_db_file=None, human_names_db_file=None,
                 logging_level="info"):
        """
        Creates a new data reader, for reading TSV files (each providing some interesting features
        about the building blocks of a specific document) and bring the contents to the format as
        needed by the model(s) used to predict the semantic roles of the building blocks.

        Args:
            input_dir (str):
                The path to the directory to read the TSV files from.
            input_file_name_pattern (str):
                The file name pattern of the TSV files in the input directory, for example: "*",
                or "*.tsv", or "vldb*.tsv".
            max_num_input_files (int):
                The maximum number of TSV files to read from the input directory. Set this value
                to -1 to read *all* input files matching the given file name pattern.
            shuffle_input_files (bool):
                Whether or not the TSV files should be shuffled before selecting a subset of
                them. If this value is set to True and <max_num_input_files> is set to 10, the
                reader will select 10 *random* TSV files matching the given file name pattern.
                If set to False, the reader will always select the first 10 files in the directory
                (or to be more precise: the first 10 files of the files returned by Python's
                directory reader).
            encoding (str):
                The name of the encoding to use to encode the word sequences.
            words_vocab (dict str:int):
                The vocabulary for encoding the word sequences, mapping tokens to ids.
            roles_vocab (dict str:int):
                The vocabulary for encoding the semantic roles, mapping roles to ids.
            word_delimiters (str or list of str):
                The characters to consider as word delimiters on splitting the texts of building
                blocks into words. For example, if set to '-+', the text 'foo-bar+baz boo' will be
                split into the words 'foo', 'bar' and 'baz boo' ('baz boo' won't be split, because
                the character ' ' is not part of 'word_delimiters').
            lowercase_words (bool):
                The boolean flag that indicates whether or not the words should be lowercased.
            num_words_per_seq (int):
                The number of words to include in the input sequence for a building block. If a
                building block contains less words, the sequence will be padded. If the building
                block contains more words, the sequence will be truncated (that is: only the first
                <num_words_per_seq>-many words of each sequence will be included in the sequence).
                If set to -1, *all* words will be included and all sequences will be padded to the
                length of the longest sequence.
            include_positions (bool):
                The boolean flag that indicates whether or not the positions of the building blocks
                should be included in the input sequences.
            include_font_sizes (bool):
                The boolean flag that indicates whether or not the font sizes of the building
                blocks should be included in the input sequences.
            include_font_styles (bool):
                The boolean flag that indicates whether or not the font styles (= the information
                whether a block is printed in bold and/or italic) of the building blocks should be
                included in the input sequences.
            include_char_features (bool):
                The boolean flag that indicates whether or not features about the characters
                contained by a building block (e.g., whether or not the block contains a '@', which
                could hint at the role 'AUTHOR_MAIL', or the number of uppercased characters)
                should be included in the input sequences.
            include_semantic_features (bool):
                The boolean flag that indicates whether or not some (simple) semantic features
                about the words in a building block (e.g., whether a word denotes the name of a
                human or a country) should be included in the input sequences.
            countries_db_file (str):
                The path to a file providing a set of country names, in the format: one country per
                line. These names are used to determine if a given word belongs to a country name
                or not (see also the information given for parameter 'include_semantic_features').
            human_names_db_file (str):
                The path to a file providing human names, in the format: one name per line.
                These names are used to determine if a given word belongs to a human name or not
                (see also the information given for parameter 'include_semantic_features').
            logging_level (str):
                The logging level.
        """
        self.input_dir = input_dir
        self.input_file_name_pattern = input_file_name_pattern
        self.max_num_input_files = max_num_input_files
        self.shuffle_input_files = shuffle_input_files
        self.encoding = encoding
        self.words_vocab = words_vocab
        self.rev_words_vocab = {v: k for k, v in words_vocab.items()}
        self.roles_vocab = roles_vocab
        # Add the symbol for an unknown role to the vocab.
        self.roles_vocab[UNKNOWN_ROLE_SYMBOL] = len(self.roles_vocab)
        self.rev_roles_vocab = {v: k for k, v in roles_vocab.items()}
        self.word_tokenizer = utils.word_tokenizer.WordTokenizer(word_delimiters)
        self.lowercase_words = lowercase_words
        self.num_words_per_seq = num_words_per_seq
        self.include_positions = include_positions
        self.include_font_sizes = include_font_sizes
        self.include_font_styles = include_font_styles
        self.include_char_features = include_char_features
        self.include_semantic_features = include_semantic_features
        self.countries_db_file = countries_db_file
        self.human_names_db_file = human_names_db_file
        # The percentage distribution of the semantic roles.
        self.roles_dist = {}

        # The building blocks, as defined in the input files.
        self.building_blocks = []
        # The word sequences of the building blocks, in the format as needed by the model.
        self.word_sequences = np.array([])
        # The auxiliary features (like position and font information, etc.) of the building blocks.
        self.features = np.array([])
        # The roles of the building blocks, in the format as needed by the model.
        self.roles = np.array([])

        # A dict mapping each non-char symbol to a whitespace.
        self.non_chars_trans = str.maketrans({c: " " for c in settings.NON_CHAR_SYMBOLS})

        # Configure the logging level.
        LOG.setLevel(utils.log.to_log_level(logging_level))

    # =============================================================================================

    def read(self):
        """
        Parses the given input directory for TSV files (each providing some interesting features
        about the building blocks of a specific document), reads them and returns:
        (1) an array of sequences consisting of the words of the building blocks (one
        sequence per building block), to be used as input vectors for a model to predict the
        semantic roles of building blocks.
        (2) an array of sequences consisting of some auxiliary features about the building blocks
        (one sequence per building block), for example: the position, the font information, etc.;
        to be used as another input vectors.
        (3) an array of the semantic roles of the building blocks; to be used as the output vector
        for a model to predict the semantic roles of building blocks.

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
        >>> dr.building_blocks
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
        # Parse the directory for the TSV files.
        LOG.info("Parsing directory '{}' for ground truth files ...".format(self.input_dir))
        input_files = utils.files.parse_dir(
            directory=self.input_dir,
            pattern=self.input_file_name_pattern,
            shuffle_files=self.shuffle_input_files,
            max_num_files=self.max_num_input_files
        )
        LOG.info("Found {:,} ground truth files.".format(len(input_files)))

        if len(input_files) == 0:
            return self.word_sequences, self.features, self.roles

        # Read the building blocks from the input files.
        self.building_blocks, pages, min_font_sizes, max_font_sizes = \
            utils.files.read_groundtruth_files(input_files)

        LOG.info("Splitting the texts into word sequences ...")
        word_seqs = self.split_into_word_sequences(self.building_blocks, self.lowercase_words)

        LOG.info("Encoding the word sequences ...")
        word_seqs = self.encode_word_sequences(word_seqs, self.encoding, self.words_vocab,
                                               self.num_words_per_seq)

        # Initialize the array of layout features (one empty sequence per building block).
        features = [[] for _ in self.building_blocks]

        if self.include_positions:
            LOG.info("Encoding the positions ...")
            encoded_positions = self.encode_positions(self.building_blocks, pages)
            features = utils.zip.zip_lists(features, encoded_positions)

        if self.include_font_sizes:
            LOG.info("Encoding the font sizes ...")
            encoded_font_sizes = self.encode_font_sizes(
                self.building_blocks, min_font_sizes, max_font_sizes)
            features = utils.zip.zip_lists(features, encoded_font_sizes)

        if self.include_font_styles:
            LOG.info("Encoding the font styles ...")
            encoded_font_styles = self.encode_font_styles(self.building_blocks)
            features = utils.zip.zip_lists(features, encoded_font_styles)

        if self.include_char_features:
            LOG.info("Encoding the character features ...")
            encoded_char_features = self.encode_character_features(self.building_blocks)
            features = utils.zip.zip_lists(features, encoded_char_features)

        if self.include_semantic_features:
            LOG.info("Encoding the semantic features ...")
            encoded_semantic_features = self.encode_semantic_features(
                self.building_blocks, self.countries_db_file, self.human_names_db_file)
            features = utils.zip.zip_lists(features, encoded_semantic_features)

        LOG.info("Encoding the roles ...")
        roles, self.roles_dist = self.encode_semantic_roles(self.building_blocks, self.roles_vocab)

        # Translate the lists to numpy arrays.
        self.word_sequences = np.array(word_seqs)
        assert not np.any(np.isnan(self.word_sequences))

        self.features = np.array(features)
        assert not np.any(np.isnan(self.features))

        self.roles = np.array(roles)
        assert not np.any(np.isnan(self.roles))

        return self.word_sequences, self.features, self.roles

    def from_doc_building_blocks(self, building_blocks: List[utils.files.BuildingBlock], \
          pages: List[utils.files.Page]):
        min_font_size = float('inf')
        max_font_size = 0
        for block in building_blocks:
            min_font_size = min(min_font_size, block.font_size)
            max_font_size = max(max_font_size, block.font_size)
        pages = [pages]
        min_font_sizes = [min_font_size]
        max_font_sizes = [max_font_size]

        LOG.info("Splitting the texts into word sequences ...")
        word_seqs = self.split_into_word_sequences(building_blocks, self.lowercase_words)

        LOG.info("Encoding the word sequences ...")
        word_seqs = self.encode_word_sequences(word_seqs, self.encoding, self.words_vocab,
                                               self.num_words_per_seq)

        # Initialize the array of layout features (one empty sequence per building block).
        features = [[] for _ in building_blocks]

        if self.include_positions:
            LOG.info("Encoding the positions ...")
            encoded_positions = self.encode_positions(building_blocks, pages)
            features = utils.zip.zip_lists(features, encoded_positions)

        if self.include_font_sizes:
            LOG.info("Encoding the font sizes ...")
            encoded_font_sizes = self.encode_font_sizes(
                building_blocks, min_font_sizes, max_font_sizes)
            features = utils.zip.zip_lists(features, encoded_font_sizes)

        if self.include_font_styles:
            LOG.info("Encoding the font styles ...")
            encoded_font_styles = self.encode_font_styles(building_blocks)
            features = utils.zip.zip_lists(features, encoded_font_styles)

        if self.include_char_features:
            LOG.info("Encoding the character features ...")
            encoded_char_features = self.encode_character_features(building_blocks)
            features = utils.zip.zip_lists(features, encoded_char_features)

        if self.include_semantic_features:
            LOG.info("Encoding the semantic features ...")
            encoded_semantic_features = self.encode_semantic_features(
                building_blocks, self.countries_db_file, self.human_names_db_file)
            features = utils.zip.zip_lists(features, encoded_semantic_features)

        LOG.info("Encoding the roles ...")
        roles, self.roles_dist = self.encode_semantic_roles(building_blocks, self.roles_vocab)

        # Translate the lists to numpy arrays.
        self.word_sequences = np.array(word_seqs)
        assert not np.any(np.isnan(self.word_sequences))

        self.features = np.array(features)
        assert not np.any(np.isnan(self.features))

        self.roles = np.array(roles)
        assert not np.any(np.isnan(self.roles))

        return self.word_sequences, self.features, self.roles

    # =============================================================================================

    def split_into_word_sequences(self, building_blocks, lowercase=False):
        """
        Splits the texts of the given building blocks at the given word delimiters into word
        sequences. Lowercases all words if 'lowercase' is set to True. Returns a list of word
        sequences, where the i-th sequence is the word sequence of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks to split into word sequences.
            lowercase (bool):
                The boolean flag that indicates whether to lowercase the words.

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

        for block in building_blocks:
            # Split the text into words.
            words = self.word_tokenizer.tokenize_into_words(block.text)

            if lowercase:
                # Lowercase the text.
                words = [w.lower() for w in words]

            word_seqs.append(words)

        return word_seqs

    # =============================================================================================

    def encode_word_sequences(self, seqs, encoding_type, vocab=None, num_words_per_seq=100):
        """
        Encodes the given word sequences using the given encoding. Returns the list of encoded
        word sequences, where each list is of length <num_words_per_seq> and the i-th sequence is
        the encoded version of the i-th input sequence.

        Args:
            seqs (list of list of str):
                The list of word sequences to encode.
            encoding_type (str):
                The type of the encoding to use.
            vocab (dict str:int):
                The vocabulary for encoding the word sequences.
            num_words_per_seq (int):
                The number of words each sequence should consist of. Word sequences of length
                larger than this value will be truncated, word sequences of length smaller than
                this value will be padded.
        """

        if encoding_type == "bpe":
            # Use byte pair encoding.
            encoder = utils.encoding.bpe.BytePairEncoder(vocab)

        return [encoder.encode(seq, num_words_per_seq) for seq in seqs]

    # def encode_word_sequences(self, word_seqs, top_k_words=-1):
    #     """
    #     Encodes the given word sequences, that is: assigns the same unique integer id to the same
    #     words. If 'top_k_words' is set to a value >= 0, only the <top_k_words> most common words
    #    re encoded, all other words will not be encoded and won't be part of the returned list. If
    #     'top_k_words' is set to -1, *all* words will be encoded. Returns (1) the list of encoded
    #    word sequences, where the i-th sequence is the encoded version of the i-th input sequence;
    #     (2) the word index, mapping each encoded word to the assigned id; and (3) the reversed
    #     version of the word index, mapping each id to the corresponding word.
    #
    #     Args:
    #         word_seqs (list of list of str):
    #             The word sequences to encode.
    #         top_k_words (int):
    #             If set to a value >= 0, only the <top_k_words> most common words will be encoded,
    #            all other words will not be encoded and won't be part of the returned list. If set
    #           to -1, *all* words will be encoded and will be part of the returned word sequences.
    #
    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> blocks, _, _ = dr.read_input_files(["examples/article.4.tsv", "examples/vldb.2.tsv"])
    #     >>> word_seqs = dr.split_into_word_sequences(blocks)
    #     >>> encoded_word_seqs = dr.encode_word_sequences(word_seqs)
    #     >>> encoded_word_seqs
    #     [[4, 2], [5, 6], [3, 7, 8], [9], [10], [3, 11, 12], [13, 2], [14, 15]]
    #     >>> sorted(dr.words_index.items(), key=lambda x: x[1])
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [('$PAD', 0), ('$UNK_WORD', 1), ('Müller', 2), ('A', 3), ('Ben', 4), ('September', 5), \
    #      ('2017', 6), ('catchy', 7), ('title!', 8), ('Abstract', 9), ('Bullshit-Bingo.', 10), \
    #      ('comprehensive', 11), ('survey.', 12), ('Anne', 13), ('1', 14), ('Introduction.', 15)]
    #     >>> sorted(dr.index_words.items(), key=lambda x: x[0])
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [(0, '$PAD'), (1, '$UNK_WORD'), (2, 'Müller'), (3, 'A'), (4, 'Ben'), (5, 'September'), \
    #      (6, '2017'), (7, 'catchy'), (8, 'title!'), (9, 'Abstract'), (10, 'Bullshit-Bingo.'), \
    #      (11, 'comprehensive'), (12, 'survey.'), (13, 'Anne'), (14, '1'), (15, 'Introduction.')]
    #
    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> blocks, _, _ = dr.read_input_files(["examples/article.4.tsv", "examples/vldb.2.tsv"])
    #     >>> word_seqs = dr.split_into_word_sequences(blocks)
    #     >>> encoded_word_seqs = dr.encode_word_sequences(word_seqs, 2)
    #     >>> encoded_word_seqs
    #     [[2], [], [3], [], [], [3], [2], []]
    #     >>> sorted(dr.words_index.items(), key=lambda x: x[1])
    #     [('$PAD', 0), ('$UNK_WORD', 1), ('Müller', 2), ('A', 3)]
    #     >>> sorted(dr.index_words.items(), key=lambda x: x[0])
    #     [(0, '$PAD'), (1, '$UNK_WORD'), (2, 'Müller'), (3, 'A')]
    #     """
    #
    #     if not self.words_index:
    #         # Compute the word frequencies.
    #         word_freqs = {}  # word -> freq.
    #         for word_seq in word_seqs:
    #             for word in word_seq:
    #                 word_freqs[word] = word_freqs[word] + 1 if word in word_freqs else 1
    #
    #         # Compute the '<top_k_words>'-most common words.
    #         word_freqs_list = sorted(list(word_freqs.items()), key=lambda x: x[1], reverse=True)
    #         if top_k_words >= 0:
    #             word_freqs_list = word_freqs_list[:top_k_words]
    #
    #         # Create the word index mapping each of the <top_k_words>-most common words to an id.
    #         self.words_index = {UNK_WORD_STR: UNK_WORD_ID, PADDING_WORD_STR: PADDING_WORD_ID}
    #         for word, freq in word_freqs_list:
    #             self.words_index[word] = len(self.words_index)
    #         # Create the reversed word index.
    #         self.index_words = {v: k for k, v in self.words_index.items()}
    #
    #     # Translate each word in each sequence to the corresponding id in the word index.
    #     encoded_word_seqs = []
    #     for word_seq in word_seqs:
    #         encoded_word_seq = []
    #         for word in word_seq:
    #             if word in self.words_index:
    #                 encoded_word_seq.append(self.words_index[word])
    #         encoded_word_seqs.append(encoded_word_seq)
    #
    #     return encoded_word_seqs

    # =============================================================================================

    def encode_positions(self, building_blocks, pages):
        """
        Encodes the positions of the given building blocks (for each: the page number and the x/y-
        coordinates of the midpoint). Normalizes the page number by dividing it by the maximum page
        number in the belonging document and normalizes the x/y coordinates by dividing them by the
        width/height of the belonging page in the document. Returns a list of encoded positions,
        where the element at the i-th position is the encoded position of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the positions should be encoded.
            pages (list of list of Page):
                The page objects; pages[i][j] should contain the metadata of page j in document i.

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

        # Compute the max page number per document.
        max_page_nums = []
        for doc_pages in pages:
            page_nums = [x.page_num for x in doc_pages]
            if page_nums:
                max_page_nums.append(max(page_nums))
            else:
                max_page_nums.append(0)

        for block in building_blocks:
            positions = []

            # Encode the page number.
            max_page_num = max_page_nums[block.doc_index]
            if max_page_num > 1:
                # Normalize the 1-based page numbers.
                positions.append((block.page_num - 1) / (max_page_num - 1))
            else:
                positions.append(0.0)


            page_width = pages[block.doc_index][block.page_num - 1].width
            page_height = pages[block.doc_index][block.page_num - 1].height

            # Encode the x/y-coordinates of the lower left.
            positions.append((block.lower_left_x / page_width) if page_width > 0 else 0.0)
            positions.append((block.lower_left_y / page_height) if page_height > 0 else 0.0)

            # # Encode the x/y-coordinates of the upper right.
            positions.append((block.upper_right_x / page_width) if page_width > 0 else 0.0)
            positions.append((block.upper_right_y / page_height) if page_height > 0 else 0.0)

            encoded_positions.append(positions)

        return encoded_positions

    # =============================================================================================

    # def encode_font_names(self, building_blocks):
    #     """
    #     Encodes the font names of the given building blocks, that is: assigns the same unique
    #     integer id to the same font names. Returns (1) the list of encoded font names, where the
    #     i-th element is the encoded font name of the i-th building block; (2) the font name index,
    #     mapping each encoded font name to its assigned id; and (3) the reversed version of the font
    #     name index, mapping each id to the corresponding font name.

    #     Args:
    #         building_blocks (list of BuildingBlock):
    #             The list of building blocks for which the font names should be encoded.

    #     >>> dr = DataReader(None, logging_level="fatal")
    #     >>> blocks, _, _ = dr.read_input_files(["examples/article.4.tsv", "examples/vldb.2.tsv"])

    #     >>> encoded_font_names = dr.encode_font_names(blocks)
    #     >>> encoded_font_names
    #     [1, 1, 2, 1, 1, 3, 2, 1]
    #     >>> sorted(dr.fonts_index.items(), key=lambda x: x[1])
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [('$UNK_FONT', 0), ('lmroman', 1), ('arial', 2), ('nimbussanl', 3)]
    #     >>> sorted(dr.index_fonts.items(), key=lambda x: x[0])
    #     ... #doctest: +NORMALIZE_WHITESPACE
    #     [(0, '$UNK_FONT'), (1, 'lmroman'), (2, 'arial'), (3, 'nimbussanl')]
    #     """

    #     if not self.fonts_index:
    #         # Build the fonts index.
    #         self.fonts_index = {UNK_FONT_STR: UNK_FONT_ID}
    #         for block in building_blocks:
    #             # Encode the font name.
    #             # TODO: Fonts can have same ids as words?
    #             if block.font_name not in self.fonts_index:
    #                 self.fonts_index[block.font_name] = len(self.fonts_index)
    #         self.index_fonts = {v: k for k, v in self.fonts_index.items()}

    #     encoded_font_names = []
    #     for block in building_blocks:
    #         encoded_font_name = self.fonts_index.get(block.font_name, UNK_FONT_ID)
    #         encoded_font_names.append(encoded_font_name)

    #     return encoded_font_names

    # =============================================================================================

    def encode_font_sizes(self, building_blocks, min_font_sizes, max_font_sizes):
        """
        Encodes the font sizes of the given building blocks. Translates the font sizes to a value
        between 0 (= the minimum fon in the document) and 1 (= the maximum font size in the
        document). Returns a list of encoded font sizes, where the element at the i-th position is
        the encoded font size of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the font sizes should be encoded.
            min_font_sizes (list of int):
                The minimum font sizes in each document; the i-th element should be the minimum
                font size of the i-th document.
            max_font_sizes (list of int):
                The maximum font sizes in each document; the i-th element should be the maximum
                font size of the i-th document.

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

        for block in building_blocks:
            # Use the whole interval [0, 1], that is: translate the min font size to 0 and the max
            # font size to 1. For example, if the min font size in a document is 8 and the max font
            # size is 10, then translate font size 10 to: (10 - 8) / (12 - 8) = 0.5
            min_fs = min_font_sizes[block.doc_index]
            max_fs = max_font_sizes[block.doc_index]
            if min_fs == max_fs:
                encoded_font_sizes.append(0.0)
            else:
                encoded_font_sizes.append((block.font_size - min_fs) / (max_fs - min_fs))

        return encoded_font_sizes

    # =============================================================================================

    def encode_font_styles(self, building_blocks):
        """
        Encodes the font styles (the information whether a text is printed in bold and/or italic)
        of the given building blocks. Returns a list of encoded font styles, where the i-th element
        is the encoded font style of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the font styles should be encoded.

        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"])
        >>> dr.encode_font_styles(blocks)
        [[0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0]]
        """
        encoded_font_styles = []

        for block in building_blocks:
            # Append the 'isBold' flag and 'isItalic' flag (both 0 or 1).
            encoded_font_styles.append([int(block.is_bold), int(block.is_italic)])

        return encoded_font_styles

    # =============================================================================================

    def encode_character_features(self, building_blocks):
        """
        Encodes some features about the characters contained by a building block, for example:
        whether a building block contains an '@' (which could hint at the semantic role
        AUTHORS_MAIL), or whether a building block is entirely written in uppercased characters.
        Returns a list of list of encoded character features, where the i-th list includes the
        character features of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the character features should be encoded.

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

        for block in building_blocks:
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

    def encode_semantic_features(self, building_blocks, countries_db_file, human_names_db_file):
        """
        Encodes some (simple) features about the semantics of words contained by a building block,
        for example: whether a word denotes the part of a human name, or a country name.
        Returns a list of list of encoded semantic features, where the i-th list includes the
        semantic features of the i-th building block.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the semantic features should be encoded.
            countries_db_file (str):
                The path to a file containing country names, in the format: one country per line.
            human_names_db_file (str):
                The path to a file containing human names, in the format: one human name per line.

        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> countries_db_file = os.path.join(base_dir, "./examples/country-names.example.txt")
        >>> human_names_db_file = os.path.join(base_dir, "./examples/human-names.example.txt")

        >>> dr = DataReader(None, logging_level="fatal")
        >>> blocks, _, _, _ = utils.files.read_groundtruth_files([
        ...   "examples/article.4.tsv",
        ...   "examples/vldb.2.tsv",
        ...   "examples/sig-alternate.1.tsv"
        ... ])
        >>> dr.encode_semantic_features(blocks, countries_db_file, human_names_db_file)
        ... #doctest: +NORMALIZE_WHITESPACE
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], \
         [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        """
        encoded_semantic_features = []

        country_names_index = None
        if countries_db_file:
            LOG.debug("Reading country names from '{0}' ...".format(countries_db_file))
            # Read the country names from file and create a name index from the country names.
            country_names = utils.files.read_lines(countries_db_file)
            country_names_index = self.create_name_index(country_names)

        human_names_index = None
        if human_names_db_file:
            LOG.debug("Reading human names from '{0}' ...".format(human_names_db_file))
            # Read the human names from file and create a name index from the human names.
            human_names = utils.files.read_lines(human_names_db_file)
            human_names_index = self.create_name_index(human_names)

        if country_names_index or human_names_index:
            for i, block in enumerate(building_blocks):
                # Translate the text of the building block to normalized names.
                words = self.name_to_normalized_words(block.text)

                features = []
                # The percentage of characters relating to a country name.
                num_countries_words = self.compute_name_word_coverage(words, country_names_index)
                features.append(num_countries_words / len(words) if len(words) > 0 else 0)

                # The percentage of characters relating to a human name.
                num_human_names_words = self.compute_name_word_coverage(words, human_names_index)
                features.append(num_human_names_words / len(words) if len(words) > 0 else 0)

                encoded_semantic_features.append(features)

        return encoded_semantic_features

    def create_name_index(self, names):
        """
        Creates a name index from the given list of names. With name index, we mean a dictionary
        that maps each word sequence of a name starting at position 0 to a boolean value. The
        boolean value denotes whether the word sequence represents the original name (True) or a
        prefix of the name (False). This index can be used to identify names in a given text
        efficiently.

        >>> dr = DataReader(None, logging_level="fatal")
        >>> sorted(dr.create_name_index([]).items())
        []
        >>> sorted(dr.create_name_index(["Papua-Neuguinea"]).items())
        ... #doctest: +NORMALIZE_WHITESPACE
        [('Papua', False), ('Papua Neuguinea', True)]
        >>> sorted(dr.create_name_index(["Germany", "People's Republic of China"]).items())
        ... #doctest: +NORMALIZE_WHITESPACE
        [('Germany', True), ('People', False), ('People s', False), ('People s Republic', False), \
         ('People s Republic of', False), ('People s Republic of China', True)]
        """
        name_index = {}
        for name in names:
            # Translate the name to normalized words.
            words = self.name_to_normalized_words(name)
            for i in range(len(words) - 1):
                text = " ".join(words[:i+1])
                if text not in name_index:
                    name_index[text] = False
            name_index[" ".join(words)] = True
        return name_index

    def name_to_normalized_words(self, name):
        """
        Splits the given name (for example a country name or a human name) at each non-character
        smybol to words and translate each word to a title word (if it starts with a uppercased
        character).

        Args:
            name (str):
                The name to split into words.

        >>> dr = DataReader(None, logging_level="fatal")
        >>> dr.name_to_normalized_words(None) is None
        True
        >>> dr.name_to_normalized_words("")
        []
        >>> dr.name_to_normalized_words("-+*")
        []
        >>> dr.name_to_normalized_words("Germany")
        ['Germany']
        >>> dr.name_to_normalized_words("GERMANY")
        ['Germany']
        >>> dr.name_to_normalized_words("germany")
        ['germany']
        >>> dr.name_to_normalized_words("GERMANY-")
        ['Germany']
        >>> dr.name_to_normalized_words("Papua-Neiguinea")
        ['Papua', 'Neiguinea']
        >>> dr.name_to_normalized_words("Papua - Neiguinea")
        ['Papua', 'Neiguinea']
        >>> dr.name_to_normalized_words("Papua - neiguinea")
        ['Papua', 'neiguinea']
        >>> dr.name_to_normalized_words("Papwazi-Nouvèl-Gine")
        ['Papwazi', 'Nouvèl', 'Gine']
        >>> dr.name_to_normalized_words("People's Republic of China")
        ['People', 's', 'Republic', 'of', 'China']
        >>> dr.name_to_normalized_words("PeOple's REPublic of CHINA")
        ['People', 's', 'Republic', 'of', 'China']
        """
        if name is None:
            return
        # Replace all non-characters by a whitespace and split the name at each whitespace.
        words = name.translate(self.non_chars_trans).split()
        # If the first character of a word is capitalized, lowercase all other characters.
        words = [w.title() if (w and w[0].isupper()) else w for w in words]
        return [w for w in words if w]

    def compute_name_word_coverage(self, words, index):
        """
        Computes the number of words that refer to a name in the given name index.

        Args:
            words (list of str):
                The list of words to search for names.
            index (dict str:bool)
                The name index to use.

        >>> index = {"Germany": True, "Papua": False, "Papua Neiguinea": True, "People": False, \
                     "People s": False, "People s Republic": False, \
                     "People s Republic of": False, "People s Republic of China": True}
        >>> dr = DataReader(None, logging_level="fatal")

        >>> words = ["Germany"]
        >>> dr.compute_name_word_coverage(words, index)
        1
        >>> words = ["I", "live", "in", "Germany"]
        >>> dr.compute_name_word_coverage(words, index)
        1
        >>> words = ["I", "live", "in", "Germany", "and", "Papua", "Neiguinea"]
        >>> dr.compute_name_word_coverage(words, index)
        3
        >>> words = ["Papua", "Neiguixxx", "is", "great"]
        >>> dr.compute_name_word_coverage(words, index)
        0
        >>> words = ["The", "cities", "in", "People", "s", "Republic", "of", "China", "are", "big"]
        >>> dr.compute_name_word_coverage(words, index)
        5
        >>> words = ["The", "cities", "in", "People", "Republic", "of", "China", "are", "big"]
        >>> dr.compute_name_word_coverage(words, index)
        0
        """
        num_matched_words = 0

        if not index:
            return 0

        i = 0
        j = 0
        while j < len(words):
            text = " ".join(words[i:j+1])
            if text not in index:
                i = j + 1
                j = j + 1
                continue

            if index[text]:
                num_matched_words += (j + 1 - i)
                i = j + 1
                j = j + 1
                continue

            j = j + 1
        return num_matched_words

    # =============================================================================================

    def encode_semantic_roles(self, building_blocks, vocab):
        """
        Encodes the semantic roles of the given building blocks, that is: assigns the same unique
        integer id to the same role. Returns (1) a list of encode roles, where the i-th element is
        the encoded role of the i-th building block; (2) the role index, mapping each encoded role
        to the assigned id; and (3) the reversed version of the role index, mapping each id to the
        corresponding role.

        Args:
            building_blocks (list of BuildingBlock):
                The list of building blocks for which the semantic roles should be encoded.
            vocabulary (dict str:int)
                The vocabulary to use on encoding the roles.

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
        # Count the total number of roles.
        roles_dist = {}
        num_roles = 0

        # Encode the roles.
        encoded_roles = []
        for block in building_blocks:
            role = block.role

            encoded_role = vocab.get(role) if role in vocab else vocab.get(UNKNOWN_ROLE_SYMBOL)

            one_hot_encoded_role = [0] * len(vocab)
            one_hot_encoded_role[encoded_role] = 1
            encoded_roles.append(one_hot_encoded_role)

            if role in roles_dist:
                roles_dist[role] += 1
            else:
                roles_dist[role] = 1
            num_roles += 1

        roles_dist = {k: v / num_roles for k, v in roles_dist.items()}

        return encoded_roles, roles_dist

# =================================================================================================
