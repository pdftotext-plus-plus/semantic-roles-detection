"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains code to read
and write vocabulary files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

from typing import Dict

from semantic_roles_detection.utils import log_utils

# ==================================================================================================
# Parameters.

# The logger.
LOG = log_utils.get_logger(__name__)

# ==================================================================================================


class VocabularyReader:
    """
    This class reads given vocabulary files and returns them in form of dictionaries.
    """

    def read(self, file_path: str) -> Dict[str, int]:
        """
        Reads the given vocabulary file.

        Args:
            file_path: str
                The path to the vocabulary file to read.
        Returns:
            Dict[str, int]
                The content of the vocabulary file in form of a dictionary.

        TODO
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> vocab = read_vocabulary_file(os.path.join(base_dir, "../examples/vocab-words.examtxt"))
        >>> vocab_list = sorted(vocab.items(), key=lambda x: x[1])
        >>> len(vocab_list)
        281
        >>> vocab_list[:25] #doctest: +NORMALIZE_WHITESPACE
        [('!', 0), ('"', 1), ('#', 2), ('$', 3), ('%', 4), ('&', 5), ("'", 6), ('(', 7), (')', 8), \
        ('*', 9), ('+', 10), (',', 11), ('-', 12), ('.', 13), ('/', 14), ('0', 15), ('1', 16), \
        ('2', 17), ('3', 18), ('4', 19), ('5', 20), ('6', 21), ('7', 22), ('8', 23), ('9', 24)]
        >>> vocab_list[-25:] #doctest: +NORMALIZE_WHITESPACE
        [('ll', 256), ('er', 257), ('er✂', 258), ('ller✂', 259), ('e✂', 260), ('en', 261), \
        ('Mü', 262), ('Müller✂', 263), ('A✂', 264), ('y✂', 265), ('ti', 266), ('tr', 267), \
        ('t✂', 268), ('co', 269), ('com', 270), ('Ben', 271), ('Ben✂', 272), ('Se', 273), \
        ('Sep', 274), ('Sept', 275), ('Septe', 276), ('Septem', 277), ('Septemb', 278), \
        ('September✂', 279), ('20', 280)]
        """
        vocab = {}
        with open(file_path, "r") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    key, value = line.split("\t")
                    vocab[key] = int(value)
        return vocab

# ==================================================================================================


class VocabularyWriter:
    """
    This class writes given vocabularies to files.
    """

    def write(self, vocab: Dict[str, int], file_path: str):
        """
        This method writes the given vocabulary to the given file.

        Args:
            vocab: Dict[str, int]
                The vocabulary to write to file.
            file_path: str
                The path to the file to which the vocabulary should be written.

        TODO
        >>> vocab = {"x": 0, "y": 1, "xy": 2}
        >>> base_dir = os.path.dirname(os.path.realpath(__file__))
        >>> path = os.path.join(base_dir, "../examples/example-vocab.tmp.txt")
        >>> write_vocabulary_file(vocab, path)
        >>> open(path).read()
        'x\\t0\\ny\\t1\\nxy\\t2\\n'
        >>> os.remove(path)
        """
        with open(file_path, "w") as stream:
            for word, id in vocab.items():
                stream.write("{}\t{}\n".format(word, id))
