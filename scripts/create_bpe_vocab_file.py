"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains a script that
(1) parses the ground truth files for the texts of the contained text blocks, (2) creates a
vocabulary from the texts of the text blocks, as needed for encoding word sequences with byte pair
encoding, and (3) writes the vocabulary to a given file.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse
import sys
from typing import List, Union

sys.path.append("../src/semantic_roles_detection")  # Needed so that utils.* can be found.  # NOQA
from utils import bpe
from utils import ground_truth_reader
from utils import log_utils
from utils import vocab_utils
from utils import word_tokenizer
from utils.models import Document

# =================================================================================================
# Configure the logging.

LOG = log_utils.get_logger(__name__)

# =================================================================================================
# Parameters.

# The default logging level.
LOGGING_LEVEL = "debug"
# The prefix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_PREFIX = ""
# The suffix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_SUFFIX = ".tsv"
# The symbols to be considered as word delimiters on splitting the text of text blocks into words.
WORD_DELIMITERS = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n "
# The number of Unicode characters with which the vocabulary should be initialized.
NUM_INITIAL_CHARS = 256
# The number of byte pairs to merge while creating the vocabulary.
NUM_MERGES = 2000

# =================================================================================================


def main(ground_truth_dir: str,
         output_file_path: str,
         ground_truth_files_prefix: str = GROUND_TRUTH_FILES_PREFIX,
         ground_truth_files_suffix: str = GROUND_TRUTH_FILES_SUFFIX,
         word_delimiters: Union[str, list] = WORD_DELIMITERS,
         num_initial_chars: int = NUM_INITIAL_CHARS,
         num_merges: int = NUM_MERGES):
    """
    This method is the main method for creating a BPE vocabulary.

    Args:
        ground_truth_dir: str
            The path to the directory from which to read the ground truth files.
        output_file_path: str
            The path to the file to which to write the vocabulary.
        ground_truth_files_prefix: str
            The prefix of the files to read from the ground truth directory.
        ground_truth_files_suffix: str
            The suffix of the files to read from the ground truth directory.
        word_delimiters: Union[str, list]
            The symbols to be considered as word delimiters on spitting the text blocks into words.
        num_initial_chars: int
            The number of Unicode characters with which to initialize the vocabulary.
        num_merges: int
            The number of byte pairs to merge while creating the vocabulary.

    TODO
    >>> LOG.setLevel(logging.ERROR)
    >>> base_dir = os.path.dirname(os.path.realpath(__file__))
    >>> input_dir = os.path.join(base_dir, "../examples")
    >>> output_file = os.path.join(base_dir, "../examples/vocab-words.tmp.txt")

    >>> main(input_dir, output_file, num_initial_chars=5, num_merges=5)
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ['!', '"', '#', '$', '%', 'er', 'er✂', 'e✂', 'll', 'ller✂']
    >>> main(input_dir, output_file, max_num_input_files=1, num_initial_chars=5, num_merges=5)
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ['!', '"', '#', '$', '%', 'Be', 'er', 'er✂', 'it', 'll']
    >>> main(input_dir, output_file, input_files_prefix="vldb", num_initial_chars=5, num_merges=5)
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ['!', '"', '#', '$', '%', 'A✂', 'co', 'com', 'comp', 've']

    >>> os.remove(output_file)
    """
    LOG.info("Creating a BPE vocabulary ...")
    LOG.debug("Arguments:")
    LOG.debug(f" - ground truth dir: '{ground_truth_dir}'")
    LOG.debug(f" - output file path: '{output_file_path}'")
    LOG.debug(f" - ground truth files prefix: '{ground_truth_files_prefix}'")
    LOG.debug(f" - ground truth files suffix: '{ground_truth_files_suffix}'")
    LOG.debug(f" - word_delimiters: {word_delimiters}")
    LOG.debug(f" - number of initial characters: {num_initial_chars}")
    LOG.debug(f" - number of byte pair merges: {num_merges}")

    LOG.info("Reading the ground truth files ...")
    gt_reader = ground_truth_reader.GroundTruthReader()
    gt_docs: List[Document] = gt_reader.read(
        directory=ground_truth_dir,
        file_name_pattern=f"{ground_truth_files_prefix}*{ground_truth_files_suffix}"
    )
    LOG.debug(f"Done. Found {len(gt_docs):,} ground truth files.")

    # Extract the text from the text blocks.
    LOG.debug("Concatenating the texts of the text blocks to a single text ...")
    text = " ".join([b.text for doc in gt_docs for b in doc.blocks])
    LOG.debug(f"Excerpt: \"{text[:min(len(text), 100)]} ...\"")

    # Split the text into words.
    LOG.debug("Splitting the text into words ...")
    tokenizer = word_tokenizer.WordTokenizer(word_delimiters)
    words = tokenizer.tokenize_into_words(text)
    LOG.debug(f"Splitted the texts of the text blocks into {len(words):,} words.")
    LOG.debug(f"Excerpt: {words[:min(len(words), 10)]}.")

    # Create a BPE vocabulary from the words.
    LOG.debug("Creating the BPE vocabulary ...")
    vocabulary = bpe.create_vocabulary(words, num_initial_chars, num_merges)
    LOG.debug(f"Vocabulary of size {len(vocabulary):,} created.")
    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[-3:]])
    LOG.debug(f"Excerpt: {{{head}, ..., {tail}}}")

    # Write the vocabulary to file.
    LOG.debug(f"Writing the vocabulary to '{output_file_path}'...")
    vocab_writer = vocab_utils.VocabularyWriter()
    vocab_writer.write(vocabulary, output_file_path)
    LOG.debug("Done.")
    LOG.debug("Excerpt:")
    with open(output_file_path, "r") as fin:
        lines = [x for x in fin.read().split("\n") if x]
        for line in lines[:3]:
            LOG.debug(line)
        LOG.debug("...")
        for line in lines[-3:]:
            LOG.debug(line)


# =================================================================================================


if __name__ == "__main__":
    # Create a command line argument parser.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.description = """
    A script that (1) parses the ground truth files for the texts of the contained text blocks, (2)
    creates a vocabulary from the texts of the text blocks, as needed for encoding word sequences
    with byte pair encoding, and (3) writes the vocabulary to a given file.
    """

    # The path to the directory from which to read the ground truth files.
    arg_parser.add_argument(
        "-g", "--ground_truth_dir",
        type=str,
        required=True,
        help="The path to the directory from which to read the ground truth files."
    )

    # The prefix of the files to read from the ground truth directory.
    arg_parser.add_argument(
        "--ground_truth_files_prefix",
        type=str,
        required=False,
        default=GROUND_TRUTH_FILES_PREFIX,
        help="The prefix of the files to read from the ground truth directory."
    )

    # The suffix of the files to read from the ground truth directory.
    arg_parser.add_argument(
        "--ground_truth_files_suffix",
        type=str,
        required=False,
        default=GROUND_TRUTH_FILES_SUFFIX,
        help="The suffix of the files to read from the ground truth directory."
    )

    # The path to the file to which to write the vocabulary.
    arg_parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="The path to the file to which to write the vocabulary."
    )

    # The number of Unicode characters with which to initialize the vocabulary.
    arg_parser.add_argument(
        "--num_initial_chars",
        type=int,
        required=False,
        default=NUM_INITIAL_CHARS,
        help="The number of Unicode characters with which to initialize the vocabulary."
    )

    # The number of byte pairs to merge while creating the vocabulary.
    arg_parser.add_argument(
        "--num_merges",
        type=int,
        required=False,
        default=NUM_MERGES,
        help="The number of byte pairs to merge while creating the vocabulary."
    )

    # The log level.
    arg_parser.add_argument(
        "--logging_level",
        type=str,
        default=LOGGING_LEVEL,
        help="The logging level."
    )

    # Parse the command line arguments.
    args = arg_parser.parse_args()

    # Set the logging level.
    LOG.setLevel(log_utils.to_log_level(args.logging_level))

    main(
        ground_truth_dir=args.ground_truth_dir,
        ground_truth_files_prefix=args.ground_truth_files_prefix,
        ground_truth_files_suffix=args.ground_truth_files_suffix,
        output_file_path=args.output_file,
        num_initial_chars=args.num_initial_chars,
        num_merges=args.num_merges
    )
