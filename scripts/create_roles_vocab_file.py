"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains a script that
(1) parses the ground truth files for the roles of the contained text blocks, (2) creates a
vocabulary from the roles of the blocks (a mapping of each role to an unique integer id) and
(3) writes the vocabulary to a file.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse
import sys
from typing import List

sys.path.append("../src/semantic_roles_detection")  # Needed so that utils.* can be found.  # NOQA
from utils import ground_truth_reader
from utils import log_utils
from utils import vocab_utils
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

# =================================================================================================


def main(ground_truth_dir: str,
         output_file_path: str,
         ground_truth_files_prefix: str = GROUND_TRUTH_FILES_PREFIX,
         ground_truth_files_suffix: str = GROUND_TRUTH_FILES_SUFFIX):
    """
    This method is the main method for creating a roles vocabulary.

    Args:
        ground_truth_dir: str
            The path to the directory from which to read the ground truth files.
        output_file_path: str
            The path to the file to which to write the vocabulary.
        ground_truth_files_prefix: str
            The prefix of the files to read from the ground truth directory.
        ground_truth_files_suffix: str
            The suffix of the files to read from the ground truth directory.

    TODO
    >>> LOG.setLevel(logging.ERROR)
    >>> base_dir = os.path.dirname(os.path.realpath(__file__))
    >>> input_dir = os.path.join(base_dir, "../examples")
    >>> output_file = os.path.join(base_dir, "../examples/roles_vocab.tmp.txt")

    >>> main(input_dir, output_file)
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ... #doctest: +NORMALIZE_WHITESPACE
    ['AUTHOR_COUNTRY', 'AUTHOR_MAIL', 'AUTHOR_NAME', 'HEADING_HEADING', 'PARAGRAPHS', \
     'PUBLICATION-DATE', 'TITLE']
    >>> main(input_dir, output_file, max_num_input_files=1)
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ['AUTHOR_NAME', 'HEADING_HEADING', 'PARAGRAPHS', 'PUBLICATION-DATE', 'TITLE']
    >>> main(input_dir, output_file, input_files_prefix="vldb")
    >>> sorted([x.split("\\t")[0] for x in open(output_file).read().split("\\n") if x])
    ['AUTHOR_COUNTRY', 'AUTHOR_NAME', 'PARAGRAPHS', 'TITLE']

    >>> os.remove(output_file)
    """
    LOG.info("Creating a roles vocabulary ...")
    LOG.debug("Arguments:")
    LOG.debug(f" - ground truth dir: '{ground_truth_dir}'")
    LOG.debug(f" - output file path: '{output_file_path}'")
    LOG.debug(f" - ground truth files prefix: '{ground_truth_files_prefix}'")
    LOG.debug(f" - ground truth files suffix: '{ground_truth_files_suffix}'")

    LOG.info("Reading the ground truth files ...")
    gt_reader = ground_truth_reader.GroundTruthReader()
    gt_docs: List[Document] = gt_reader.read(
        directory=ground_truth_dir,
        file_name_pattern=f"{ground_truth_files_prefix}*{ground_truth_files_suffix}"
    )
    LOG.debug(f"Done. Found {len(gt_docs):,} ground truth files.")

    # Create the vocabulary.
    LOG.debug("Creating the vocabulary ...")
    vocabulary = {}
    for doc in gt_docs:
        for block in doc.blocks:
            role = block.role
            if role not in vocabulary:
                vocabulary[role] = len(vocabulary)
    LOG.debug(f"Created vocabulary of size {len(vocabulary):,}.")
    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[-3:]])
    LOG.debug(f"Excerpt: {{{head}, ..., {tail}}}")

    # Write the vocabulary to file.
    LOG.debug(f"Writing the vocabulary to '{output_file_path}'...")
    vocab_writer = vocab_utils.VocabularyWriter()
    vocab_writer.write(vocabulary, output_file_path)
    LOG.debug("Done.")
    LOG.debug("Excerpt:")
    with open(output_file_path, "r") as stream:
        lines = [x for x in stream.read().split("\n") if x]
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
    A script that (1) parses the ground truth files for the roles of the contained text blocks, (2)
    creates a vocabulary from the roles of the blocks (a mapping of each role to an unique integer
    id) and (3) writes the vocabulary to a file.
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
        output_file_path=args.output_file
    )
