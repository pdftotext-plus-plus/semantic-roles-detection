"""
A script that (1) parses the ground truth files of this project (aka "detecting semantic roles of
building blocks") for the semantic roles of the contained building blocks, (2) creates a vocabulary
from the roles of the blocks (a mapping of each role to an unique integer id) and (3) writes the
vocabulary to a file.
"""

import argparse
import logging  # NOQA
import os.path  # NOQA
import sys
sys.path.append("..")  # Needed so that utils.* can be found.  # NOQA

import utils.encoding.bpe
import utils.files
import utils.log
import utils.word_tokenizer

# =================================================================================================

# The logger.
LOG = utils.log.get_logger(__name__)

# =================================================================================================

# The prefix of the ground truth files to read from the input directory.
INPUT_FILES_PREFIX = ""
# The suffix of the ground truth files to read from the input directory.
INPUT_FILES_SUFFIX = ".tsv"
# The max number of ground truth files to read from the input directory (-1 means: read all files).
MAX_NUM_INPUT_FILES = -1

# =================================================================================================


def main(input_dir, output_file, input_files_prefix=INPUT_FILES_PREFIX,
         input_files_suffix=INPUT_FILES_SUFFIX, max_num_input_files=MAX_NUM_INPUT_FILES):
    """
    The main method.

    Args:
        input_dir (str):
            The path to the directory from which the ground truth files should be read.
        output_file (str):
            The path to the file to which the vocabulary should be stored.
        input_files_prefix (str):
            The prefix of the ground truth files to read from the input directory.
        input_files_suffix (str):
            The suffix of the ground truth files to read from the input directory.
        max_num_input_files (int):
            The maximum number of ground truth files to read (-1 means: read all files).

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
    LOG.debug(" - input dir: '{}'".format(input_dir))
    LOG.debug(" - output file: '{}'".format(output_file))
    LOG.debug(" - input files prefix: '{}'".format(input_files_prefix))
    LOG.debug(" - input files suffix: '{}'".format(input_files_suffix))
    LOG.debug(" - max. number of input files to read: {}".format(max_num_input_files))

    # Parse the input directory for the ground truth files.
    LOG.debug("Parsing directory '{}' for ground truth files ...".format(input_dir))
    input_files = utils.files.parse_dir(
        directory=input_dir,
        pattern="{0}*{1}".format(input_files_prefix, input_files_suffix),
        max_num_files=max_num_input_files,
        shuffle_files=False
    )
    LOG.debug("Done. Selected {:,} ground truth files.".format(len(input_files)))
    LOG.debug("Excerpt: {}.".format(input_files[:min(len(input_files), 3)]))

    # Read the ground truth files.
    LOG.debug("Reading the ground truth files ...")
    building_blocks, _, _, _ = utils.files.read_groundtruth_files(input_files)
    LOG.debug("{:,} ground truth files read. Found {:,} building blocks."
              .format(len(input_files), len(building_blocks)))

    # Create the vocabulary.
    LOG.debug("Creating the vocabulary ...")
    vocabulary = {}
    for block in building_blocks:
        role = block.role
        # FIXME: There are two roles for a figure-caption ("FIGURES_FIGURE-CAPTION" and
        # "FIGURES-WIDE_FIGURE-CAPTION"), which should be one.
        if role == "FIGURES-WIDE_FIGURE-CAPTION":
            role = "FIGURES_FIGURE-CAPTION"
        if role not in vocabulary:
            vocabulary[role] = len(vocabulary)
    LOG.debug("Vocabulary of size {:,} created.".format(len(vocabulary)))
    head = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[:3]])
    tail = ", ".join([": ".join([x[0], str(x[1])]) for x in list(vocabulary.items())[-3:]])
    LOG.debug("Excerpt: {{{}, ..., {}}}".format(head, tail))

    # Write the vocabulary to file.
    LOG.debug("Writing the vocabulary to '{}'...".format(output_file))
    utils.files.write_vocabulary_file(vocabulary, output_file)
    LOG.debug("Done.")
    LOG.debug("Excerpt:")
    with open(output_file, "r") as fin:
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
    A script that (1) parses the ground truth files of this project (aka "detecting semantic roles
    of building blocks") for the semantic roles of the contained building blocks, (2) creates a
    vocabulary from the roles of the blocks (a mapping of each role to an unique integer id) and
    (3) writes the vocabulary to a file.
    """

    # The path to the directory from which the ground truth files should be read.
    arg_parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="The path to the directory from which the ground truth files should be read."
    )

    # The prefix of the ground truth files to read from the input directory.
    arg_parser.add_argument(
        "--input_files_prefix",
        type=str,
        required=False,
        default=INPUT_FILES_PREFIX,
        help="The prefix of the ground truth files to read from the input directory."
    )

    # The suffix of the ground truth files to read from the input directory.
    arg_parser.add_argument(
        "--input_files_suffix",
        type=str,
        required=False,
        default=INPUT_FILES_SUFFIX,
        help="The suffix of the ground truth files to read from the input directory."
    )

    # The max number of ground truth files to read from the input directory.
    arg_parser.add_argument(
        "--max_num_input_files",
        type=int,
        required=False,
        default=MAX_NUM_INPUT_FILES,
        help="The max number of ground truth files to read from the input directory (set to -1 to \
read all files)."
    )

    # The path to the file to which the vocabulary should be stored.
    arg_parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="The path to the file to which the vocabulary should be stored."
    )

    # Parse the command line arguments.
    args = arg_parser.parse_args()

    main(
        input_dir=args.input_dir,
        input_files_prefix=args.input_files_prefix,
        input_files_suffix=args.input_files_suffix,
        max_num_input_files=args.max_num_input_files,
        output_file=args.output_file
    )
