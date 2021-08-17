"""
A script that (1) parses the ground truth files relating to this project (aka "detecting semantic
roles of building blocks") for the texts of the contained building blocks, (2) creates a vocabulary
from the texts of the building blocks, as needed for byte pair encoding the word sequences, and
(3) writes the vocabulary to a given file.
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

class OriginLogRecord(logging.LogRecord):
    """
    A custom log record that combines the fields "filename" and "lineno" to the single field
    "origin". This enables the option to define a fixed width for this field. Otherwise, we would
    have to define a fixed width for both fields which breaks both fields apart.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = f"{self.filename}:{self.lineno}"

logging.setLogRecordFactory(OriginLogRecord)
logging.basicConfig(
  level=logging.DEBUG,
  format="%(asctime)s - %(origin)50.50s - %(levelname)-5s : %(message)s"
)
LOG = utils.log.get_logger(__name__)

# =================================================================================================

# The prefix of the ground truth files to read from the input directory.
INPUT_FILES_PREFIX = ""
# The suffix of the ground truth files to read from the input directory.
INPUT_FILES_SUFFIX = ".tsv"
# The max number of ground truth files to read from the input directory (-1 means: read all files).
MAX_NUM_INPUT_FILES = -1
# The symbols to be consider as word delimiters within building block texts.
WORD_DELIMITERS = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n "
# The number of Unicode characters with which the vocabulary should be initialized.
NUM_INITIAL_CHARS = 256
# The number of byte pairs to merge while creating the vocabulary.
NUM_MERGES = 2000

# =================================================================================================


def main(input_dir, output_file, input_files_prefix=INPUT_FILES_PREFIX,
         input_files_suffix=INPUT_FILES_SUFFIX, max_num_input_files=MAX_NUM_INPUT_FILES,
         word_delimiters=WORD_DELIMITERS, num_initial_chars=NUM_INITIAL_CHARS,
         num_merges=NUM_MERGES):
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
        word_delimiters (str or list):
            The symbols to be consider as word delimiters within building block texts.
        num_initial_chars (int):
            The number of Unicode characters with which the vocabulary should be initialized.
        num_merges (int):
            The number of byte pairs to merge while creating the vocabulary.

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
    LOG.debug(" - input dir: '{}'".format(input_dir))
    LOG.debug(" - output file: '{}'".format(output_file))
    LOG.debug(" - input files prefix: '{}'".format(input_files_prefix))
    LOG.debug(" - input files suffix: '{}'".format(input_files_suffix))
    LOG.debug(" - max. number of input files to read: {}".format(max_num_input_files))
    LOG.debug(" - word_delimiters: {}".format(word_delimiters))
    LOG.debug(" - number of initial characters: {}".format(num_initial_chars))
    LOG.debug(" - number of byte pair merges: {}".format(num_merges))

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

    # Extract the the text from the building blocks.
    LOG.debug("Concatenating the texts of the blocks to a single text ...")
    text = " ".join([b.text for b in building_blocks])
    LOG.debug("Excerpt: \"{} ...\"".format(text[:min(len(text), 100)]))

    # Split the text into words.
    LOG.debug("Splitting the texts of the building blocks into words ...")
    word_tokenizer = utils.word_tokenizer.WordTokenizer(word_delimiters)
    words = word_tokenizer.tokenize_into_words(text)
    LOG.debug("Splitted the texts of the building blocks into {:,} words.".format(len(words)))
    LOG.debug("Excerpt: {}.".format(words[:min(len(words), 10)]))

    # Create a BPE vocabulary from the words.
    LOG.debug("Creating the vocabulary ...")
    vocabulary = utils.encoding.bpe.create_vocabulary(words, num_initial_chars, num_merges)
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
    A script that (1) parses the ground truth files relating to this project (aka "detecting
    semantic roles of building blocks") for the texts of the contained building blocks, (2) creates
    a vocabulary from the texts of the building blocks, as needed for byte pair encoding the word
    sequences, and (3) writes the vocabulary to a given file.
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

    # The number of Unicode characters with which the vocabulary should be initialized.
    arg_parser.add_argument(
        "--num_initial_chars",
        type=int,
        required=False,
        default=NUM_INITIAL_CHARS,
        help="The number of Unicode characters with which the vocabulary should be initialized."
    )

    # The number of byte pairs to merge while creating the vocabulary.
    arg_parser.add_argument(
        "--num_merges",
        type=int,
        required=False,
        default=NUM_MERGES,
        help="The number of byte pairs to merge while creating the vocabulary."
    )

    # Parse the command line arguments.
    args = arg_parser.parse_args()

    main(
        input_dir=args.input_dir,
        input_files_prefix=args.input_files_prefix,
        input_files_suffix=args.input_files_suffix,
        max_num_input_files=args.max_num_input_files,
        output_file=args.output_file,
        num_initial_chars=args.num_initial_chars,
        num_merges=args.num_merges
    )
