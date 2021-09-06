"""
A script to start the evaluation of a learning model of the "semantic-roles-detection" module from
the command line.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse

from semantic_roles_detection.evaluate import evaluate

# =================================================================================================
# Parameters.

# The default logging level.
LOGGING_LEVEL = "debug"
# The default ground-truth directory.
GROUND_TRUTH_DIR = "/ground-truth"
# The default prefix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_PREFIX = ""
# The default suffix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_SUFFIX = ".tsv"
# The default number of files to read from the ground truth dir (set to -1 to read all files).
NUM_GROUND_TRUTH_FILES = -1
# The default directory of the model to evaluate.
MODEL_DIR = "/model"
# The name of the args file.
ARGS_FILE_NAME = "model-args.json"
# The name of the BPE vocab file.
BPE_VOCAB_FILE_NAME = "bpe-vocab.tsv"
# The name of the roles vocab file.
ROLES_VOCAB_FILE_NAME = "roles-vocab.tsv"
# The default output directory.
OUTPUT_DIR = "/output"
# The name of the evaluation result file.
EVALUATION_RESULT_FILE_NAME = "evaluation-results.tsv"

# =================================================================================================


if __name__ == "__main__":
    # Create a command line argument parser.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The path to the ground truth directory.
    arg_parser.add_argument(
        "-g", "--ground_truth_dir",
        type=str,
        required=False,
        default=GROUND_TRUTH_DIR,
        help="The path to the ground truth directory (with the files containing the text blocks)."
    )

    # The path to the output directory, where the evaluation result file and the images of the
    # blocks should be stored.
    arg_parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=False,
        default=OUTPUT_DIR,
        help="The path to the directory to which the evaluation result file (and the images of "
             "the blocks, when --create_images is set) should be written."
    )

    # The path to the model dir.
    arg_parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=False,
        default=MODEL_DIR,
        help="The path to the directory of the model to evaluate (containing a saved_model.pb)."
    )

    # A boolean flag indicating whether or not to create images of the text blocks.
    arg_parser.add_argument(
        "--create-images",
        action="store_true",
        help="Create images (in PNG format) of the text blocks in the PDF and store them in the "
             "specified output directory. The filename pattern of the images is <i>.jpg where i "
             "is the id of the block as specified in the evaluation result file."
    )

    # The prefix of the ground truth files in the ground truth directory.
    arg_parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default=GROUND_TRUTH_FILES_PREFIX,
        help="The prefix of the files to read from the ground truth directory."
    )

    # The suffix of the ground truth files in the ground truth directory.
    arg_parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        default=GROUND_TRUTH_FILES_SUFFIX,
        help="The prefix of the files to read from the ground truth directory."
    )

    # The number of ground truth files to read.
    arg_parser.add_argument(
        "--num_files",
        type=int,
        required=False,
        default=NUM_GROUND_TRUTH_FILES,
        help="The number of files to read from the ground truth dir (set to -1 to read all files)."
    )

    # The logging level.
    arg_parser.add_argument(
        "--logging_level",
        type=str,
        default=LOGGING_LEVEL,
        help="The log level."
    )

    # Start the evaluation.
    evaluate(arg_parser.parse_args())
