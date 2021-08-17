"""
A script to evaluate a given model for detecting the semantic roles of building blocks.
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import List
from utils.files import BuildingBlock, Page

import data_reader

import numpy as np

import tensorflow.keras as keras

import utils.log
import utils.metrics
import utils.type
import utils.image_utils


# =================================================================================================

# The logger.
LOG = utils.log.get_logger(__name__)

# =================================================================================================
# Settings.

# The default logging level.
LOG_LEVEL = "debug"

# The default input directory.
INPUT_DIR = "/input"
# The default maximum number of input files to read from the input directory (None: read all files)
MAX_NUM_INPUT_FILES = -1
# The default file prefix to consider on reading the input directory.
INPUT_FILES_PREFIX = ""
# The default file suffix to consider on reading the input directory.
INPUT_FILES_SUFFIX = ".tsv"
# The default model directory.
MODEL_DIR = "/model"
# The default output directory.
OUTPUT_DIR = "/output"

# =================================================================================================

def predict(blocks: List[BuildingBlock], pages: List[Page]):
    """
    TODO
    """

    model_dir = "./models/2021-08-13_model-with-header-footers-included" # TODO

    # Configure the loggers.
    LOG.setLevel(utils.log.to_log_level(LOG_LEVEL))
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    logging.getLogger("polyaxon.client").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    LOG.info("Evaluating model '{}' ...".format(model_dir))

    # ---------------------------------------------------------------------------------------------
    # Find the args file (containing the args used on training the model) belonging to the model.

    LOG.debug("Identifying the belonging args file ...")
    args_file_path = os.path.join(model_dir, "model_args.json")
    if not os.path.exists(args_file_path):
        LOG.error("No args file \"{}\" found.".format(args_file_path))
        sys.exit(1)

    with open(args_file_path, "rb") as stream:
        args = json.load(stream)
        # for k, v in args_json.items():
        #     # Don't overwrite existing arguments.
        #     if not hasattr(args, k):
        #         setattr(args, k, v)

    # ---------------------------------------------------------------------------------------------
    # Read the input data.

    reader = data_reader.DataReader(
        # input_dir=args.input_dir,
        # input_file_name_pattern="{0}*{1}".format(args.prefix, args.suffix),
        # max_num_input_files=args.max_num_files,
        shuffle_input_files=True,
        encoding=args.encoding,
        words_vocab=utils.files.read_vocabulary_file(os.path.join(model_dir, "vocab_bpe.tsv")),  # TODO
        roles_vocab=utils.files.read_vocabulary_file(os.path.join(model_dir, "vocab_roles.tsv")),  # TODO
        word_delimiters=args.word_delimiters,
        lowercase_words=args.lowercase_words,
        num_words_per_seq=args.num_words_per_seq,
        include_positions=args.include_positions,
        include_font_sizes=args.include_font_sizes,
        include_font_styles=args.include_font_styles,
        include_char_features=args.include_char_features,
        include_semantic_features=args.include_semantic_features,
        # countries_db_file=args.countries_db_file,
        # human_names_db_file=args.human_names_db_file,
    )
    word_seqs, features, _ = reader.from_doc_building_blocks(blocks, pages)

    if len(word_seqs) == 0:
        LOG.error("No word sequences given.")
        return

    # ---------------------------------------------------------------------------------------------
    # Evaluate.

    # Load the model.
    LOG.info("Loading model from '{}' ...".format(model_dir))
    custom_metrics = {}
    for i in reader.roles_vocab.items():
        f = utils.metrics.role_acc(*i)
        custom_metrics[f.__name__] = f
    model = keras.models.load_model(model_dir, custom_objects=custom_metrics)

    # Use the model for predicting the semantic roles.
    LOG.info("Predicting ...")
    predicted_roles = model.predict([word_seqs, features], verbose=1)

    for i, role in enumerate(predicted_roles):
        blocks[i].role = role

# =================================================================================================


if __name__ == "__main__":
    # Create a command line argument parser.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The path to the input data directory.
    arg_parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=False,
        default=INPUT_DIR,
        help="The path to the input directory (with the tsv files to read the input data from)."
    )

    # The path to the output directory, where the prediction result file and the images of the
    # blocks should be stored.
    arg_parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=False,
        default=OUTPUT_DIR,
        help="The path to the directory to which the prediction result file (and the images of the "
             "blocks, when --create_images is set) should be written."
    )

    # The path to the model dir.
    arg_parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=False,
        default=MODEL_DIR,
        help="The path to the directory containing a saved_model.pb and model_args.json file."
    )

    # The boolean flag indicating whether or not images of the blocks should be created.
    arg_parser.add_argument(
        "--create-images",
        action="store_true",
        help="Create images (in JPG format) of the text blocks and write them to the specified "
             "output directory. The filename pattern of the images is <i>.jpg where i corresponds "
             "to the index of the block in the prediction result file."
    )

    # The prefix of the training data files in the input directory.
    arg_parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default=INPUT_FILES_PREFIX,
        help="The prefix of the training data files in the input directory."
    )

    # The suffix of the training data files in the input directory.
    arg_parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        default=INPUT_FILES_SUFFIX,
        help="The suffix of the training data files in the input directory."
    )

    # The maximum number of files to read.
    arg_parser.add_argument(
        "--max_num_files",
        type=int,
        required=False,
        default=MAX_NUM_INPUT_FILES,
        help="The maximum number of files to read (set to -1 for reading all files)."
    )

    # The log level.
    arg_parser.add_argument(
        "--log_level",
        type=str,
        default=LOG_LEVEL,
        help="The log level."
    )

    main(arg_parser.parse_args())