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


def main(args):
    """
    The main method.

    Args:
        args (namespace):
            The command line arguments.
    """
    # Configure the loggers.
    LOG.setLevel(utils.log.to_log_level(args.log_level))
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    logging.getLogger("polyaxon.client").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    vargs = vars(args)
    LOG.info("Evaluating model '{}' ...".format(args.model_dir))
    for name in vargs:
        LOG.debug(" - {}: {}".format(name, vargs[name]))

    # ---------------------------------------------------------------------------------------------
    # Find the args file (containing the args used on training the model) belonging to the model.

    LOG.debug("Identifying the belonging args file ...")
    args_file_path = os.path.join(args.model_dir, "model_args.json")
    if not os.path.exists(args_file_path):
        LOG.error("No args file \"{}\" found.".format(args_file_path))
        sys.exit(1)

    with open(args_file_path, "rb") as stream:
        args_json = json.load(stream)
        for k, v in args_json.items():
            # Don't overwrite existing arguments.
            if not hasattr(args, k):
                setattr(args, k, v)

    # ---------------------------------------------------------------------------------------------
    # Read the input data.

    reader = data_reader.DataReader(
        input_dir=args.input_dir,
        input_file_name_pattern="{0}*{1}".format(args.prefix, args.suffix),
        max_num_input_files=args.max_num_files,
        shuffle_input_files=True,
        encoding=args.encoding,
        words_vocab=utils.files.read_vocabulary_file(os.path.join(args.model_dir, "vocab_bpe.tsv")),  # TODO
        roles_vocab=utils.files.read_vocabulary_file(os.path.join(args.model_dir, "vocab_roles.tsv")),  # TODO
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
    word_seqs, features, expected_roles = reader.read()

    if len(word_seqs) == 0:
        LOG.error("No word sequences given.")
        return

    if len(expected_roles) == 0:
        LOG.error("No roles given.")
        return

    # ---------------------------------------------------------------------------------------------
    # Evaluate.

    # Load the model.
    LOG.info("Loading model from '{}' ...".format(args.model_dir))
    custom_metrics = {}
    for i in reader.roles_vocab.items():
        f = utils.metrics.role_acc(*i)
        custom_metrics[f.__name__] = f
    model = keras.models.load_model(args.model_dir, custom_objects=custom_metrics)

    # Use the model for predicting the semantic roles.
    LOG.info("Predicting ...")
    predicted_roles = model.predict([word_seqs, features], verbose=1)

    # Compare the predicted roles with the expected roles.
    LOG.info("Evaluating ...")
    expected_roles = np.argmax(reader.roles, axis=1)
    predicted_roles = np.argmax(predicted_roles, axis=1)

    # The breakdown of the correct predictions, in the format: {expected_role: freq}
    correct_breakdown = defaultdict(int)
    # The breakdown of the wrong predictions, in format: {expected_role: {predicted_role: freq}}
    wrong_breakdowns = defaultdict(lambda: defaultdict(int))

    # Open the prediction result file (to which the prediction result should be written).
    prediction_result_file = None
    if args.output_dir is not None:
        # Create the output directory if it does not exist.
        os.makedirs(args.output_dir, exist_ok=True)
        # Compose the path to the prediction result file.
        prediction_result_file = open(os.path.join(args.output_dir, "evaluation-results.tsv"), "w")
        # Write the header.
        prediction_result_file.write("#Id\t#Text\tFont Name\tFont Size\tBold?\tItalic?\tPage\t"
            "Bounding Box\tDoc Slug\tExpected Role\tPredicted Role\n")

    # Iterate through the building blocks and evaluate the prediction result for each. Write the
    # prediction result to file and create an image of each block ((if the respective flag is set).
    for i in range(expected_roles.shape[0]):
        LOG.info("Evaluating building block #{}/{}".format(i + 1, expected_roles.shape[0]))

        building_block = reader.building_blocks[i]
        expected_role = reader.rev_roles_vocab[expected_roles[i]]
        predicted_role = reader.rev_roles_vocab[predicted_roles[i]]

        # Write the prediction result to file.
        if prediction_result_file is not None:
            # Get the path to the TSV file, relative to the input dir.
            rel_doc_path = os.path.relpath(building_block.doc_path, args.input_dir)
            # Get the relative basename.
            rel_doc_base_name = rel_doc_path.replace(".gt.tsv", "")

            prediction_result_file.write(f"{building_block.id}\t{building_block.text}\t"
              f"{building_block.font_name}\t{building_block.font_size}\t{building_block.is_bold}\t"
              f"{building_block.is_italic}\t{building_block.page_num}\t"
              f"{building_block.lower_left_x},{building_block.lower_left_y},"
              f"{building_block.upper_right_x},{building_block.upper_right_y}\t"
              f"{rel_doc_base_name}\t{expected_role}\t{predicted_role}\n")

        # Create an image of the respective text block.
        if args.create_images:
            # Compute "math0307072" from "math0307072.gt.tsv"
            doc_base_name = building_block.doc_path.replace(".gt.tsv", "")

            if building_block.page_num and building_block.page_num > 0:
                utils.image_utils.to_cropped_png(
                    pdf_file=os.path.join(args.input_dir, f"{doc_base_name}.wc.pdf"),
                    png_file=os.path.join(args.output_dir, f"{building_block.id}.png"),
                    page_num=building_block.page_num - 1,
                    crop_box=[
                      building_block.lower_left_x - 100,
                      building_block.lower_left_y - 100,
                      building_block.upper_right_x + 100,
                      building_block.upper_right_y + 100
                    ],
                    highlight_box=[
                      building_block.lower_left_x,
                      building_block.lower_left_y,
                      building_block.upper_right_x,
                      building_block.upper_right_y
                    ]
                )

        if expected_role == predicted_role:
            correct_breakdown[expected_role] += 1
            continue

        LOG.info("-" * 20)
        LOG.info("Wrong prediction for building block #{}".format(i))
        LOG.info("Building block: {}".format(building_block))
        LOG.info("Expected role: {}".format(expected_role))
        LOG.info("Predicted role: {}".format(predicted_role))
        wrong_breakdowns[expected_role][predicted_role] += 1

    # Close the prediction result file.
    if prediction_result_file is not None:
        prediction_result_file.close()

    # Compute the total accuracy.
    correct_num = sum(correct_breakdown.values())
    wrong_num = sum([sum(x.values()) for x in wrong_breakdowns.values()])
    accuracy = correct_num / (correct_num + wrong_num)

    # Compute the accuracies per role.
    accuracies_per_role = []
    for role in reader.roles_vocab.keys():
        correct_num_role = correct_breakdown[role]
        wrong_num_role = sum(wrong_breakdowns[role].values())
        total_num_role = correct_num_role + wrong_num_role
        accuracy_role = correct_num_role / total_num_role if total_num_role > 0 else 0
        accuracies_per_role.append((role, accuracy_role, correct_num_role, wrong_num_role))
    accuracies_per_role.sort(key=lambda x: x[1])

    LOG.info("=" * 50)
    LOG.info("\033[1mEvaluation done.\033[0m")
    LOG.info("-" * 50)
    LOG.info("\033[1mNumber of evaluated sequences:\033[0m {:>7,}".format(correct_num + wrong_num))
    LOG.info("\033[1mNumber of correct predictions:\033[0m {:>7,}".format(correct_num))
    LOG.info("\033[1mNumber of wrong predictions:\033[0m   {:>7,}".format(wrong_num))
    LOG.info("\033[1mAccuracy:\033[0m {:.2f}".format(accuracy))
    LOG.info("\033[1m#correct:\033[0m {}".format(correct_num))
    LOG.info("\033[1m#wrong:\033[0m   {}".format(wrong_num))
    LOG.info("-" * 50)
    LOG.info("Accuracies per role:")

    k = 3
    for role, accuracy, correct_num, wrong_num in accuracies_per_role:
        LOG.info("-" * 50)
        LOG.info("\033[1mRole:\033[0m {}; \033[1mAccuracy:\033[0m {:.2f}; "
                 "\033[1m#correct:\033[0m {}; \033[1m#wrong:\033[0m {}; "
                 "\033[1mtop-{} wrong predictions:\033[0m".format(
                    role, accuracy, correct_num, wrong_num, k))
        wrong_breakdown = list(wrong_breakdowns[role].items())
        wrong_breakdown.sort(key=lambda x: x[1], reverse=True)
        if len(wrong_breakdown) > 0:
            for i in range(min(k, len(wrong_breakdown))):
                LOG.info(" - {}: {}".format(wrong_breakdown[i][0], wrong_breakdown[i][1]))

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