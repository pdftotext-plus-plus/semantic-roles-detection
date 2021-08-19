"""
A script to evaluate the prediction result of a deep learning model for predicting the semantic
roles of given text blocks extracted from PDF files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse
import json
import logging
import os
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log output of tensorflow  # NOQA

import numpy as np

import tensorflow.keras as keras

from utils import feature_encoder
from utils import ground_truth_reader
from utils import image_utils
from utils import log_utils
from utils import metrics
from utils import vocab_utils
from utils.models import TextBlock

# =================================================================================================
# Configure the logging.

LOG = log_utils.get_logger(__name__)

# Disable the debug output of some third-party libraries.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

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


def evaluate(args):
    """
    This method evaluates the prediction results of a model for predicting the semantic roles of
    text blocks extracted from PDF files.

    Args:
        args: argparse.Namespace
            The command line arguments.
    """
    LOG.info(f"Evaluating model '{args.model_dir}' ...")
    vargs = vars(args)
    for name in vargs:
        LOG.debug(f" - {name}: {vargs[name]}")

    # ---------------------------------------------------------------------------------------------
    # Read the args file of the model (containing the parameters used on training the model).

    # Get the path to the args file.
    args_file_path = os.path.join(args.model_dir, ARGS_FILE_NAME)
    LOG.info(f"Reading args file '{args_file_path}' ...")

    if not os.path.exists(args_file_path):
        raise ValueError(f"Args file '{args_file_path}' not found.")

    # Read the args file.
    with open(args_file_path, "rb") as stream:
        args_json = json.load(stream)
        for k, v in args_json.items():
            # Don't overwrite existing arguments.
            if not hasattr(args, k):
                setattr(args, k, v)

    # ---------------------------------------------------------------------------------------------
    # Read the input data.

    LOG.info("Reading the ground truth files ...")
    gt_reader = ground_truth_reader.GroundTruthReader()
    gt_docs = gt_reader.read(
        directory=args.ground_truth_dir,
        file_name_pattern=f"{args.prefix}*{args.suffix}",
        num_files=args.num_files,
        shuffle_files=args.shuffle
    )

    LOG.info("Reading the vocabulary files ...")
    vocab_reader = vocab_utils.VocabularyReader()
    bpe_vocab = vocab_reader.read(os.path.join(MODEL_DIR, BPE_VOCAB_FILE_NAME))
    roles_vocab = vocab_reader.read(os.path.join(MODEL_DIR, ROLES_VOCAB_FILE_NAME))

    LOG.info("Encoding the ground truth ...")
    encoder = feature_encoder.FeatureEncoder(
        bpe_vocab=bpe_vocab,
        roles_vocab=roles_vocab,
        word_delimiters=args.word_delimiters,
        is_lowercase_text=args.lowercase_text,
        word_seq_length=args.word_seq_length,
        is_include_positions=args.include_positions,
        is_include_font_sizes=args.include_font_sizes,
        is_include_font_styles=args.include_font_styles,
        is_include_char_features=args.include_char_features,
        is_include_semantic_features=args.include_semantic_features,
        countries_db_file=args.countries_db_file,
        human_names_db_file=args.human_names_db_file,
    )
    text_blocks, word_seqs, layout_seqs, expected_roles = encoder.encode_documents(gt_docs)

    if len(word_seqs) == 0:
        LOG.error("No word sequences given.")
        return

    if len(layout_seqs) == 0:
        LOG.error("No layout sequences given.")
        return

    if len(expected_roles) == 0:
        LOG.error("No roles given.")
        return

    # ---------------------------------------------------------------------------------------------
    # Load the model.

    LOG.info(f"Loading model '{MODEL_DIR}' ...")

    # Read the custom metrics (e.g., accuracy_TITLE, accuracy_HEADING, etc.).
    custom_metrics = {}
    for i in encoder.roles_vocab.items():
        f = metrics.role_acc(*i)
        custom_metrics[f.__name__] = f
    # TODO
    f = metrics.role_acc("UNK_ROLE", len(encoder.roles_vocab))
    custom_metrics[f.__name__] = f

    # Load the model.
    model = keras.models.load_model(MODEL_DIR, custom_objects=custom_metrics)

    # ---------------------------------------------------------------------------------------------
    # Evaluate.

    # Use the model for predicting the semantic roles of the text blocks.
    LOG.info("Predicting ...")
    predicted_roles = model.predict([word_seqs, layout_seqs], verbose=1)

    LOG.info("Evaluating ...")

    # Compare the predicted roles with the expected roles.
    expected_roles = np.argmax(expected_roles, axis=1)
    predicted_roles = np.argmax(predicted_roles, axis=1)

    # The breakdown of the correct predictions, in the format: {expected_role: freq}
    correct_breakdown = defaultdict(int)
    # The breakdown of the wrong predictions, in format: {expected_role: {predicted_role: freq}}
    wrong_breakdowns = defaultdict(lambda: defaultdict(int))

    # Open the evaluation result file (to which the evaluation result should be written).
    evaluation_result_file = None
    if args.output_dir is not None:
        # Create the output directory if it does not exist.
        os.makedirs(args.output_dir, exist_ok=True)
        # Compose the path to the evaluation result file.
        evaluation_result_file_path = os.path.join(args.output_dir, EVALUATION_RESULT_FILE_NAME)
        evaluation_result_file = open(evaluation_result_file_path, "w")
        # Write the header.
        evaluation_result_file.write(
            "#Id\t#Text\tFont Name\tFont Size\tBold?\tItalic?\tPage\tBounding Box\tDoc Slug\t"
            "Expected Role\tPredicted Role\n")

    # Iterate through the text blocks and evaluate the prediction for each. Write the evaluation
    # result to file and create an image of each block (if the respective flag is set).
    for i in range(expected_roles.shape[0]):
        LOG.info("Evaluating text block #{}/{}".format(i + 1, expected_roles.shape[0]))

        block: TextBlock = text_blocks[i]
        expected_role = encoder.rev_roles_vocab[expected_roles[i]]
        predicted_role = encoder.rev_roles_vocab[predicted_roles[i]]

        # Write the evaluation result to file.
        if evaluation_result_file is not None:
            # Get the path to the TSV file, relative to the input dir.
            rel_doc_path = os.path.relpath(block.ground_truth_file_path, args.ground_truth_dir)
            # Get the relative basename.
            rel_doc_base_name = rel_doc_path.replace(".gt.tsv", "")

            evaluation_result_file.write(
                f"{block.id}\t{block.text}\t"
                f"{block.font_name}\t{block.font_size}\t{block.is_bold}\t"
                f"{block.is_italic}\t{block.page_num}\t"
                f"{block.lower_left_x},{block.lower_left_y},"
                f"{block.upper_right_x},{block.upper_right_y}\t"
                f"{rel_doc_base_name}\t{expected_role}\t{predicted_role}\n"
            )

        # Create an image of the respective text block.
        if args.create_images:
            # Compute "math0307072" from "math0307072.gt.tsv"
            doc_base_name = block.ground_truth_file_path.replace(".gt.tsv", "")

            if block.page_num and block.page_num > 0:
                image_utils.create_png(
                    pdf_file_path=os.path.join(args.ground_truth_dir, f"{doc_base_name}.wc.pdf"),
                    png_file_path=os.path.join(args.output_dir, f"{block.id}.png"),
                    page_num=block.page_num - 1,
                    area=[
                      block.lower_left_x - 100, block.lower_left_y - 100,
                      block.upper_right_x + 100, block.upper_right_y + 100
                    ],
                    highlight_area=[
                      block.lower_left_x, block.lower_left_y,
                      block.upper_right_x, block.upper_right_y
                    ]
                )

        if expected_role == predicted_role:
            correct_breakdown[expected_role] += 1
            continue

        LOG.info("-" * 20)
        LOG.info(f"Wrong prediction for text block #{i}")
        LOG.info(f"Text block: {block}")
        LOG.info(f"Expected role: {expected_role}")
        LOG.info(f"Predicted role: {predicted_role}")
        wrong_breakdowns[expected_role][predicted_role] += 1

    # Close the evaluation result file.
    if evaluation_result_file is not None:
        evaluation_result_file.close()

    # Compute the total accuracy.
    correct_num = sum(correct_breakdown.values())
    wrong_num = sum([sum(x.values()) for x in wrong_breakdowns.values()])
    accuracy = correct_num / (correct_num + wrong_num)

    # Compute the accuracies per role.
    accuracies_per_role = []
    for role in encoder.roles_vocab.keys():
        correct_num_role = correct_breakdown[role]
        wrong_num_role = sum(wrong_breakdowns[role].values())
        total_num_role = correct_num_role + wrong_num_role
        accuracy_role = correct_num_role / total_num_role if total_num_role > 0 else 0
        accuracies_per_role.append((role, accuracy_role, correct_num_role, wrong_num_role))
    accuracies_per_role.sort(key=lambda x: x[1])

    LOG.info("=" * 50)
    LOG.info("\033[1mEvaluation done.\033[0m")
    LOG.info("-" * 50)
    LOG.info(f"\033[1mNumber of evaluated sequences:\033[0m {correct_num + wrong_num:>7,}")
    LOG.info(f"\033[1mNumber of correct predictions:\033[0m {correct_num:>7,}")
    LOG.info(f"\033[1mNumber of wrong predictions:\033[0m   {wrong_num:>7,}")
    LOG.info(f"\033[1mAccuracy:\033[0m {accuracy:.2f}")
    LOG.info(f"\033[1m#correct:\033[0m {correct_num}")
    LOG.info(f"\033[1m#wrong:\033[0m   {wrong_num}")
    LOG.info("-" * 50)
    LOG.info("Accuracies per role:")

    k = 3
    for role, accuracy, correct_num, wrong_num in accuracies_per_role:
        LOG.info("-" * 50)
        LOG.info(f"\033[1mRole:\033[0m {role}; \033[1mAccuracy:\033[0m {accuracy:.2f}; "
                 f"\033[1m#correct:\033[0m {correct_num}; \033[1m#wrong:\033[0m {wrong_num}; "
                 f"\033[1mtop-{k} wrong predictions:\033[0m")
        wrong_breakdown = list(wrong_breakdowns[role].items())
        wrong_breakdown.sort(key=lambda x: x[1], reverse=True)
        if len(wrong_breakdown) > 0:
            for i in range(min(k, len(wrong_breakdown))):
                LOG.info(" - {}: {}".format(wrong_breakdown[i][0], wrong_breakdown[i][1]))

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

    args = arg_parser.parse_args()

    # Set the logging level.
    LOG.setLevel(log_utils.to_log_level(args.logging_level))

    # Start the training.
    evaluate(args)
