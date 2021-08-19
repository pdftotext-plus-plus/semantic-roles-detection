"""
A script to train a deep learning model for predicting the semantic roles of text blocks extracted
from PDF files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import argparse
import json
import logging
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log output of tensorflow  # NOQA

import tensorflow.keras as keras

from utils import argparse_types
from utils import feature_encoder
from utils import ground_truth_reader
from utils import log_utils
from utils import metrics
from utils import vocab_utils


# =================================================================================================
# Configure the logging.

LOG = log_utils.get_logger(__name__)

# Disable the debug output of some third-party libraries.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# =================================================================================================
# Parameters.

# The default logging level.
LOGGING_LEVEL = "info"
# The default path to the ground truth directory.
GROUND_TRUTH_DIR = "/ground-truth"
# The default prefix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_PREFIX = ""
# The default suffix of the files to read from the ground truth directory.
GROUND_TRUTH_FILES_SUFFIX = ".tsv"
# The default number of files to read from the ground truth dir (set to -1 to read all files).
NUM_GROUND_TRUTH_FILES = -1
# Whether or not to shuffle the files before selecting <num_files> from the ground truth
# directory and whether or not to shuffle the input sequences before feeding them into the model.
SHUFFLE = True
# The default directory where to store the output (e.g., the trained model and the args file).
OUTPUT_DIR = "/output"
# The default file name of the args file.
ARGS_FILE_NAME = "model-args.json"
# The default path to the BPE vocabulary file.
BPE_VOCAB_FILE_PATH = "/data/vocabs/bpe-vocab.tsv"  # TODO
# The default path to the roles vocabulary file.
ROLES_VOCAB_FILE_PATH = "/data/vocabs/roles-vocab.tsv"  # TODO
# The default characters to be considered as word delimiters.
WORD_DELIMITERS = "\r\f\t\n "
# The default number of words each input sequence should include from a text block.
WORD_SEQ_LENGTH = 100
# Whether or not to convert the words to lower cases.
IS_LOWERCASE_TEXT = False
# Whether or not to include the positions of the text blocks into the feature vector.
IS_INCLUDE_POSITIONS = True
# Whether or not to include the font sizes of the text blocks into the feature vector.
IS_INCLUDE_FONT_SIZES = True
# Whether or not to include the font styles of the text blocks into the feature vector.
IS_INCLUDE_FONT_STYLES = True
# Whether or not to include the character features into the feature vector.
IS_INCLUDE_CHAR_FEATURES = True
# Whether or not to inlucde the semantic features into the feature vector.
IS_INCLUDE_SEMANTIC_FEATURES = True

# The default dropout value.
DROPOUT = 0.2
# The default activation function.
ACTIVATION = "softmax"
# The default loss function.
LOSS = "categorical_crossentropy"
# The available optimizer functions, per name.
OPTIMIZERS = {"adam": keras.optimizers.Adam}
# The default optimizer function.
OPTIMIZER = "adam"
# The default (logarithmic) learning rate.
LOG_LEARNING_RATE = -3
# The default validation split.
VALIDATION_SPLIT = 0.1
# The default number of epochs.
EPOCHS = 10
# The default update frequency of the progress bar.
PROGRESS_BAR_UPDATE_FREQUENCY = None

# =================================================================================================


def train(args: argparse.Namespace):
    """
    This method trains a new model for predicting the semantic roles of text blocks extracted from
    PDF files.

    Args:
        args: argparse.Namespace
            The command line arguments.
    """
    LOG.info("Training a model for detecting the semantic roles of text blocks ...")
    vargs = vars(args)
    for name in vargs:
        LOG.debug(f" - {name}: {vargs[name]}")

    # ---------------------------------------------------------------------------------------------
    # Read and encode the ground truth.

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
    bpe_vocab = vocab_reader.read(args.bpe_vocab_file)
    roles_vocab = vocab_reader.read(args.roles_vocab_file)

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
    text_blocks, word_seqs, layout_seqs, roles = encoder.encode_documents(gt_docs)

    if len(word_seqs) == 0:
        LOG.error("No word sequences given.")
        return

    if len(layout_seqs) == 0:
        LOG.error("No features given.")
        return

    # ---------------------------------------------------------------------------------------------
    # Build the model.

    LOG.info("Building the model ...")

    # Define the input layer for the word sequences.
    words_input = keras.layers.Input(shape=(word_seqs.shape[1],), dtype="int32", name="words_input")

    # Define an embedding layer that encodes the word sequences into lower-dimensional space.
    words_embedding = keras.layers.Embedding(
        output_dim=256,
        input_dim=len(bpe_vocab),
        input_length=word_seqs.shape[1],
        name="words_embedding"
    )(words_input)

    # Define a single LSTM layer that processes the word sequences.
    words_lstm = keras.layers.LSTM(
        256,
        dropout=args.dropout,
        recurrent_dropout=args.dropout,
        name="words_lstm"
    )(words_embedding)

    # Define the input layer for the layout features (with e.g., font and position information).
    features_input = keras.layers.Input(shape=(layout_seqs.shape[1],), name="layout_features_input")

    # Concatenate the "words_lstm" layer and the "features_input" layer.
    x = keras.layers.concatenate([words_lstm, features_input])
    x = keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(args.dropout, name="dropout")(x)

    # Define the output layer.
    main_output = keras.layers.Dense(
        len(roles_vocab),
        activation=args.activation,
        name="main_output"
    )(x)

    # As metric functions, use accuracy per role and the total accuracy.
    metric_functions = [metrics.role_acc(*i) for i in roles_vocab.items()] + ["accuracy"]

    # Build the model.
    model = keras.models.Model(inputs=[words_input, features_input], outputs=[main_output])
    model.compile(
        OPTIMIZERS[args.optimizer](lr=10**args.log_learning_rate, amsgrad=True),
        loss=args.loss,
        metrics=metric_functions
    )

    # Print the model summary.
    model.summary(print_fn=LOG.debug)

    # ---------------------------------------------------------------------------------------------
    # Train the model.

    LOG.info("Training the model ...")

    # Define the callbacks.
    # callbacks = [
    #     # A callback for showing an advanced progress bar while training.
    #     utils.callback.ProgressBarCallback(
    #         log_level=args.log_level,
    #         dynamic=not IN_CLUSTER,
    #         update_freq=args.progress_bar_update_frequency
    #     )
    # ]

    # Train the model.
    model.fit(
        [word_seqs, layout_seqs], roles,
        validation_split=args.validation_split,
        epochs=args.epochs,
        shuffle=args.shuffle,
        # callbacks=callbacks,
        verbose=1
    )

    # ---------------------------------------------------------------------------------------------
    # Write all files to the output directory that are necessary to use the model for prediction.

    # Save the model.
    LOG.info(f"Saving the trained model to '{args.output_dir}' ...")
    model.save(args.output_dir, save_format="tf")

    # Write the args used to train the model to file, for reproducibility reasons and to be able to
    # use the same arguments on doing some prediction using the model.
    args_file_path = os.path.join(args.output_dir, ARGS_FILE_NAME)
    LOG.info(f"Writing model arguments to '{args_file_path}' ...")
    with open(args_file_path, "w") as fout:
        json.dump(vargs, fout)

    # Copy the used BPE vocabulary to the output directory.
    bpe_vocab_file_path = os.path.join(args.output_dir, "bpe-vocab.tsv")
    LOG.info(f"Copying the BPE vocabulary to '{bpe_vocab_file_path}' ...")
    shutil.copyfile(args.bpe_vocab_file, bpe_vocab_file_path)

    # Copy the used roles vocabulary to the output directory.
    roles_vocab_file_path = os.path.join(args.output_dir, "roles-vocab.tsv")
    LOG.info(f"Copying the roles vocabulary to '{roles_vocab_file_path}' ...")
    shutil.copyfile(args.roles_vocab_file, roles_vocab_file_path)

# =================================================================================================


if __name__ == "__main__":
    # Create a command line argument parser.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The path to the directory to read the input files from.
    arg_parser.add_argument(
        "-g", "--ground_truth_dir",
        type=str,
        required=False,
        default=GROUND_TRUTH_DIR,
        help="The path to the directory to read the ground truth files from."
    )

    # The path to the directory where to store the output files.
    arg_parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=False,
        default=OUTPUT_DIR,
        help="The path to the directory where to store the output files (e.g., the trained model)."
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

    # Whether to shuffle the ground truth files before selecting <num_files>-many files and
    # whether or not to shuffle the input sequences before feeding them into the model.
    arg_parser.add_argument(
        "--shuffle",
        type=argparse_types.boolean,
        required=False,
        default=SHUFFLE,
        help="Whether to shuffle the ground truth files before selecting <num_files> and "
             "whether or not to shuffle the input sequences before feeding them into the model."
    )

    # The path to the file with the vocabulary to use on encoding the word sequences with BPE.
    arg_parser.add_argument(
        "--bpe_vocab_file",
        type=str,
        required=False,
        default=BPE_VOCAB_FILE_PATH,
        help="The path to the file with the vocabulary to use on encoding the word sequences "
             "with byte pair encoding."
    )

    # The path to the file with the vocabulary to use on encoding the semantic roles.
    arg_parser.add_argument(
        "--roles_vocab_file",
        type=str,
        required=False,
        default=ROLES_VOCAB_FILE_PATH,
        help="The path to the file with the vocabulary to use on encoding the semantic roles."
    )

    # The word delimiters.
    arg_parser.add_argument(
        "--word_delimiters",
        type=argparse_types.escaped_char_sequence,
        default=WORD_DELIMITERS,
        help="The characters to be considered as word delimiters. Control characters like '\\n' "
             "and '\\t' need to be unescaped to '\\\\n' and '\\\\t'."
    )

    # Whether or not to translate the text of text blocks to lower cases.
    arg_parser.add_argument(
        "--lowercase_text",
        type=argparse_types.boolean,
        default=IS_LOWERCASE_TEXT,
        help="Whether or not to translate the text of text blocks to lower cases."
    )

    # The length of the words sequences.
    arg_parser.add_argument(
        "--word_seq_length",
        type=int,
        default=WORD_SEQ_LENGTH,
        help="The target length of the word sequences."
    )

    # Whether or not to include the positions of the text blocks in the layout feature vector.
    arg_parser.add_argument(
        "--include_positions",
        type=argparse_types.boolean,
        default=IS_INCLUDE_POSITIONS,
        help="Whether or not to include the positions of the text blocks in the layout features."
    )

    # Whether or not to include the font sizes of the text blocks in the feature vector.
    arg_parser.add_argument(
        "--include_font_sizes",
        type=argparse_types.boolean,
        default=IS_INCLUDE_FONT_SIZES,
        help="Whether or not to include the font sizes of the text blocks in the layout features."
    )

    # Whether or not to include the font styles of the text blocks in the feature vector.
    arg_parser.add_argument(
        "--include_font_styles",
        type=argparse_types.boolean,
        default=IS_INCLUDE_FONT_STYLES,
        help="Whether or not to include the font styles of the text blocks in the layout features."
    )

    # Whether or not to include the character features (e.g., the type of the characters of which
    # the text is composed of) in the input features.
    arg_parser.add_argument(
        "--include_char_features",
        type=argparse_types.boolean,
        default=IS_INCLUDE_CHAR_FEATURES,
        help="Whether or not the character features (e.g., the type of the characters of which "
             "the text is composed of) should be included in the layout features."
    )

    # Whether to include features about the text semantics (e.g., the probability that a text
    # denotes the name of a human or a country) in the feature vector.
    arg_parser.add_argument(
        "--include_semantic_features",
        type=argparse_types.boolean,
        default=IS_INCLUDE_SEMANTIC_FEATURES,
        help="Whether or not the semantic features (e.g., the probability that a text denotes "
             "the name of a human or a country) should be included in the feature vector."
    )

    # The path to a database file providing country names.
    arg_parser.add_argument(
        "--countries_db_file",
        type=str,
        help="The path to a file providing country names (in the format: one country per line)."
    )

    # The path to a database file providing human names.
    arg_parser.add_argument(
        "--human_names_db_file",
        type=str,
        help="The path to a file providing human names (in the format: one name per line)."
    )

    # The dropout.
    arg_parser.add_argument(
        "--dropout",
        type=float,
        required=False,
        default=DROPOUT,
        help="The dropout value to use."
    )

    # The activation function.
    arg_parser.add_argument(
        "--activation",
        type=str,
        required=False,
        default=ACTIVATION,
        help="The activation function to use."
    )

    # The loss function.
    arg_parser.add_argument(
        "--loss",
        type=str,
        required=False,
        default=LOSS,
        help="The loss function to use."
    )

    # The optimizer function.
    arg_parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default=OPTIMIZER,
        help="The optimizer function to use."
    )

    # The (logarithmic) learning rate.
    arg_parser.add_argument(
        "--log_learning_rate",
        type=int,
        default=LOG_LEARNING_RATE,
        help="The (logarithmic) learning rate."
    )

    # The validation split.
    arg_parser.add_argument(
        "--validation_split",
        type=float,
        required=False,
        default=VALIDATION_SPLIT,
        help="The validation split to use."
    )

    # The number of epochs.
    arg_parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=EPOCHS,
        help="The number of epochs to use."
    )

    # The update frequency of the progress bar.
    arg_parser.add_argument(
        "--progress_bar_update_frequency",
        type=argparse_types.num,
        default=PROGRESS_BAR_UPDATE_FREQUENCY,
        help="The update frequency of the progress bar."
    )

    # The log level.
    arg_parser.add_argument(
        "--logging_level",
        type=str,
        default=LOGGING_LEVEL,
        help="The logging level."
    )

    args = arg_parser.parse_args()

    # Set the logging level.
    LOG.setLevel(log_utils.to_log_level(args.logging_level))

    # Start the training.
    train(args)
