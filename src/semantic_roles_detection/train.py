"""
A script to train a deep learning model for predicting the semantic roles of text blocks extracted
from PDF files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import json
import logging
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log output of tensorflow  # NOQA

import tensorflow.keras as keras

from semantic_roles_detection.utils import feature_encoder
from semantic_roles_detection.utils import ground_truth_reader
from semantic_roles_detection.utils import log_utils
from semantic_roles_detection.utils import metrics
from semantic_roles_detection.utils import vocab_utils


# =================================================================================================
# Configure the logging.

LOG = log_utils.get_logger(__name__)

# Disable the debug output of some third-party libraries.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# =================================================================================================
# Parameters.

# The default file name of the args file.
ARGS_FILE_NAME = "model-args.json"
# The availbale optimizers.
OPTIMIZERS = {"adam": keras.optimizers.Adam}

# =================================================================================================


def train(args):
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

    # Set the logging level.
    LOG.setLevel(log_utils.to_log_level(args.logging_level))

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