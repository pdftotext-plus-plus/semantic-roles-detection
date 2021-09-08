"""
A script to use a trained deep learning model for predicting the semantic roles of given text
blocks extracted from PDF files.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

import json
import logging
import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log output of tensorflow  # NOQA

import numpy as np

from semantic_roles_detection.utils import dotdict
from semantic_roles_detection.utils import feature_encoder
from semantic_roles_detection.utils import log_utils
from semantic_roles_detection.utils import metrics
from semantic_roles_detection.utils import vocab_utils
from semantic_roles_detection.utils.models import Page, TextBlock

import tensorflow.keras as keras

# =================================================================================================
# Configure the logging.

LOG = log_utils.get_logger(__name__)
LOG.setLevel(log_utils.to_log_level("error"))

# Disable the debug output of some third-party libraries.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# =================================================================================================
# Parameters.

# The parent directory of this script.
PARENT_DIR = os.path.dirname(__file__)
# The directory of the model to use for the prediction.
# MODEL_DIR = os.path.join(PARENT_DIR, "models/2021-08-13_model-with-header-footers-included")
MODEL_DIR = os.path.join(PARENT_DIR, "models/2021-08-30_model-for-pdftotei-text-block-extraction")
# The name of the args file.
ARGS_FILE_NAME = "model-args.json"
# The name of the BPE vocab file.
BPE_VOCAB_FILE_NAME = "bpe-vocab.tsv"
# The name of the roles vocab file.
ROLES_VOCAB_FILE_NAME = "roles-vocab.tsv"

# =================================================================================================


def predict(blocks: List[TextBlock], pages: List[Page]):
    """
    This method predicts the semantic roles of the given text blocks extracted from PDF. Note that
    this method assumes that all text blocks were extracted from the same PDF document. The
    predicted semantic roles are written to the `role` attribute of the text blocks.

    Args:
        blocks: List[TextBlock]
            The text blocks for which to predict the semantic roles.
        pages: List[Page]
            The pages of the PDF document from which the text blocks were extracted, each
            together with its page number, page height and page width. Note that it is assumed that
            the pages are sorted in ascending order (with the first page being the first element of
            this list).
    """
    # Do nothing if there are no blocks.
    if not blocks:
      return

    # ---------------------------------------------------------------------------------------------
    # Read the args file of the model (containing the parameters used on training the model).

    # Get the path to the args file.
    args_file_path = os.path.join(MODEL_DIR, ARGS_FILE_NAME)
    LOG.info(f"Reading args file '{args_file_path}' ...")

    if not os.path.exists(args_file_path):
        raise ValueError(f"Args file '{args_file_path}' not found.")

    # Read the args file.
    with open(args_file_path, "rb") as stream:
        args = dotdict.DotDict(json.load(stream))

    # ---------------------------------------------------------------------------------------------
    # Read the ground truth.

    LOG.info("Reading the vocabulary files ...")
    vocab_reader = vocab_utils.VocabularyReader()
    bpe_vocab = vocab_reader.read(os.path.join(MODEL_DIR, BPE_VOCAB_FILE_NAME))
    roles_vocab = vocab_reader.read(
        os.path.join(MODEL_DIR, ROLES_VOCAB_FILE_NAME))

    # Encode the text blocks.
    LOG.info("Encoding the text ground truth ...")
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
    word_seqs, layout_seqs, _ = encoder.encode_text_blocks(blocks, pages)

    if len(word_seqs) == 0:
        raise ValueError("No word sequences given.")

    if len(layout_seqs) == 0:
        LOG.error("No features given.")
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

    # ----------------------------------------------------------------------------------------------
    # Predict the semantic roles.

    LOG.info("Predicting ...")

    # For each text block, get a probability distribution of the semantic roles.
    predicted_roles = model.predict([word_seqs, layout_seqs], verbose=1)
    # For each text block, get the id of the role with the highest probability.
    predicted_roles = np.argmax(predicted_roles, axis=1)
    # Translate the ids to role names.
    for i in range(len(predicted_roles)):
        blocks[i].role = encoder.rev_roles_vocab[predicted_roles[i]]

if __name__ == "__main__":
    page = Page(page_num=1, width=500, height=720)

    block1 = TextBlock(
      id = 123,
      text = "Hello World",
      page_num = 1,
      lower_left_x = 30,
      lower_left_y = 70,
      upper_right_x = 120,
      upper_right_y = 140,
      font_name = "Arial",
      font_size = 12,
      is_bold=False,
      is_italic=True
    )

    block2 = TextBlock(
      id = 123,
      text = "",
      page_num = 1,
      lower_left_x = 30,
      lower_left_y = 70,
      upper_right_x = 120,
      upper_right_y = 140,
      font_name = "Arial",
      font_size = 12,
      is_bold=False,
      is_italic=True
    )

    predict([block1, block2], [page])