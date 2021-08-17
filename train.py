"""
A script to train a deep learning model aimed to detect the semantic roles of building blocks from
PDF files.
"""

import argparse
import json
import logging
import os
import shutil

import data_reader

import tensorflow.keras as keras

import utils.callback
import utils.log
import utils.metrics
import utils.type

# =================================================================================================

# Check if polyaxon is installed and if we are in "cluster-mode".
try:
    import polyaxon_client as polyaxon
    import polyaxon_client.tracking as polytracking
    IN_CLUSTER = polyaxon.settings.IN_CLUSTER
except ImportError:
    IN_CLUSTER = False

# The logger.
LOG = utils.log.get_logger(__name__)

# =================================================================================================
# Settings.

# The logging level.
LOG_LEVEL = polytracking.get_log_level() if IN_CLUSTER and polytracking.get_log_level() else "info"

# The default input directory.
INPUT_DIR = "/input"
# The prefix of the files to read from the input directory.
INPUT_FILES_PREFIX = ""
# The suffix of the files to read from the input directory.
INPUT_FILES_SUFFIX = ".tsv"
# The maximum number of input files to read from the input directory (set to -1 to read all files).
MAX_NUM_INPUT_FILES = -1
# Whether to shuffle the input files before selecting <max_num_files> and whether to shuffle
# the sequences in the input array passed to the model.
SHUFFLE = True
# The default directory where to store output files (e.g., the trained model).
OUTPUT_DIR = "/output"
# The file name pattern for the args output file.
ARGS_FILE_NAME_PATTERN = "model_args.json"
# The default path to the words vocab file.
WORDS_VOCAB_FILE = "/data/vocabs/vocab-words.tsv"
# The default path to the roles vocab file.
ROLES_VOCAB_FILE = "/data/vocabs/vocab-roles.tsv"

# The characters to be considered as word delimiters.
# WORD_DELIMITERS = "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n "
WORD_DELIMITERS = "\r\f\t\n "
# The number of words each input sequence should include from a building block.
NUM_WORDS_PER_SEQ = 100
# The encoding to use for the words.
ENCODING = "bpe"
# Whether or not words should be lowercased.
LOWERCASE_WORDS = False
# Whether or not the positions of the building blocks should be included in the feature vector.
INCLUDE_POSITIONS = True
# Whether or not the font sizes of the building blocks should be included in the feature vector.
INCLUDE_FONT_SIZES = True
# Whether or not the font styles of the building blocks should be included in the feature vector.
INCLUDE_FONT_STYLES = True
# Whether or not the character features should be included in the feature vector.
INCLUDE_CHAR_FEATURES = True
# Whether or not the semantic features should be included in the feature vector.
INCLUDE_SEMANTIC_FEATURES = True

# The dropout value.
DROPOUT = 0.2
# The activation function.
ACTIVATION = "softmax"
# The loss function.
LOSS = "categorical_crossentropy"
# The name of the optimizer function.
OPTIMIZER_NAME = "adam"
# The (logarithmic) learning rate.
LOG_LEARNING_RATE = -3
# The validation split.
VALIDATION_SPLIT = 0.1
# The number of epochs.
EPOCHS = 10
# The boolean value that indicates whether tensorboard log files should be created.
USE_TENSORBOARD = False
# The update frequency of the progress bar.
PROGRESS_BAR_UPDATE_FREQUENCY = None
# The available optimizer functions, per name.
OPTIMIZERS = {
  "adam": keras.optimizers.Adam,
}

# =================================================================================================


def main(args):
    """
    The main method.

    Args:
        args (namespace):
            The command line arguments.
    """
    # Configure the logging level.
    LOG.setLevel(utils.log.to_log_level(args.log_level))
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("polyaxon.client").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    vargs = vars(args)
    LOG.info("Training a model for detecting semantic roles of building blocks ...")
    for name in vargs:
        LOG.debug(" - {}: {}".format(name, vargs[name]))

    # Create a new Polyaxon experiment, e.g., to send the used parameters to the Polyaxon UI.
    experiment = polytracking.Experiment() if IN_CLUSTER else None
    if experiment:
        experiment.log_params(**vargs)

    # ---------------------------------------------------------------------------------------------
    # Read and preprocess the input data.

    LOG.info("Reading the data ...")

    # Read the input files.
    reader = data_reader.DataReader(
        input_dir=args.input_dir,
        input_file_name_pattern="{0}*{1}".format(args.prefix, args.suffix),
        max_num_input_files=args.max_num_files,
        shuffle_input_files=args.shuffle,
        encoding=args.encoding,
        words_vocab=utils.files.read_vocabulary_file(args.words_vocab_file),
        roles_vocab=utils.files.read_vocabulary_file(args.roles_vocab_file),
        word_delimiters=args.word_delimiters,
        lowercase_words=args.lowercase_words,
        num_words_per_seq=args.num_words_per_seq,
        include_positions=args.include_positions,
        include_font_sizes=args.include_font_sizes,
        include_font_styles=args.include_font_styles,
        include_char_features=args.include_char_features,
        include_semantic_features=args.include_semantic_features,
        countries_db_file=args.countries_db_file,
        human_names_db_file=args.human_names_db_file,
    )
    word_seqs, features, roles = reader.read()

    if len(word_seqs) == 0:
        LOG.error("No word sequences given.")
        return

    if len(roles) == 0:
        LOG.error("No roles given.")
        return

    # ---------------------------------------------------------------------------------------------
    # Print some statistics.

    LOG.debug("Data statistics:")
    LOG.debug(" - # word sequences: {}".format(word_seqs.shape[0]))
    LOG.debug(" - length of word sequences: {}".format(word_seqs.shape[1]))
    LOG.debug(" - # feature vectors: {}".format(features.shape[0]))
    LOG.debug(" - length of feature vectors: {}".format(features.shape[1]))
    LOG.debug(" - # role vectors: {}".format(roles.shape[0]))
    LOG.debug(" - length of role vectors: {}".format(roles.shape[0]))

    if experiment:
        experiment.log_params(stats_num_word_seqs=word_seqs.shape[0])
        experiment.log_params(stats_len_word_seqs=word_seqs.shape[1])
        experiment.log_params(stats_num_feature_vectors=features.shape[0])
        experiment.log_params(stats_len_feature_vectors=features.shape[1])
        experiment.log_params(stats_num_role_vectors=roles.shape[0])
        experiment.log_params(stats_len_role_vectors=roles.shape[1])

    # Sort the roles distribution by values, in descending order.
    roles_distr = sorted(reader.roles_dist.items(), key=lambda x: x[1], reverse=True)

    LOG.debug(" - distribution of roles:")
    for role, perc in roles_distr:
        LOG.debug("   - {}: {:.3f}".format(role, perc))
        if experiment:
            experiment.log_params(**{"stats_roles_distr_{}".format(role): "{:.3f}".format(perc)})

    # ---------------------------------------------------------------------------------------------
    # Build the model.

    LOG.info("Building the model ...")

    # Define the input layer for the word sequences.
    words_input = keras.layers.Input(shape=(word_seqs.shape[1],), dtype="int32", name="words_input")

    # Define an embedding layer that encodes the word sequences into lower-dimensional space.
    words_embedding = keras.layers.Embedding(
        output_dim=256,
        input_dim=len(reader.words_vocab),
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

    # Define the input layer for further features (e.g., font and position information).
    features_input = keras.layers.Input(shape=(features.shape[1],), name="features_input")

    # Concatenate the "words_lstm" layer and the "features_input" layer.
    x = keras.layers.concatenate([words_lstm, features_input])

    x = keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(args.dropout, name="dropout")(x)

    # Define the output layer.
    main_output = keras.layers.Dense(len(reader.roles_vocab), activation=args.activation, name="main_output")(x)

    # As metrics, use accuracy per role and the total accuracy.
    metrics = [utils.metrics.role_acc(*i) for i in reader.roles_vocab.items()] + ["accuracy"]

    # Build the model.
    model = keras.models.Model(inputs=[words_input, features_input], outputs=[main_output])
    model.compile(
        OPTIMIZERS[args.optimizer](lr=10**args.log_learning_rate, amsgrad=True),
        loss=args.loss,
        metrics=metrics
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

    # if IN_CLUSTER:
    #     # A callback for passing the metric values to the Polyaxon UI.
    #     callbacks.append(utils.callback.PolyaxonCallback(experiment))
    #     if args.use_tensorboard:
    #         # A callback for logging the metrics values to Tensorboard.
    #         callbacks.append(keras.callbacks.TensorBoard(polytracking.get_outputs_path()))

    # Train the model.
    model.fit(
        [word_seqs, features], roles,
        validation_split=args.validation_split,
        epochs=args.epochs,
        shuffle=args.shuffle,
        # callbacks=callbacks,
        # ----Don't be verbose, the progress is displayed by the ProgressBarCallback defined above.
        verbose=1
    )

    # ---------------------------------------------------------------------------------------------
    # Write all files to the output directory that are necessary to use the model for prediction.

    # Save the model.
    LOG.info(f"Saving the trained model to '{args.output_dir}' ...")
    model.save(args.output_dir, save_format="tf")

    # Save the args used to train the model to file, for reproducibility reasons and to be able to
    # use the same arguments on doing some prediction using the model.
    args_file_path = os.path.join(args.output_dir, ARGS_FILE_NAME_PATTERN)
    LOG.info(f"Saving the used arguments to '{args_file_path}' ...")
    with open(args_file_path, "w") as fout:
        json.dump(vargs, fout)

    # Copy the used vocabularies to the output directory.
    bpe_vocab_file_path = os.path.join(args.output_dir, "vocab_bpe.tsv")
    LOG.info(f"Copying the BPE vocabulary to '{bpe_vocab_file_path}' ...")
    shutil.copyfile(args.words_vocab_file, bpe_vocab_file_path)

    # Copy the used roles vocabulary to the output directory.
    roles_vocab_file_path = os.path.join(args.output_dir, "vocab_roles.tsv")
    LOG.info(f"Copying the roles vocabulary to '{roles_vocab_file_path}' ...")
    shutil.copyfile(args.roles_vocab_file, roles_vocab_file_path)

    # Return some objects such that they can be tested.
    # return word_seqs, features, roles, roles_distr, model_file_path, args_file_path
    return word_seqs, features, roles, roles_distr, args_file_path

# =================================================================================================


if __name__ == "__main__":
    # Create a command line argument parser.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The path to the directory to read the input files from.
    arg_parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=False,
        default=INPUT_DIR,
        help="The path to the directory to read the input files from."
    )

    # The path to the directory where to store the output files.
    arg_parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=False,
        default=OUTPUT_DIR,
        help="The path to the directory where to store the output files (e.g., the trained model)."
    )

    # The prefix of the input files in the input directory.
    arg_parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default=INPUT_FILES_PREFIX,
        help="The prefix of the input files in the input directory."
    )

    # The suffix of the input files in the input directory.
    arg_parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        default=INPUT_FILES_SUFFIX,
        help="The suffix of the input files in the input directory."
    )

    # The maximum number of files to read.
    arg_parser.add_argument(
        "--max_num_files",
        type=int,
        required=False,
        default=MAX_NUM_INPUT_FILES,
        help="The max. number of files to read from the directory (set to -1 to read all files)."
    )

    # Whether to shuffle the input files before selecting <max_num_files> and whether to shuffle
    # the sequences in the input array passed to the model.
    arg_parser.add_argument(
        "--shuffle",
        type=utils.type.boolean,
        required=False,
        default=SHUFFLE,
        help="Whether to shuffle the input files before selecting <max_num_files> and whether to \
shuffle the sequences in the input array passed to the model."
    )

    # The name of the encoding to use to encode the word sequences.
    arg_parser.add_argument(
        "--encoding",
        type=str,
        default=ENCODING,
        help="The name of the encoding to use to encode the word sequences."
    )

    # The path to the file with the vocabulary to use on encoding the word sequences.
    arg_parser.add_argument(
        "--words_vocab_file",
        type=str,
        required=False,
        default=WORDS_VOCAB_FILE,
        help="The path to the file with the vocabulary to use on encoding the word sequences."
    )

    # The path to the file with the vocabulary to use on encoding the semantic roles.
    arg_parser.add_argument(
        "--roles_vocab_file",
        type=str,
        required=False,
        default=ROLES_VOCAB_FILE,
        help="The path to the file with the vocabulary to use on encoding the semantic roles."
    )

    # The word delimiters.
    arg_parser.add_argument(
        "--word_delimiters",
        type=utils.type.escaped_char_sequence,
        default=WORD_DELIMITERS,
        help="The characters to be considered as word delimiters. Control characters like '\\n' \
and '\\t' need to be unescaped to '\\\\n' and '\\\\t'."
    )

    # Whether to lowercase the words.
    arg_parser.add_argument(
        "--lowercase_words",
        type=utils.type.boolean,
        default=LOWERCASE_WORDS,
        help="Whether or not the words should be lowercased."
    )

    # The number of words per sequence.
    arg_parser.add_argument(
        "--num_words_per_seq",
        type=int,
        default=NUM_WORDS_PER_SEQ,
        help="The number of words each word sequence should include."
    )

    # Whether to include the positions of the building blocks in the feature vector.
    arg_parser.add_argument(
        "--include_positions",
        type=utils.type.boolean,
        default=INCLUDE_POSITIONS,
        help="Whether to include the positions of the building blocks in the feature vector."
    )

    # Whether to include the font sizes of the building blocks in the feature vector.
    arg_parser.add_argument(
        "--include_font_sizes",
        type=utils.type.boolean,
        default=INCLUDE_FONT_SIZES,
        help="Whether to include the font sizes of the building blocks in the feature vector."
    )

    # Whether to include the font styles of the building blocks in the feature vector.
    arg_parser.add_argument(
        "--include_font_styles",
        type=utils.type.boolean,
        default=INCLUDE_FONT_STYLES,
        help="Whether to include the font styles of the building blocks in the feature vector."
    )

    # Whether to include the character features (e.g., the type of the characters of which the text
    # is composed of) in the input features.
    arg_parser.add_argument(
        "--include_char_features",
        type=utils.type.boolean,
        default=INCLUDE_CHAR_FEATURES,
        help="Whether or not the character features (e.g., the type of the characters of which \
the text is composed of) should be included in the feature vector."
    )

    # Whether to include features about the text semantics (e.g., the probability that a text
    # denotes the name of a human or a country) in the feature vector.
    arg_parser.add_argument(
        "--include_semantic_features",
        type=utils.type.boolean,
        default=INCLUDE_SEMANTIC_FEATURES,
        help="Whether or not the semantic features (e.g., the probability that a text denotes the \
name of a human or a country) should be included in the feature vector."
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
        default=OPTIMIZER_NAME,
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

    # Whether to use tensorboard.
    arg_parser.add_argument(
        "--use_tensorboard",
        type=utils.type.boolean,
        default=USE_TENSORBOARD,
        help="Whether or not tensorboard should be used."
    )

    # The update frequency of the progress bar.
    arg_parser.add_argument(
        "--progress_bar_update_frequency",
        type=utils.type.num,
        default=PROGRESS_BAR_UPDATE_FREQUENCY,
        help="The update frequency of the progress bar."
    )

    # The log level.
    arg_parser.add_argument(
        "--log_level",
        type=str,
        default=LOG_LEVEL,
        help="The log level."
    )

    # Start the training.
    main(arg_parser.parse_args())
