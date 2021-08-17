import os
import tensorflow as tf
from tensorflow.keras import backend as K
K.set_learning_phase(0)

from tensorflow.keras.models import load_model

import utils.files
import utils.metrics

# ==================================================================================================

K.set_learning_phase(0)

# Read the roles voabulary.
roles_vocab = utils.files.read_vocabulary_file("./data/vocab-roles.tsv")
roles_vocab['UNK_ROLE'] = len(roles_vocab)

# Load the model.
custom_metrics = {}
for i in roles_vocab.items():
    f = utils.metrics.role_acc(*i)
    custom_metrics[f.__name__] = f

model = load_model("IChXO-model.h5", custom_objects=custom_metrics)
model.save("model_tf2", save_format="tf")
