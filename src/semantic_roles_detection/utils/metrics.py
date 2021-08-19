"""
This file is part of the "semantic-roles-detection" module of PdfActML. It contains some custom
metrics to be printed to the console while training a model and predicting semantic roles using a
model.

Copyright 2021, University of Freiburg.

Claudius Korzen <korzen@cs.uni-freiburg.de>
"""

from typing import Callable

import tensorflow.keras.backend as K

# ==================================================================================================


def role_acc(role_name: str, role_id: int) -> Callable:
    """
    This method returns a metric function that computes the accuracy of a model in predicting the
    given semantic role.

    Args:
        role_name: str
            The name of the role.
        role_id: int
            The role in encoded form.
    Returns:
        Callable
            The created metric function.
    """

    def fn(y_true, y_pred):
        # The ids of the true roles.
        role_ids_true = K.argmax(y_true, axis=-1)
        # The ids of the predicted roles.
        role_ids_preds = K.argmax(y_pred, axis=-1)

        accuracy_mask = K.cast(K.equal(role_ids_true, role_id), "int32")
        role_acc_tensor = K.cast(K.equal(role_ids_true, role_ids_preds), "int32") * accuracy_mask

        # The number of correct predictions (with regard to the given semantic role).
        num_correct = K.sum(role_acc_tensor)
        # The number how often the given role occurs in role_ids_true in total.
        num_roles = K.sum(accuracy_mask)
        # Avoid a division by zero error: If num_roles is 0, return an accuracy of 1.
        return K.switch(K.equal(num_roles, 0), K.cast(1, "float64"), num_correct / num_roles)

    # Set the name of the metric. This name is used as a label on outputting the value to the
    # console, so use a descriptive name.
    fn.__name__ = "acc_{}".format(role_name)
    return fn
