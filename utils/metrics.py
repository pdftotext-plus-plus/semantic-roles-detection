import tensorflow.keras.backend as K


def role_acc(role_name, role_id):
    """
    Returns a metric function that computes the accuracy for the given role.
    """

    def fn(y_true, y_pred):
        # The ids of the true roles.
        role_ids_true = K.argmax(y_true, axis=-1)
        # The ids of the predicted roles.
        role_ids_preds = K.argmax(y_pred, axis=-1)

        accuracy_mask = K.cast(K.equal(role_ids_true, role_id), "int32")
        role_acc_tensor = K.cast(K.equal(role_ids_true, role_ids_preds), "int32") * accuracy_mask

        # The number of correct predictions (restricted to the given role).
        num_correct = K.sum(role_acc_tensor)
        # The number how often the given role occurs in role_ids_true in total.
        num_roles = K.sum(accuracy_mask)
        # Avoid a division by zero error: If num_roles is 0, return an accuracy of 1.
        return K.switch(K.equal(num_roles, 0), K.cast(1, "float64"), num_correct / num_roles)

    # The function name is used as a label on outputting the value, so use a descriptive name.
    fn.__name__ = "acc_{}".format(role_name)
    return fn
