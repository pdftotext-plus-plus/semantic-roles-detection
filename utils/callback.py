import math

import tensorflow.keras as keras

import utils.log


class ProgressBarCallback(keras.callbacks.ProgbarLogger):
    """
    A callback that prints a custom progress bar while training a Keras model. This callback is
    required to get a human-readable progress bar in the log output while training a model in
    Polyaxon (with one line per progress step and configurable update interval).
    """
    def __init__(self, log_level=None, dynamic=True, update_freq=None):
        """
        Creates a new progress bar callback.

        Args:
            log_level (str):
                The logging level (one of "fatal", "error", "warn", "info", "debug").
            dynamic (bool):
                Whether or not the progress bar should be displayed dynamically. If set to true
                and the verbose flag == 1, the progress of each epoch is displayed dynamically in
                a single line. If set to False and the verbose flag is == 1, each update of the
                progress is displayed in a separate line.
            update_freq (float or int):
                The frequency with which the progress bar should be updated. If this freqency is
                an int i, the progress bar is updated on every i-th batch. If the frequency is a
                float f (0 < f < 1), the progress bar is updated on every (#batches*f)-th batch.
                Otherwise, the progress bar is updated whenever a batch ended.
        """
        super(ProgressBarCallback, self).__init__(count_mode="samples", stateful_metrics=None)
        # We need to translate the log level to a keras' verbose flag:
        # 0 = silent, 1 = progress bar, 2 = one line per epoch
        self.verbose_flag = utils.log.to_keras_verbose_flag(log_level)
        self.dynamic = dynamic
        self.update_freq = update_freq
        self.target = 0
        self.log_values = []

    def on_train_begin(self, logs=None):
        super(ProgressBarCallback, self).on_train_begin(logs)
        self.verbose = self.verbose_flag

    def on_epoch_begin(self, epoch, logs=None):
        super(ProgressBarCallback, self).on_epoch_begin(epoch, logs)
        if self.verbose:
            self.progbar._dynamic_display = self.dynamic
        # The sample for which the progress bar should be updated.
        self.next_update = self.params["batch_size"]

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = self.params.get("batch_size", 0)
        num_samples = self.params.get("samples", 0)
        num_batches = math.ceil(num_samples / batch_size) if batch_size > 0 else 0

        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and (self.target is None or self.seen < self.target):
            # If no update frequency is given, update the progress bar instantly.
            if self.update_freq is None:
                self.progbar.update(self.seen, self.log_values)

            if self.seen == self.next_update:
                self.progbar.update(self.seen, self.log_values)

                factor = 1
                # Increment the next update batch index.
                if isinstance(self.update_freq, int):
                    factor = self.update_freq
                elif isinstance(self.update_freq, float):
                    factor = int(self.update_freq * num_batches)
                self.next_update += factor * batch_size

# =================================================================================================


class PolyaxonCallback(keras.callbacks.Callback):
    """
    A callback that logs metrics to polyaxon after each epoch.
    """
    def __init__(self, experiment):
        self.experiment = experiment

    def on_epoch_end(self, batch, logs={}):
        if self.experiment:
            self.experiment.log_metrics(step=batch, **logs)
