import os

import tensorflow as tf

from utils.dirs import create_dirs
import logging as log


class LogBatchLossCallBack(tf.keras.callbacks.Callback):
    def __init__(self, board_dir: str, log_freq: int, _initial_epoch, initial_batch, create_dirs_flag: bool):
        super(LogBatchLossCallBack, self).__init__()

        self._log_freq = log_freq
        self._board_dir = board_dir
        self._current_total_iteration = initial_batch if initial_batch else 0

        log_path = os.path.join(self._board_dir, 'metrics')
        if create_dirs_flag:
            create_dirs([log_path])

        log.info("Total batch loss will be saved to {}".format(log_path))
        log.info("Initial iteration {}".format(initial_batch))

        self._file_writer = tf.summary.create_file_writer(log_path)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if batch % self._log_freq == 0:
            batch_loss_value = logs['loss']

            log.debug("batch: {}, step: {}".format(batch, self._current_total_iteration))
            with self._file_writer.as_default():
                tf.summary.scalar('batch loss', data=batch_loss_value, step=self._current_total_iteration)
                self._file_writer.flush()

        self._current_total_iteration += 1
        return super().on_batch_end(batch, logs)
