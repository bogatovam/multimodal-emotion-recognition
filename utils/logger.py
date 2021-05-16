import tensorflow as tf

from utils.callbacks.log_batch_loss_callback import LogBatchLossCallBack
from utils.dirs import create_dirs


class Logger:

    @staticmethod
    def get_logger_callbacks(board_path: str, log_freq: int, create_dirs_flag: bool, epoch: int = 0,
                             iteration: int = 0):
        return [
            Logger.get_tensorboard_logger(board_path, log_freq, create_dirs_flag),
            Logger.get_batch_loss_logger(board_path, log_freq, epoch, iteration, create_dirs_flag)
        ]

    @staticmethod
    def get_tensorboard_logger(board_path: str, log_freq: int, create_dirs_flag: bool) -> tf.keras.callbacks.Callback:
        if create_dirs_flag:
            create_dirs([board_path])

        return tf.keras.callbacks.TensorBoard(log_dir=board_path,
                                              update_freq='epoch',
                                              histogram_freq=log_freq,
                                              write_images=True,
                                              embeddings_freq=log_freq)

    @staticmethod
    def get_batch_loss_logger(board_path: str, log_freq: int, epoch: int,
                              iteration: int,
                              create_dirs_flag: bool) -> tf.keras.callbacks.Callback:
        return LogBatchLossCallBack(board_path, log_freq, epoch, iteration, create_dirs_flag)
