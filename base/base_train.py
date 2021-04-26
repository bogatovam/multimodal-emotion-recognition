from abc import ABC, abstractmethod
import tensorflow as tf


class BaseTrain(ABC):
    """ Базовый класс, описывающий процесс обучения модели """

    def __init__(self, model, data):
        self.model = model
        self.data = data

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @staticmethod
    def get_learning_rate_scheduler_callback(func):
        def get_learning_rate(epoch):
            return func[epoch]

        return tf.keras.callbacks.LearningRateScheduler(schedule=get_learning_rate, verbose=1)

    @staticmethod
    def _get_terminate_on_nan_callback():
        return tf.keras.callbacks.TerminateOnNaN()
