from abc import ABC, abstractmethod
import tensorflow as tf


class BaseDatasetProcessor(ABC):

    @abstractmethod
    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        raise NotImplementedError

