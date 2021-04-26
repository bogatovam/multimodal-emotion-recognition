from abc import ABC, abstractmethod
import tensorflow as tf


class BaseDatasetProcessor(ABC):

    @abstractmethod
    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        raise NotImplementedError

    @abstractmethod
    def map_record_to_dictionary_of_tensors(self, serialized_example):
        raise NotImplementedError

    @abstractmethod
    def _decode_example(self, serialized_example: tf.Tensor):
        raise NotImplementedError
