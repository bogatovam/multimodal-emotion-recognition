from enum import Enum

import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeature, TimeDependentModality


class AudioFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, *output_shape):
        self.output_shape = output_shape

    # L3 = (46, 32, 24, 512),
    L3 = (1, 48000)  # just after net preprocessor
    OPENSMILE_GeMAPSv01b = (1, 62)
    OPENSMILE_eGeMAPSv02 = (1, 88)
    OPENSMILE_ComParE_2016 = (1, 6373)


class AudioModalityPreprocessor(BaseDatasetProcessor):

    def __init__(self, extractor: AudioFeatureExtractor,
                 center: bool = True,
                 target_sr: int = 48000):
        self._extractor = extractor
        self.output_shape = extractor.output_shape

        self._center = center
        self._target_sr = target_sr

        self._feature_description = {
            DatasetFeature.L3.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.OPENSMILE_GeMAPSv01b.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.OPENSMILE_eGeMAPSv02.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.OPENSMILE_ComParE_2016.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_SCENE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_SHAPE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_FACE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)

        if self._extractor == AudioFeatureExtractor.L3:
            dataset = dataset.flat_map(self._l3_preprocess_audio)
        else:
            dataset = dataset.map(self.concat_with_labels, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def concat_with_labels(self, example: tf.train.Example):
        return (example[DatasetFeature.AUDIO.name]), (example[DatasetFeature.CLASS.name])

    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        clazz = tf.io.parse_tensor(example[DatasetFeature.CLASS.name], tf.int32)
        clazz = tf.cast(clazz, dtype=tf.float32)
        clazz = tf.ensure_shape(clazz, 1)

        return {
            self._extractor.name: tf.io.parse_tensor(example[self._extractor.name], tf.float32),
            DatasetFeature.CLASS.name: clazz
        }

    @tf.function
    def _l3_preprocess_audio(self, example: tf.train.Example):
        audio_frames = example[DatasetFeature.L3.name]
        clazz = example[DatasetFeature.CLASS.name]

        audio_frames = tf.ensure_shape(audio_frames, shape=(None, *self._extractor.output_shape))
        return tf.data.Dataset.from_tensor_slices(audio_frames).map(lambda x: (x, clazz))
