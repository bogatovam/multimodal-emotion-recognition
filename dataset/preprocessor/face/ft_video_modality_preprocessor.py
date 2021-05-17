from enum import Enum

import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeature, TimeDependentModality
from dataset.preprocessor.feature_extractors_metadata import VideoFeatureExtractor


class FineTuneVideoModalityPreprocessor(BaseDatasetProcessor):

    def __init__(self, extractor: VideoFeatureExtractor):
        self._extractor = extractor
        self.output_shape = extractor.output_shape

        self._feature_description = {
            DatasetFeature.L3.name: tf.io.FixedLenFeature([], tf.string),
            # DatasetFeature.VIDEO_SCENE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.VIDEO_FACE_RAW.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeature.CLASS.name: tf.io.FixedLenFeature([], tf.string)
        }

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors, num_parallel_calls=parallel_calls)
        if self._extractor == VideoFeatureExtractor.C3D:
            dataset = dataset.flat_map(self._C3D_preprocess_frames)
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

        video_fragment = tf.io.parse_tensor(example[DatasetFeature.VIDEO_FACE_RAW.name], tf.uint8)
        # video_fragment = example[DatasetFeature.VIDEO_SCENE_RAW.name]
        clazz = tf.io.parse_tensor(example[DatasetFeature.CLASS.name], tf.double)

        clazz = tf.cast(clazz, dtype=tf.float32)
        video_fragment = tf.cast(video_fragment, dtype=tf.float32)

        clazz = tf.ensure_shape(clazz, 9)

        return {
            DatasetFeature.VIDEO_FACE_RAW.name: video_fragment,
            DatasetFeature.CLASS.name: clazz
        }

    @tf.function
    def _C3D_preprocess_frames(self, example: tf.train.Example):
        video_frames = example[DatasetFeature.VIDEO_FACE_RAW.name]
        clazz = example[DatasetFeature.CLASS.name]

        # video_frames = tf.ensure_shape(video_frames, shape=(None, None, None, None))
        video_frames = tf.map_fn(self._decode_image, video_frames)
        video_frames = tf.ensure_shape(video_frames, shape=(None, 112, 112, 3))
        # video_frames = tf.expand_dims(video_frames, axis=0)
        print(video_frames.shape)

        # tf.signal.frame(audio, 32, 8)
        video_frames = tf.signal.frame(video_frames, 32, 8, axis=0)
        # video_frames = tf.reshape(video_frames, shape=(None, 16, 112, 112, 3))
        print(video_frames.shape)
        return tf.data.Dataset.from_tensor_slices(video_frames).map(lambda x: (x, clazz))

    @tf.function
    def _decode_image(self, image: tf.Tensor):
        # image = tf.image.decode_jpeg(image, channels=3)
        # todo
        image = tf.ensure_shape(image, shape=(224, 224, 3))
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image = tf.image.resize(image, (112, 112))
        return image
